"""
app.py
------
Flask REST API wrapping the hierarchical two-stage XAI Prescriptive
Cybersecurity pipeline, plus static hosting for the frontend dashboard.

Stage 1 (binary anomaly gate) runs on every request. Stage 2 (attack-type
categorizer) only runs if Stage 1's P(anomaly) clears the decision
threshold -- so benign port-80/443 traffic is never forced into an attack
subtype it doesn't belong to.

Endpoints
---------
GET  /                          -> serves the frontend
GET  /api/schema                -> field defs for the data-entry form
GET  /api/status                -> whether a model is currently trained
POST /api/dataset/upload        -> multipart CSV upload of a real dataset
POST /api/train                 -> {threshold?} train both stages
GET  /api/metrics               -> last training metrics (stage 1 + stage 2)
GET  /api/feature-importance    -> global feature importance bars (stage 2)
POST /api/predict                -> single-record inference (the whole point)
"""

import os
import io
import threading
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import joblib

import data_pipeline as dp
import train_ensemble as te
import explainability as xp
import prescriptive_engine as pe

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
MODEL_STORE = os.path.join(os.path.dirname(__file__), "model_store")
MODEL_PATH = os.path.join(MODEL_STORE, "latest.joblib")
os.makedirs(MODEL_STORE, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

DEFAULT_THRESHOLD = 0.5

STATE = {
    "df": None,
    "dataset_source": None,
    "class_distribution": None,

    "stage1_model": None, "stage1_label_encoder": None,
    "stage2_model": None, "stage2_label_encoder": None,
    "preprocessor": None,
    "feature_names": None,
    "threshold": DEFAULT_THRESHOLD,

    "background_sample": None,  # small matrix of preprocessed Normal rows, for differential SHAP
    "background_vector": None,  # per-feature median of the above, for occlusion fallback

    "metrics": None,          # Stage 2 (multiclass) evaluation
    "stage1_metrics": None,   # Stage 1 (binary gate) evaluation
}
LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Model persistence (pre-caching so a server restart doesn't force a retrain)
# ---------------------------------------------------------------------------

def _persist_model():
    try:
        joblib.dump({
            "stage1_model": STATE["stage1_model"], "stage1_label_encoder": STATE["stage1_label_encoder"],
            "stage2_model": STATE["stage2_model"], "stage2_label_encoder": STATE["stage2_label_encoder"],
            "preprocessor": STATE["preprocessor"], "feature_names": STATE["feature_names"],
            "threshold": STATE["threshold"],
            "background_sample": STATE["background_sample"], "background_vector": STATE["background_vector"],
            "metrics": STATE["metrics"], "stage1_metrics": STATE["stage1_metrics"],
            "dataset_source": STATE["dataset_source"], "class_distribution": STATE["class_distribution"],
        }, MODEL_PATH)
    except Exception:
        pass  # persistence is best-effort; the in-memory model is still usable


def _load_cached_model():
    if not os.path.exists(MODEL_PATH):
        return
    try:
        cached = joblib.load(MODEL_PATH)
        STATE.update(cached)
    except Exception:
        pass  # a corrupt/incompatible cache should never block startup


_load_cached_model()


# ---------------------------------------------------------------------------
# Input validation guardrails
# ---------------------------------------------------------------------------

def _sanitize_record(body: dict) -> dict:
    """
    Coerces/clamps an incoming flow record before it reaches the model
    transformers: numeric fields are cast to numbers (bad input -> default,
    not a 500), ports are clamped to the valid range, durations/byte/packet
    counts can't be negative, and categorical fields are lowercased and
    left as-is for the encoder (OneHotEncoder(handle_unknown="ignore")
    tolerates a genuinely unrecognized category without erroring).
    """
    clean = dict(body) if isinstance(body, dict) else {}

    for col in dp.BASE_NUMERIC_COLS:
        if col not in clean:
            continue
        try:
            val = float(clean[col])
        except (TypeError, ValueError):
            val = dp.FEATURE_DEFAULTS[col]
        if col in ("src_port", "dst_port"):
            val = max(0, min(65535, int(val)))
        else:
            val = max(0.0, val)
        clean[col] = val

    for col in dp.CATEGORICAL_COLS:
        if col in clean and clean[col] is not None:
            clean[col] = str(clean[col]).strip().lower()

    return clean


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@app.route("/api/schema")
def schema():
    return jsonify({
        "numeric_fields": dp.BASE_NUMERIC_COLS,
        "categorical_fields": dp.CATEGORICAL_OPTIONS,
        "defaults": dp.FEATURE_DEFAULTS,
        "attack_types": dp.ATTACK_TYPES,
        "severity_weights": pe.SEVERITY_WEIGHTS,
        "threshold": STATE["threshold"],
    })


@app.route("/api/status")
def status():
    with LOCK:
        return jsonify({
            "dataset_loaded": STATE["df"] is not None,
            "dataset_rows": None if STATE["df"] is None else int(len(STATE["df"])),
            "dataset_source": STATE["dataset_source"],
            "model_trained": STATE["stage1_model"] is not None and STATE["stage2_model"] is not None,
            "threshold": STATE["threshold"],
        })


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@app.route("/api/dataset/upload", methods=["POST"])
def dataset_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file' in the request."}), 400
    f = request.files["file"]
    try:
        raw_df = pd.read_csv(io.BytesIO(f.read()))
        df = dp.clean_data(raw_df)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {e}"}), 400

    if dp.TARGET_TYPE not in df.columns:
        return jsonify({"error": f"CSV must contain a '{dp.TARGET_TYPE}' column."}), 400

    with LOCK:
        STATE["df"] = df
        STATE["dataset_source"] = f"uploaded ({f.filename}, {len(df)} rows)"
        STATE["class_distribution"] = df["type"].value_counts().to_dict()

    return jsonify({
        "message": f"Loaded {len(df)} rows from {f.filename}.",
        "rows": int(len(df)),
        "class_distribution": STATE["class_distribution"],
        "preview": df.head(10).to_dict(orient="records"),
    })


# ---------------------------------------------------------------------------
# Train (both stages)
# ---------------------------------------------------------------------------

@app.route("/api/train", methods=["POST"])
def train():
    with LOCK:
        df = STATE["df"]
    if df is None:
        return jsonify({"error": "No dataset loaded. Upload one first."}), 400

    body = request.get_json(silent=True) or {}
    try:
        threshold = float(body.get("threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        threshold = DEFAULT_THRESHOLD
    threshold = max(0.01, min(0.99, threshold))

    (X_train, X_test, y_label_train, y_label_test,
     y_type_train, y_type_test, preprocessor, feature_names,
     X_train_raw, X_test_raw) = dp.prepare_dataset_two_stage(df)

    stage1_model, stage1_encoder, stage2_model, stage2_encoder = te.train_two_stage(
        X_train, y_label_train, y_type_train
    )

    stage1_metrics = te.evaluate_binary(stage1_model, stage1_encoder, X_test, y_label_test, threshold)

    anomaly_test_mask = np.asarray(y_label_test) == 1
    if anomaly_test_mask.any():
        stage2_metrics = te.evaluate(
            stage2_model, stage2_encoder,
            X_test[anomaly_test_mask], y_type_test[anomaly_test_mask],
        )
    else:
        stage2_metrics = {
            "accuracy": None, "macro_f1": None, "roc_auc_ovr_macro": None, "pr_auc_macro": None,
            "confusion_matrix": [], "confusion_matrix_labels": [], "classification_report": {},
            "backend": te.get_backend_info(),
        }

    importance = te.global_feature_importance(stage2_model, feature_names)

    background_sample = dp.normal_background_sample(X_train, y_type_train, max_rows=50)
    background_vector = (
        np.median(background_sample, axis=0) if background_sample is not None
        else np.zeros(len(feature_names))
    )

    with LOCK:
        STATE["stage1_model"], STATE["stage1_label_encoder"] = stage1_model, stage1_encoder
        STATE["stage2_model"], STATE["stage2_label_encoder"] = stage2_model, stage2_encoder
        STATE["preprocessor"], STATE["feature_names"] = preprocessor, feature_names
        STATE["threshold"] = threshold
        STATE["background_sample"], STATE["background_vector"] = background_sample, background_vector
        STATE["metrics"], STATE["stage1_metrics"] = stage2_metrics, stage1_metrics

    _persist_model()

    return jsonify({
        "message": "Training complete (Stage 1 anomaly gate + Stage 2 attack categorizer).",
        "threshold": threshold,
        "metrics": stage2_metrics,
        "stage1_metrics": stage1_metrics,
        "global_feature_importance": importance,
    })


@app.route("/api/metrics")
def metrics_endpoint():
    with LOCK:
        if STATE["metrics"] is None:
            return jsonify({"error": "Model not trained yet."}), 400
        importance = te.global_feature_importance(STATE["stage2_model"], STATE["feature_names"])
        return jsonify({
            "metrics": STATE["metrics"],
            "stage1_metrics": STATE["stage1_metrics"],
            "global_feature_importance": importance,
            "class_distribution": STATE["class_distribution"],
            "dataset_source": STATE["dataset_source"],
            "threshold": STATE["threshold"],
        })


@app.route("/api/feature-importance")
def feature_importance_endpoint():
    with LOCK:
        if STATE["stage2_model"] is None:
            return jsonify({"error": "Model not trained yet."}), 400
        return jsonify(te.global_feature_importance(STATE["stage2_model"], STATE["feature_names"]))


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    with LOCK:
        stage1_model = STATE["stage1_model"]
        stage1_encoder = STATE["stage1_label_encoder"]
        stage2_model = STATE["stage2_model"]
        stage2_encoder = STATE["stage2_label_encoder"]
        preprocessor = STATE["preprocessor"]
        feature_names = STATE["feature_names"]
        background_sample = STATE["background_sample"]
        threshold = STATE["threshold"]

    if stage1_model is None or stage2_model is None:
        return jsonify({"error": "Model not trained yet. Call /api/train first."}), 400

    body = request.get_json(silent=True) or {}
    raw_record = _sanitize_record({**dp.FEATURE_DEFAULTS, **body})

    try:
        row_df = dp.finalize_row(raw_record)
        X_row = preprocessor.transform(row_df)
        if hasattr(X_row, "toarray"):
            X_row = X_row.toarray()
        X_row = np.asarray(X_row, dtype=float)
    except Exception as e:
        return jsonify({"error": f"Invalid input record: {e}"}), 400

    # --- Stage 1: binary anomaly gate ---
    stage1_classes = list(stage1_encoder.classes_)
    anomaly_idx = stage1_classes.index(1) if 1 in stage1_classes else int(np.argmax(stage1_classes))
    p_anomaly = float(stage1_model.predict_proba(X_row)[0, anomaly_idx])
    stage1_gate_passed = p_anomaly >= threshold

    if not stage1_gate_passed:
        # Fast path: confidently normal, Stage 2 skipped entirely.
        confidence = 1.0 - p_anomaly
        risk_score = pe.bypass_risk_score(p_anomaly, threshold)
        tier = pe.risk_tier(risk_score)
        diagnosis = (
            "Traffic matches baseline web/network activity. Key normal indicators: "
            "standard connection state, balanced byte ratio, and expected port usage."
        )
        recommendations = pe.generate_recommendations("normal", [], raw_record)

        return jsonify({
            "input_record": raw_record,
            "predicted_class": "normal",
            "confidence": confidence,
            "proba_by_class": {"normal": confidence, "anomaly": p_anomaly},
            "risk_score": risk_score,
            "risk_tier": tier,
            "top_contributors": [],
            "diagnosis": diagnosis,
            "explainer_backend": "stage1-gate (Stage 2 bypassed — below anomaly threshold)",
            "recommendations": recommendations,
            "stage1_gate_passed": False,
            "anomaly_probability": p_anomaly,
            "low_confidence_context": False,
        })

    # --- Stage 2: attack-type categorizer (only reached if Stage 1 flagged anomaly) ---
    proba2 = stage2_model.predict_proba(X_row)[0]
    class_names2 = list(stage2_encoder.classes_)
    proba_by_class = {c: float(p) for c, p in zip(class_names2, proba2)}

    pred_idx = int(np.argmax(proba2))
    predicted_class = class_names2[pred_idx]
    confidence = float(proba2[pred_idx])

    risk_score = pe.compute_risk_score(proba_by_class)
    tier = pe.risk_tier(risk_score)

    contributors, explainer_backend = xp.explain_instance(
        stage2_model, stage2_encoder, X_row, feature_names, predicted_class,
        background_sample=background_sample, top_n=5,
    )
    diagnosis = xp.format_diagnosis(contributors)

    sorted_probs = sorted(proba_by_class.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    contributor_features = [c["feature"] for c in contributors]
    shallow_evidence = len(contributor_features) == 0 or all(
        f.startswith(("dst_port", "src_port", "proto_", "service_", "conn_state_"))
        for f in contributor_features
    )
    low_confidence_context = predicted_class in dp.WEB_LAYER_CLASSES and (margin < 0.15 or shallow_evidence)

    recommendations = pe.generate_recommendations(
        predicted_class, contributors, raw_record, low_confidence_context=low_confidence_context
    )

    return jsonify({
        "input_record": raw_record,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "proba_by_class": dict(sorted(proba_by_class.items(), key=lambda x: -x[1])),
        "risk_score": risk_score,
        "risk_tier": tier,
        "top_contributors": contributors,
        "diagnosis": diagnosis,
        "explainer_backend": explainer_backend,
        "recommendations": recommendations,
        "stage1_gate_passed": True,
        "anomaly_probability": p_anomaly,
        "low_confidence_context": low_confidence_context,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)