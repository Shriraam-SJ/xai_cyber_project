"""
demo_inference.py
------------------
End-to-end CLI test: trains on a real ToN_IoT CSV export, then runs
inference on a single raw flow row and prints:
  - Predicted Class & Confidence
  - Quantitative Risk Index (0-100) & Tier
  - Top Root-Cause Drivers (SHAP / occlusion values)
  - Prescribed Preventive & Recovery Actions

Usage:
    python demo_inference.py --csv path/to/ton_iot.csv
    python demo_inference.py --csv path/to/ton_iot.csv --sample-json my_row.json
"""

import argparse
import json
import numpy as np

import data_pipeline as dp
import train_ensemble as te
import explainability as xp
import prescriptive_engine as pe

DEFAULT_SAMPLE_ROW = {  # looks like a backdoor / reverse-shell style flow
    "duration": 290.37, "src_bytes": 101568, "dst_bytes": 2592, "missed_bytes": 0,
    "src_pkts": 108, "src_ip_bytes": 108064, "dst_pkts": 31, "dst_ip_bytes": 3832,
    "src_port": 4444, "dst_port": 49178, "proto": "tcp", "service": "unknown",
    "conn_state": "OTH",
}


def run_pipeline(df):
    (X_train, X_test, y_train, y_test, preprocessor,
     feature_names, X_train_raw, X_test_raw) = dp.prepare_dataset(df)
    model, label_encoder = te.train(X_train, y_train)
    metrics = te.evaluate(model, label_encoder, X_test, y_test)
    return model, label_encoder, preprocessor, feature_names, metrics


def predict_one(model, label_encoder, preprocessor, feature_names, raw_record: dict):
    import pandas as pd
    row_df = pd.DataFrame([{**dp.FEATURE_DEFAULTS, **raw_record}])[dp.ALL_INPUT_COLS]
    X_row = preprocessor.transform(row_df)
    if hasattr(X_row, "toarray"):
        X_row = X_row.toarray()
    X_row = np.asarray(X_row)

    proba = model.predict_proba(X_row)[0]
    class_names = list(label_encoder.classes_)
    proba_by_class = {c: float(p) for c, p in zip(class_names, proba)}

    pred_idx = int(np.argmax(proba))
    predicted_class = class_names[pred_idx]
    confidence = float(proba[pred_idx])

    risk_score = pe.compute_risk_score(proba_by_class)
    tier = pe.risk_tier(risk_score)

    contributors, backend = xp.explain_instance(
        model, label_encoder, X_row, feature_names, predicted_class, top_n=5)
    diagnosis = xp.format_diagnosis(contributors)

    recommendations = pe.generate_recommendations(predicted_class, contributors, raw_record)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "proba_by_class": proba_by_class,
        "risk_score": risk_score,
        "risk_tier": tier,
        "top_contributors": contributors,
        "diagnosis": diagnosis,
        "explainer_backend": backend,
        "recommendations": recommendations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to a real ToN_IoT CSV export")
    parser.add_argument("--sample-json", type=str, default=None,
                         help="Optional path to a JSON file with a single flow record to run inference on. "
                              "Defaults to a built-in backdoor-like sample.")
    args = parser.parse_args()

    print("=" * 70)
    print(" XAI PRESCRIPTIVE CYBERSECURITY SYSTEM — DEMO INFERENCE")
    print("=" * 70)

    print(f"Loading dataset: {args.csv}")
    df = dp.load_csv(args.csv)
    print(f"Loaded {len(df)} rows, {df['type'].nunique()} classes.")

    print("Training ensemble...")
    model, label_encoder, preprocessor, feature_names, metrics = run_pipeline(df)
    print(f"Backend: {te.get_backend_info()}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f} "
          f"| ROC-AUC(ovr): {metrics['roc_auc_ovr_macro']}")

    if args.sample_json:
        with open(args.sample_json) as f:
            sample_row = json.load(f)
    else:
        sample_row = DEFAULT_SAMPLE_ROW

    print("\nRunning inference on sample flow record:")
    print(json.dumps(sample_row, indent=2))

    result = predict_one(model, label_encoder, preprocessor, feature_names, sample_row)

    print("\n--- RESULT ------------------------------------------------------")
    print(f"Predicted Class : {result['predicted_class'].upper()}  (confidence {result['confidence']:.1%})")
    print(f"Risk Index      : {result['risk_score']:.1f} / 100  [{result['risk_tier']}]")
    print(f"Root-Cause      : {result['diagnosis']}  (via {result['explainer_backend']})")
    print("Technical Actions:")
    for a in result["recommendations"]["technical_actions"]:
        print(f"  - {a}")
    print("Governance Controls:")
    for c in result["recommendations"]["governance_controls"]:
        print(f"  - [{c['framework']}] {c['control']} — {c['name']}")
    print("-" * 70)


if __name__ == "__main__":
    main()