"""
train_ensemble.py
------------------
Multi-stage ensemble modeling for attack-type classification, plus the
hierarchical two-stage architecture used by app.py:

    Stage 1 (Binary Anomaly Gate): label 0 vs 1. If P(anomaly) is below the
    decision threshold, the sample is classified Normal immediately and
    Stage 2 is skipped entirely (fast path, low risk, no attack-subtype
    guess forced onto benign traffic).

    Stage 2 (Multi-Class Attack Categorizer): trained ONLY on the anomalous
    subset (label == 1) and only ever runs when Stage 1 flags a sample as
    anomalous, so it never has to discriminate "attack subtype vs normal"
    on the confounded port/service signature -- only "which attack".

Preferred stack (per spec): CatBoost + LightGBM + XGBoost combined via a
soft VotingClassifier. Auto-detected: if unavailable, falls back to a
scikit-learn ensemble (HistGradientBoosting + RandomForest + ExtraTrees)
so the system still runs end-to-end. The active backend is reported via
get_backend_info() and surfaced in the API/UI.

Class imbalance: rather than oversampling (SMOTE), which can distort the
learned decision boundary for exactly the confounded classes this system
is trying to fix, every fit call uses balanced *sample weights*
(sklearn.utils.class_weight.compute_sample_weight("balanced", y)). This
lets the real class proportions in the data stay intact (so e.g.
dst_port=80 + conn_state=SF keeps being seen mostly in Normal rows, as it
should be) while still giving minority attack classes proportionate
influence on the loss. One weighting mechanism, applied identically
regardless of which boosting backend is active.
"""

import numpy as np
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, precision_score,
    recall_score, confusion_matrix, classification_report, accuracy_score,
)

RANDOM_STATE = 42

_BACKEND = {"boosting_stack": "sklearn-fallback", "balancing": "sample_weight=balanced"}

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


def get_backend_info():
    return dict(_BACKEND)


def build_ensemble():
    """
    Builds a soft-voting ensemble. Uses CatBoost + LightGBM + XGBoost when
    all three are installed (matches the project spec exactly); otherwise
    falls back to a comparably diverse tree-ensemble stack built entirely
    on scikit-learn. Class balancing is handled uniformly via sample_weight
    at fit time (see train()), not via per-estimator class_weight kwargs,
    so the same code path works for both stacks.
    """
    if _HAS_XGB and _HAS_LGBM and _HAS_CATBOOST:
        _BACKEND["boosting_stack"] = "catboost+lightgbm+xgboost"
        estimators = [
            ("catboost", CatBoostClassifier(
                iterations=300, depth=6, verbose=False, random_state=RANDOM_STATE)),
            ("lightgbm", LGBMClassifier(
                n_estimators=300, max_depth=6, random_state=RANDOM_STATE, verbosity=-1)),
            ("xgboost", XGBClassifier(
                n_estimators=300, max_depth=6, random_state=RANDOM_STATE,
                eval_metric="mlogloss", use_label_encoder=False)),
        ]
    else:
        _BACKEND["boosting_stack"] = "sklearn-fallback (HistGradientBoosting + RandomForest + ExtraTrees)"
        estimators = [
            ("hgb", HistGradientBoostingClassifier(
                max_depth=8, max_iter=300, random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)),
            ("etc", ExtraTreesClassifier(
                n_estimators=300, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)),
        ]

    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=None)


def train(X_train, y_train_labels, sample_weight=None):
    """
    y_train_labels: array of class labels (strings for the attack-type
    model, or 0/1 ints for the binary anomaly gate).
    Returns (fitted_ensemble, fitted_label_encoder).
    """
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)

    if sample_weight is None:
        sample_weight = compute_sample_weight("balanced", y_train)

    model = build_ensemble()
    try:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except TypeError:
        # Extremely defensive fallback in case some installed estimator
        # version doesn't accept sample_weight -- train unweighted rather
        # than crash.
        model.fit(X_train, y_train)

    return model, label_encoder


def train_two_stage(X_train, y_label_train, y_type_train):
    """
    Trains Stage 1 (binary anomaly gate) on the full training set, then
    Stage 2 (attack-type categorizer) on only the rows Stage 1's ground
    truth marks anomalous. Both stages consume the same preprocessed
    feature space (caller is expected to have used
    data_pipeline.prepare_dataset_two_stage, which fits one shared
    preprocessor for both).

    Returns: stage1_model, stage1_label_encoder, stage2_model, stage2_label_encoder
    """
    stage1_model, stage1_label_encoder = train(X_train, y_label_train)

    anomaly_mask = np.asarray(y_label_train) == 1
    X_train_s2 = X_train[anomaly_mask]
    y_type_train_s2 = np.asarray(y_type_train)[anomaly_mask]
    stage2_model, stage2_label_encoder = train(X_train_s2, y_type_train_s2)

    return stage1_model, stage1_label_encoder, stage2_model, stage2_label_encoder


def evaluate(model, label_encoder, X_test, y_test_labels):
    """Multi-class evaluation (used for Stage 2, on the anomalous test subset)."""
    y_test = label_encoder.transform(y_test_labels)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    class_names = list(label_encoder.classes_)
    present = sorted(set(y_test) | set(y_pred))

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = None

    try:
        y_test_bin = np.zeros((len(y_test), len(class_names)))
        y_test_bin[np.arange(len(y_test)), y_test] = 1
        pr_auc = average_precision_score(y_test_bin, y_proba, average="macro")
    except ValueError:
        pr_auc = None

    cm = confusion_matrix(y_test, y_pred, labels=present)
    report = classification_report(
        y_test, y_pred, labels=present,
        target_names=[class_names[i] for i in present],
        output_dict=True, zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "roc_auc_ovr_macro": float(roc_auc) if roc_auc is not None else None,
        "pr_auc_macro": float(pr_auc) if pr_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [class_names[i] for i in present],
        "classification_report": report,
        "backend": get_backend_info(),
    }


def evaluate_binary(model, label_encoder, X_test, y_label_test, threshold: float = 0.5):
    """
    Stage-1 evaluation: precision/recall/F1/ROC-AUC/confusion matrix for
    the anomaly gate at the given decision threshold. High precision here
    matters more than high recall -- a false "anomaly" just costs a Stage-2
    lookup, but a false "normal" means Stage 2 never runs at all.
    """
    y_test = np.asarray(y_label_test).astype(int)
    classes = list(label_encoder.classes_)
    anomaly_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))

    proba_anomaly = model.predict_proba(X_test)[:, anomaly_idx]
    y_pred = (proba_anomaly >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, proba_anomaly)
    except ValueError:
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["normal", "anomaly"],
        "backend": get_backend_info(),
    }


def global_feature_importance(model, feature_names, top_n=15):
    """
    Averages feature_importances_ across ensemble members that expose it
    (all current backends do), giving a global summary comparable in spirit
    to a SHAP beeswarm's magnitude axis.
    """
    importances = []
    fitted = getattr(model, "estimators_", [])
    for est in fitted:
        if hasattr(est, "feature_importances_"):
            importances.append(np.asarray(est.feature_importances_))
    if not importances:
        return []

    avg = np.mean(importances, axis=0)
    order = np.argsort(avg)[::-1][:top_n]
    return [{"feature": feature_names[i], "importance": float(avg[i])} for i in order]