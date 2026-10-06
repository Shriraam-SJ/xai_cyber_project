"""
explainability.py
------------------
Root-cause attribution engine.

Preferred method (per spec): shap.TreeExplainer on the best-performing
tree ensemble member, producing local waterfall-style contributions and
a global beeswarm-style summary.

`shap` requires network access to install and may not be present in every
deployment environment. This module auto-detects it:
  - If `shap` is importable, it runs a real TreeExplainer against one of
    the tree-based members of the voting ensemble.
  - If not, it falls back to a model-agnostic **occlusion attribution**:
    for the predicted class, each feature is temporarily replaced with a
    background (median/mode) value and the resulting drop in predicted
    probability is taken as that feature's contribution. This is a
    well-established, simple approximation to Shapley-style local
    attribution (a single-feature-coalition marginal contribution) and
    works regardless of which model backend trained the ensemble.

Both paths return the same structure so the API/UI never need to know
which one ran (the active method is included in the response for
transparency).
"""

import numpy as np

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def get_explainer_backend():
    return "shap.TreeExplainer" if _HAS_SHAP else "occlusion-attribution (shap unavailable)"


def _pick_tree_member(model):
    """Grab a tree-based fitted sub-estimator from the voting ensemble for SHAP."""
    fitted = getattr(model, "estimators_", [])
    for est in fitted:
        cls_name = est.__class__.__name__
        if any(k in cls_name for k in ["Forest", "XGB", "LGBM", "CatBoost", "GradientBoosting", "ExtraTrees"]):
            return est
    return fitted[0] if fitted else None


def _shap_explain(model, X_row, feature_names, class_index, top_n):
    tree_model = _pick_tree_member(model)
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_row)

    # Normalize output shape across shap versions / model types.
    if isinstance(shap_values, list):
        vals = np.asarray(shap_values[class_index])[0]
    else:
        arr = np.asarray(shap_values)
        vals = arr[0, :, class_index] if arr.ndim == 3 else arr[0]

    return _top_contributors(vals, feature_names, top_n)


def _occlusion_explain(model, X_row, feature_names, class_index, background, top_n):
    base_proba = model.predict_proba(X_row)[0, class_index]
    contributions = np.zeros(X_row.shape[1])

    for i in range(X_row.shape[1]):
        occluded = X_row.copy()
        occluded[0, i] = background[i]
        occluded_proba = model.predict_proba(occluded)[0, class_index]
        contributions[i] = base_proba - occluded_proba  # drop caused by removing feature i

    return _top_contributors(contributions, feature_names, top_n)


def _top_contributors(values, feature_names, top_n):
    order = np.argsort(values)[::-1]
    out = []
    for i in order:
        if values[i] <= 0:
            continue
        out.append({"feature": feature_names[i], "phi": float(values[i])})
        if len(out) >= top_n:
            break
    return out


def explain_instance(model, label_encoder, X_row, feature_names, predicted_class_name,
                      background=None, top_n=5):
    """
    X_row: 1 x n_features preprocessed array for a single sample.
    background: 1D array of per-feature background values (medians), used
                only by the occlusion fallback. Defaults to zeros
                (reasonable since numeric features are RobustScaler-scaled
                around 0 and categoricals are one-hot).
    Returns: list of {"feature": str, "phi": float}, top_n positive
             contributors pushing toward the predicted class, plus which
             attribution backend produced them.
    """
    class_index = list(label_encoder.classes_).index(predicted_class_name)

    if _HAS_SHAP:
        try:
            contributors = _shap_explain(model, X_row, feature_names, class_index, top_n)
            if contributors:
                return contributors, "shap.TreeExplainer"
        except Exception:
            pass  # fall through to occlusion method

    if background is None:
        background = np.zeros(X_row.shape[1])
    contributors = _occlusion_explain(model, X_row, feature_names, class_index, background, top_n)
    return contributors, "occlusion-attribution"


def format_diagnosis(contributors):
    """Human-readable one-liner, e.g. the example in the spec."""
    if not contributors:
        return "No strong positive drivers identified for this class."
    parts = [f"{c['feature']} ({c['phi']:.2f})" for c in contributors]
    return "Elevated risk driven by: " + ", ".join(parts)
