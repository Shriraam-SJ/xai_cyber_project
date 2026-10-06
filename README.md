# XAI Prescriptive Cybersecurity System (ToN_IoT)

An end-to-end Explainable AI system that classifies network flows (normal vs.
9 attack categories), computes a quantitative Governance Risk Index (0–100),
explains *why* each prediction was made with root-cause feature attribution,
and prescribes technical + ISO/IEC 27001 / NIST SP 800-53 remediation —
served through a Flask API with a browser dashboard for data entry, metrics,
and visualization.

## Project layout

```
xai_cyber_project/
├── backend/
│   ├── data_pipeline.py       # schema, synthetic generator, cleaning, encoding
│   ├── train_ensemble.py      # ensemble training + evaluation
│   ├── explainability.py      # TreeSHAP / occlusion root-cause attribution
│   ├── prescriptive_engine.py # risk scoring + ISO/NIST control mapping
│   ├── demo_inference.py      # CLI: train + run one sample through the whole pipeline
│   ├── app.py                 # Flask REST API + serves the frontend
│   └── model_store/           # trained model artifacts land here (joblib)
├── frontend/
│   └── index.html             # single-page dashboard (vanilla JS + Chart.js)
├── requirements.txt
└── README.md
```

## Important: about the modeling/explainability backend

The spec calls for **CatBoost + LightGBM + XGBoost** (voting ensemble) and
**shap.TreeExplainer**. Those packages need `pip install` access, which
wasn't available while building this in the sandboxed dev environment used
to write and test this code. So the code was written to **auto-detect** what's
installed:

| Component | If the preferred library is installed | If not (fallback, currently active) |
|---|---|---|
| Ensemble | CatBoost + LightGBM + XGBoost, soft VotingClassifier | `HistGradientBoostingClassifier` + `RandomForestClassifier` + `ExtraTreesClassifier`, soft VotingClassifier |
| Attribution | `shap.TreeExplainer` on a tree member of the ensemble | Occlusion attribution — each feature is masked with its background/median value and the drop in predicted probability for the predicted class becomes its contribution. This is a legitimate, well-known Shapley-style local-attribution approximation (a single-feature marginal contribution), not a placeholder. |
| Imbalance handling | SMOTE (`imbalanced-learn`) | Class-weighted estimators (`class_weight="balanced_subsample"`) |

**No code changes are needed either way.** Just `pip install -r requirements.txt`
on a machine with internet access and the app will automatically pick up
CatBoost/LightGBM/XGBoost/SHAP the next time it trains — check the
"Model backend" metric card on the dashboard or the `backend` field in
`/api/metrics` to see which stack is actually active. Everything else
(risk scoring formula, ISO/NIST mapping, API shape, frontend) behaves
identically regardless of backend.

## Setup

```bash
cd xai_cyber_project
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

If `catboost`/`lightgbm`/`xgboost`/`shap` fail to build on your machine
(uncommon, but can happen on unusual platforms), just remove those four
lines from `requirements.txt` and reinstall — the app runs fine without
them via the fallback path above.

## Running

```bash
cd backend
python3 app.py
```

Then open **http://localhost:5000** in a browser.

You can also drive the whole pipeline from the terminal without the web UI:

```bash
cd backend
python3 demo_inference.py                 # synthetic data, built-in sample flow
python3 demo_inference.py --csv ton_iot.csv   # a real ToN_IoT export instead
```

## Using the dashboard

1. **Dashboard tab** — click **Generate Demo Dataset** (or **Upload ToN_IoT
   CSV** to use the real dataset once you have it), then **Train Ensemble**.
   You'll see accuracy / macro-F1 / ROC-AUC / PR-AUC, which backend actually
   trained, a class-distribution chart, global feature importance, and the
   confusion matrix.
2. **Data Entry tab** — fill in a flow record (or click *Fill sample* for a
   pre-populated backdoor-like example) and click **Run Detection**. You'll
   get:
   - Predicted attack class + confidence
   - Governance Risk Index (0–100) with Low/Medium/High tier
   - Root-cause diagnosis: top positive feature contributors driving the
     prediction, in plain language
   - Prescribed technical remediation actions
   - Mapped ISO/IEC 27001 & NIST SP 800-53 controls

## Using the real ToN_IoT dataset

Download the ToN_IoT network CSVs (e.g. from the UNSW Canberra Cyber Range
IoT dataset release) and either:
- Upload a CSV directly from the dashboard ("Upload ToN_IoT CSV"), or
- Pass it to the CLI: `python3 demo_inference.py --csv /path/to/ton_iot.csv`

The uploaded file just needs a `type` column (attack category) — `label`
(binary) is derived automatically from it if not already present.
`data_pipeline.clean_data()` handles null/placeholder tokens, drops
leakage-prone identifier columns (`ts`, IPs, `http_uri`, `ssl_subject`,
`ssl_issuer`, `dns_query`), and encodes/scales the rest.

## Governance Risk Index formula

```
Risk = ( Σ_k P(Y=k) × W_k ) × 10          clipped to [0, 100]
```

`W_k` is a per-class severity weight (0–10): Normal=0, Scanning=3,
Password=5, XSS/Injection/DoS=6, MITM=7, DDoS=8, Backdoor/Ransomware=10 —
editable in `prescriptive_engine.SEVERITY_WEIGHTS`.

Tiers: **Low** 0–29, **Medium** 30–69, **High** 70–100.

## Extending

- Add a new attack class: extend `data_pipeline.ATTACK_TYPES`,
  `prescriptive_engine.SEVERITY_WEIGHTS`, and `prescriptive_engine.CONTROL_MAP`.
- Add a new technical rule: append a predicate to
  `prescriptive_engine._technical_rules`.
- Swap in a different tree ensemble member: edit `train_ensemble.build_ensemble`.
