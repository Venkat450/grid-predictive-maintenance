# Grid Predictive Maintenance

A production-oriented, **predictive maintenance system** for utility analytics teams. This repository starts with the **AI4I 2020 Predictive Maintenance Dataset** (UCI), and is extensible to open power grid sensor streams.

## 1) Problem Statement
Utilities need to predict equipment failures before they happen to reduce unplanned downtime, avoid cascading incidents, and optimize maintenance schedules.

## 2) Dataset Description
Primary starter dataset:
- **UCI AI4I 2020 Predictive Maintenance Dataset** (`data/raw/ai4i2020.csv`)
- Core features: air/process temperature, rotational speed, torque, tool wear
- Target: machine failure (rare event)

The codebase is designed for extension to external telemetry in `data/external/`.

## 3) Modeling Approach
- **XGBoost** baseline for tabular rare-event classification
- **LSTM (optional)** for temporal multivariate sequences
- **Survival-style risk scoring stub** for time-to-failure extensibility
- Imbalance handling through `scale_pos_weight`

## 4) Performance Metrics
- ROC-AUC
- PR-AUC
- F1-score
- Cost-sensitive expected impact (false negative weighted)

## 5) Explainability
- SHAP global and local explanations
- Optional interaction-level interpretation

## 6) Deployment Architecture
- `FastAPI` service in `api/`
- `POST /predict` for failure probability and risk level
- `GET /health` for runtime checks
- Artifacts persisted in `models/`

## 7) Business Impact Simulation
See `reports/cost_impact_analysis.md` for a practical expected-value framework to estimate avoided downtime and intervention costs.

## 8) Future Improvements
- Live streaming inference with Kafka
- Online drift adaptation
- Calibration and threshold optimization by asset class
- Utility-specific regulatory reporting

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline --data-path data/raw/ai4i2020.csv
uvicorn api.app:app --reload
pytest -q
```

## Repository Structure
See root tree for modular code organization across data, modeling, explainability, API serving, monitoring, reports, and tests.
pytest -q
uvicorn api.app:app --reload
```

## Training and Tracking
```bash
python -m src.modeling.train_xgboost --data-path data/raw/ai4i2020.csv
python -m src.pipeline --data-path data/raw/ai4i2020.csv
```

MLflow tracking URI defaults to local file store (`mlruns/`) and can be overridden with `MLFLOW_TRACKING_URI`.
