# Grid Predictive Maintenance

A production-oriented predictive maintenance system for utility analytics teams. The project starts from the UCI AI4I 2020 dataset and is intentionally structured to scale toward grid telemetry pipelines.

## Problem Statement
Predict likely equipment failures early enough to reduce unplanned downtime, optimize maintenance crews, and improve reliability KPIs.

## Dataset
- **UCI AI4I 2020 Predictive Maintenance Dataset** (starter data provided in `data/raw/ai4i2020.csv`)
- Core features: air/process temperatures, rotational speed, torque, tool wear
- Binary target: failure (rare event)

## Modeling Stack
- **XGBoost** rare-event classifier (with `scale_pos_weight`)
- **LSTM extension scaffold** for multivariate sequence modeling
- **Survival risk stub** for time-to-failure style scoring
- Metrics: ROC-AUC, PR-AUC, F1, expected cost

## Explainability
- SHAP global importance and local single-prediction explanation when SHAP is available
- Safe fallback explanation proxy if SHAP dependency is missing

## Experiment Tracking
- MLflow integration in model training (`src/modeling/train_xgboost.py`)
- Pipeline writes aggregate output to `reports/pipeline_summary.json`

## FastAPI Deployment
- `GET /health`
- `POST /predict`
- `POST /predict_with_explanation`

Example payload:
```json
{
  "air_temperature": 300,
  "process_temperature": 310,
  "rotational_speed": 1500,
  "torque": 40,
  "tool_wear": 120
}
```

## Monitoring
- KS-test drift checks (`monitoring/drift_detection.py`)
- Prediction distribution stability (`monitoring/performance_tracking.py`)
- Baseline feature profile generation (`src/monitoring_report.py`)

## Reports
- `reports/model_defensibility.md`
- `reports/executive_summary.md`
- `reports/cost_impact_analysis.md`

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
