# Model Defensibility Report

## Business Context
This model estimates near-term machine failure risk to support maintenance triage and reduce unplanned downtime in utility operations.

## Data Lineage
- Source: AI4I 2020 Predictive Maintenance Dataset (UCI)
- Raw file location: `data/raw/ai4i2020.csv`
- Processed with deterministic feature engineering in `src/feature_engineering.py`

## Imbalance Handling Strategy
- Rare-event target addressed with XGBoost `scale_pos_weight = negatives / positives`
- Metrics include PR-AUC and cost-aware evaluation (high FN penalty)

## Validation Framework
- Stratified train/test split
- Multi-metric evaluation: ROC-AUC, PR-AUC, F1, expected cost

## Bias Assessment
- Current bias checks are scoped to technical sensor signals.
- Future enhancement: asset/site-level subgroup parity checks.

## Stability Testing
- Prediction stability summaries in `monitoring/performance_tracking.py`
- Drift tests via KS statistics in `monitoring/drift_detection.py`

## Drift Monitoring Plan
- Monitor feature distribution divergence daily/weekly
- Trigger alerts when p-values fall below threshold across key features

## Retraining Strategy
- Retrain monthly or when drift alert frequency exceeds policy threshold.
- Compare challenger model vs. production with holdout and backtest windows.

## Ethical Considerations
- Model supports operator judgment; never sole authority for shutdown decisions.
- Maintain audit trails for inputs, outputs, and threshold decisions.

## Regulatory Alignment
- Supports explainability and traceability expectations for critical infrastructure analytics workflows.
