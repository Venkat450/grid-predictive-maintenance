from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ProjectConfig:
    data_path: str = "data/raw/ai4i2020.csv"
    model_path: str = "models/xgboost_model.pkl"
    mlflow_experiment: str = "grid_predictive_maintenance"
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    random_state: int = 42
    test_size: float = 0.2
    risk_threshold_high: float = 0.75
    risk_threshold_medium: float = 0.4
