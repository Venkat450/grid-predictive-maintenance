from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.config import ProjectConfig
from src.data_loader import load_dataset
from src.explainability.shap_analysis import compute_shap_summary
from src.feature_engineering import FEATURE_COLUMNS, build_features
from src.modeling.train_xgboost import train_xgboost
from src.monitoring_report import build_monitoring_baseline
from src.preprocessing import preprocess
from src.utils import setup_logger

logger = setup_logger(__name__)


def run_pipeline(data_path: str, model_path: str) -> dict:
    metrics = train_xgboost(data_path=data_path, model_path=model_path)
    artifact = joblib.load(model_path)

    df = preprocess(load_dataset(data_path))
    X = build_features(df)[FEATURE_COLUMNS]

    shap_summary = compute_shap_summary(artifact["model"], X)
    monitoring = build_monitoring_baseline(X)

    outputs = {
        "metrics": metrics,
        "top_features": shap_summary.head(5).to_dict(),
        "monitoring_baseline": monitoring,
    }

    out_path = Path("reports/pipeline_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, indent=2))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=ProjectConfig().data_path)
    parser.add_argument("--model-path", default=ProjectConfig().model_path)
    args = parser.parse_args()

    outputs = run_pipeline(args.data_path, args.model_path)
    logger.info("Pipeline complete. Summary written to reports/pipeline_summary.json")
    logger.info("Top feature drivers: %s", outputs["top_features"])


if __name__ == "__main__":
    main()
