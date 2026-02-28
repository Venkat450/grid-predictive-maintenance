from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.config import ProjectConfig
from src.data_loader import load_dataset
from src.feature_engineering import FEATURE_COLUMNS, build_features
from src.modeling.evaluate import Metrics, evaluate_binary
from src.preprocessing import preprocess
from src.utils import ensure_parent_dir, setup_logger

logger = setup_logger(__name__)


def _resolve_model(scale_pos_weight: float, random_state: int) -> Any:
    """Prefer XGBoost when installed, fallback to sklearn for constrained environments."""
    try:
        from xgboost import XGBClassifier

        logger.info("Using XGBoost backend")
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
    except Exception as exc:  # pragma: no cover - only used when xgboost is unavailable
        logger.warning("XGBoost unavailable (%s). Falling back to GradientBoostingClassifier.", exc)
        return GradientBoostingClassifier(random_state=random_state)


def train_xgboost(data_path: str, model_path: str, random_state: int = 42) -> dict:
    df = preprocess(load_dataset(data_path))
    feat_df = build_features(df)

    X = feat_df[FEATURE_COLUMNS]
    y = feat_df["failure"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    scale_pos_weight = neg / pos

    model = _resolve_model(scale_pos_weight=scale_pos_weight, random_state=random_state)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:  # pragma: no cover
        raw = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-raw))

    metrics: Metrics = evaluate_binary(np.asarray(y_test), np.asarray(y_proba), threshold=0.5)

    ensure_parent_dir(model_path)
    artifact = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "metrics": asdict(metrics),
        "backend": type(model).__name__,
    }
    joblib.dump(artifact, model_path)

    return {
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "f1": metrics.f1,
        "expected_cost": metrics.expected_cost,
        "scale_pos_weight": scale_pos_weight,
        "backend": type(model).__name__,
    }


def _log_with_mlflow(metrics: dict[str, float | str], data_path: str, cfg: ProjectConfig) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)
        with mlflow.start_run(run_name="xgboost_baseline"):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
                else:
                    mlflow.log_param(key, value)
            mlflow.log_param("data_path", data_path)
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow logging skipped: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=ProjectConfig().data_path)
    parser.add_argument("--model-path", default=ProjectConfig().model_path)
    args = parser.parse_args()

    cfg = ProjectConfig()
    metrics = train_xgboost(args.data_path, args.model_path, random_state=cfg.random_state)
    _log_with_mlflow(metrics, args.data_path, cfg)
    logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    main()
