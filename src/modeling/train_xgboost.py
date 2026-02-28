from __future__ import annotations

import argparse
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import ProjectConfig
from src.data_loader import load_dataset
from src.feature_engineering import FEATURE_COLUMNS, build_features
from src.modeling.evaluate import evaluate_binary
from src.preprocessing import preprocess
from src.utils import ensure_parent_dir


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

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary(np.asarray(y_test), np.asarray(y_proba), threshold=0.5)

    ensure_parent_dir(model_path)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, model_path)

    return {
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
        "f1": metrics.f1,
        "expected_cost": metrics.expected_cost,
        "scale_pos_weight": scale_pos_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=ProjectConfig().data_path)
    parser.add_argument("--model-path", default=ProjectConfig().model_path)
    args = parser.parse_args()

    cfg = ProjectConfig()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run(run_name="xgboost_baseline"):
        metrics = train_xgboost(args.data_path, args.model_path, random_state=cfg.random_state)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("data_path", args.data_path)


if __name__ == "__main__":
    main()
