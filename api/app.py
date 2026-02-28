from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI

from api.schema import PredictionRequest, PredictionResponse
from src.config import ProjectConfig
from src.feature_engineering import FEATURE_COLUMNS, build_features

app = FastAPI(title="Grid Predictive Maintenance API", version="0.1.0")
MODEL_PATH = Path(ProjectConfig().model_path)


def _load_artifact():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    artifact = _load_artifact()
    data = pd.DataFrame([payload.model_dump()])
    feats = build_features(data)

    if artifact is None:
        # deterministic fallback in absence of trained model
        score = min(1.0, max(0.0, (payload.torque / 100.0) + (payload.tool_wear / 500.0)))
    else:
        model = artifact["model"]
        feature_list = artifact.get("features", FEATURE_COLUMNS)
        score = float(model.predict_proba(feats[feature_list])[:, 1][0])

    cfg = ProjectConfig()
    if score >= cfg.risk_threshold_high:
        risk = "HIGH"
    elif score >= cfg.risk_threshold_medium:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return PredictionResponse(failure_probability=round(score, 4), risk_level=risk)
