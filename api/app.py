from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from api.schema import PredictionRequest, PredictionResponse
from src.config import ProjectConfig
from src.explainability.shap_analysis import explain_single_prediction
from src.feature_engineering import FEATURE_COLUMNS, build_features

app = FastAPI(title="Grid Predictive Maintenance API", version="0.2.0")
MODEL_PATH = Path(ProjectConfig().model_path)


class _ModelCache:
    artifact = None



def _load_artifact():
    if _ModelCache.artifact is not None:
        return _ModelCache.artifact
    if MODEL_PATH.exists():
        _ModelCache.artifact = joblib.load(MODEL_PATH)
    return _ModelCache.artifact


@app.get("/health")
def health() -> dict:
    artifact = _load_artifact()
    return {
        "status": "ok",
        "model_loaded": artifact is not None,
        "model_path": str(MODEL_PATH),
    }


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


@app.post("/predict_with_explanation")
def predict_with_explanation(payload: PredictionRequest) -> dict:
    artifact = _load_artifact()
    if artifact is None:
        pred = predict(payload)
        return {"prediction": pred.model_dump(), "explanation": {"note": "Train model to enable local explanations."}}

    data = pd.DataFrame([payload.model_dump()])
    feats = build_features(data)[artifact.get("features", FEATURE_COLUMNS)]
    score = float(artifact["model"].predict_proba(feats)[:, 1][0])
    local = explain_single_prediction(artifact["model"], feats)

    return {
        "failure_probability": round(score, 4),
        "top_contributors": dict(sorted(local.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]),
    }
