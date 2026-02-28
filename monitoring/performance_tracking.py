from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictionStats:
    mean_probability: float
    std_probability: float
    high_risk_rate: float


def summarize_predictions(probabilities, high_risk_threshold: float = 0.75) -> PredictionStats:
    probs = list(probabilities)
    if not probs:
        return PredictionStats(0.0, 0.0, 0.0)
    mean = sum(probs) / len(probs)
    var = sum((p - mean) ** 2 for p in probs) / len(probs)
    high_rate = sum(p >= high_risk_threshold for p in probs) / len(probs)
    return PredictionStats(mean, var ** 0.5, high_rate)
