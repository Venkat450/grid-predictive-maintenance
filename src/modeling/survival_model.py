from __future__ import annotations

import numpy as np


def surrogate_survival_risk(failure_probability: float, horizon_days: int = 30) -> float:
    """Simple placeholder: converts failure probability to horizon-adjusted risk score."""
    daily_risk = min(max(failure_probability / max(horizon_days, 1), 0.0), 1.0)
    return float(1 - np.exp(-daily_risk * horizon_days))
