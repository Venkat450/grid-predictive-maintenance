from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_shap_summary(model: Any, X: pd.DataFrame) -> pd.Series:
    """Compute mean absolute SHAP contribution per feature.

    Falls back to permutation-style importance proxy when SHAP is unavailable.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    except Exception:
        baseline = model.predict_proba(X)[:, 1]
        scores: dict[str, float] = {}
        for col in X.columns:
            shuffled = X.copy()
            shuffled[col] = np.random.permutation(shuffled[col].to_numpy())
            perturbed = model.predict_proba(shuffled)[:, 1]
            scores[col] = float(np.mean(np.abs(baseline - perturbed)))
        return pd.Series(scores).sort_values(ascending=False)


def explain_single_prediction(model: Any, row: pd.DataFrame) -> dict[str, float]:
    """Return per-feature local contribution for one row."""
    if len(row) != 1:
        raise ValueError("row must contain exactly one sample")

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        values = shap_values[0]
        return {col: float(val) for col, val in zip(row.columns, values)}
    except Exception:
        # proxy local explanation: directional sensitivity from small perturbation
        base = float(model.predict_proba(row)[:, 1][0])
        out: dict[str, float] = {}
        for col in row.columns:
            delta = max(abs(float(row[col].iloc[0])) * 0.01, 0.01)
            probe = row.copy()
            probe[col] = probe[col] + delta
            pert = float(model.predict_proba(probe)[:, 1][0])
            out[col] = pert - base
        return out
