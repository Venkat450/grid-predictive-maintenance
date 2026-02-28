from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import shap


def compute_shap_summary(model: Any, X: pd.DataFrame) -> pd.Series:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
