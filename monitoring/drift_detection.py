from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class DriftResult:
    feature: str
    p_value: float
    drift_detected: bool


def ks_drift_report(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05) -> Dict[str, DriftResult]:
    report: Dict[str, DriftResult] = {}
    numeric_features = [c for c in reference.columns if c in current.columns and reference[c].dtype.kind in {"i", "f"}]
    for feature in numeric_features:
        stat = ks_2samp(reference[feature], current[feature])
        report[feature] = DriftResult(feature=feature, p_value=float(stat.pvalue), drift_detected=stat.pvalue < alpha)
    return report
