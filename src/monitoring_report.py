from __future__ import annotations

import pandas as pd


def build_monitoring_baseline(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=["number"])
    summary = {}
    for col in numeric.columns:
        summary[col] = {
            "mean": float(numeric[col].mean()),
            "std": float(numeric[col].std(ddof=0)),
            "p95": float(numeric[col].quantile(0.95)),
        }
    return summary
