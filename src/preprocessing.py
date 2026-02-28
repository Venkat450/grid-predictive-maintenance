from __future__ import annotations

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in data.columns:
        if data[col].dtype.kind in {"i", "f"}:
            data[col] = data[col].fillna(data[col].median())
    return data
