import pytest

pytest.importorskip("pandas")

from src.data_loader import load_dataset
from src.feature_engineering import build_features


def test_load_and_feature_build():
    df = load_dataset("data/raw/ai4i2020.csv")
    assert "failure" in df.columns
    feat_df = build_features(df)
    assert "temp_diff" in feat_df.columns
    assert len(feat_df) == len(df)
