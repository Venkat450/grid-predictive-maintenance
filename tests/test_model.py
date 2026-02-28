import pytest

pytest.importorskip("xgboost")
pytest.importorskip("mlflow")

from src.modeling.train_xgboost import train_xgboost


def test_train_xgboost(tmp_path):
    model_path = tmp_path / "model.pkl"
    metrics = train_xgboost("data/raw/ai4i2020.csv", str(model_path), random_state=7)
    assert model_path.exists()
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
