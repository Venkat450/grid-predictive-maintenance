import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.modeling.evaluate import evaluate_binary


def test_evaluate_binary_outputs_reasonable_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate_binary(y_true, y_proba, threshold=0.5)
    assert 0.0 <= m.roc_auc <= 1.0
    assert 0.0 <= m.pr_auc <= 1.0
    assert 0.0 <= m.f1 <= 1.0
    assert m.expected_cost >= 0
