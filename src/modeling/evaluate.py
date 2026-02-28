from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    f1: float
    expected_cost: float


def evaluate_binary(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_pred = (y_proba >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    expected_cost = 10 * fn + 1 * fp
    return Metrics(
        roc_auc=roc_auc_score(y_true, y_proba),
        pr_auc=average_precision_score(y_true, y_proba),
        f1=f1_score(y_true, y_pred, zero_division=0),
        expected_cost=float(expected_cost),
    )
