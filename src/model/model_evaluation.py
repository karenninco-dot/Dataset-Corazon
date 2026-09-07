"""Model evaluation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a trained pipeline on the test set.

    Args:
        model: Fitted Pipeline (preprocessor + classifier).
        x_test: Test features.
        y_test: Test target.

    Returns:
        A dict with accuracy, precision, recall and f1 — the same
        metrics reported in notebook 06.ModelSelection. Recall gets
        special attention for this problem, since a false negative
        (missing a patient who actually has heart disease) is more
        costly than a false positive.
    """
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
