"""Unit tests for src.model.model_evaluation."""

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.model.model_evaluation import evaluate_model

EXPECTED_ACCURACY = 0.5
EXPECTED_RECALL = 1.0


def test_evaluate_model_returns_expected_metric_keys_and_values() -> None:
    """evaluate_model debe devolver accuracy, precision, recall y f1 correctos."""
    x_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_test = pd.Series([0, 1, 0, 1])
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(x_test, y_test)

    metrics = evaluate_model(model, x_test, y_test)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert metrics["accuracy"] == EXPECTED_ACCURACY
    assert metrics["recall"] == EXPECTED_RECALL
