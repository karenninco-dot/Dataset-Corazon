"""Model validation utilities: cross-validation and train/CV/test comparison.

Provides a more robust validation of the model than a single train/test
split, per the course's Model Validation guidance: stratified k-fold
cross-validation on train (which preserves the target's class balance in
every fold, like the train/test split itself), and a comparison between
train, cross-validation and test scores to detect overfitting or
underfitting.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.model.model_training import build_preprocessor

SCORING = ["accuracy", "precision", "recall", "f1"]
OVERFIT_GAP_THRESHOLD = 0.1
LOW_SCORE_THRESHOLD = 0.6


def build_model_pipeline(cfg: DictConfig) -> Pipeline:
    """Build an unfitted preprocessing + Random Forest pipeline.

    Args:
        cfg: Configuration with the ``model_columns``, ``valid_categories``
            and ``train.random_forest`` sections.

    Returns:
        An unfitted Pipeline combining the preprocessor and the model.
    """
    preprocessor = build_preprocessor(cfg)
    model = RandomForestClassifier(**cfg.train.random_forest)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def cross_validate_model(
    x_train: pd.DataFrame, y_train: pd.Series, cfg: DictConfig
) -> dict[str, np.ndarray]:
    """Run stratified k-fold cross-validation on the training set.

    Args:
        x_train: Training features.
        y_train: Training target.
        cfg: Configuration with the ``validation.cv_folds`` and
            ``train.random_state`` sections.

    Returns:
        The raw ``cross_validate`` results: per-metric arrays with one
        score per fold (keys like ``test_accuracy``, ``test_f1``, ...).
    """
    pipeline = build_model_pipeline(cfg)
    cv = StratifiedKFold(
        n_splits=cfg.validation.cv_folds,
        shuffle=True,
        random_state=cfg.train.random_state,
    )
    results: dict[str, np.ndarray] = cross_validate(
        pipeline, x_train, y_train, cv=cv, scoring=SCORING
    )
    return results


def _score_on_set(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute accuracy, precision, recall and f1 for a fitted model on a set.

    Args:
        model: Fitted pipeline.
        x: Features to predict on.
        y: True target values.

    Returns:
        A dict with the four metrics.
    """
    y_pred = model.predict(x)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }


def compare_train_cv_test(
    model: Pipeline,
    train_data: tuple[pd.DataFrame, pd.Series],
    test_data: tuple[pd.DataFrame, pd.Series],
    cv_results: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compare train, cross-validation and test scores per metric.

    Warns if the gap between the train score and the cross-validation or
    test score suggests overfitting (the model performs much better on
    train than on unseen data), or if all scores are low, which suggests
    underfitting (the model does not fit the data well enough anywhere).

    Args:
        model: Fitted pipeline (already trained on the full x_train).
        train_data: A ``(x_train, y_train)`` tuple with the training
            features and target.
        test_data: A ``(x_test, y_test)`` tuple with the test features
            and target.
        cv_results: Output of :func:`cross_validate_model`.

    Returns:
        A dict per metric with ``train``, ``cv_mean``, ``cv_std`` and
        ``test`` scores.
    """
    x_train, y_train = train_data
    x_test, y_test = test_data
    train_scores = _score_on_set(model, x_train, y_train)
    test_scores = _score_on_set(model, x_test, y_test)

    comparison: dict[str, dict[str, float]] = {}
    for metric in SCORING:
        cv_scores = cv_results[f"test_{metric}"]
        comparison[metric] = {
            "train": train_scores[metric],
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "test": test_scores[metric],
        }

        train_cv_gap = comparison[metric]["train"] - comparison[metric]["cv_mean"]
        train_test_gap = comparison[metric]["train"] - comparison[metric]["test"]
        if train_cv_gap > OVERFIT_GAP_THRESHOLD or train_test_gap > OVERFIT_GAP_THRESHOLD:
            warnings.warn(
                f"Posible overfitting en '{metric}': "
                f"train={comparison[metric]['train']:.3f}, "
                f"cv={comparison[metric]['cv_mean']:.3f}, "
                f"test={comparison[metric]['test']:.3f}.",
                stacklevel=2,
            )
        elif (
            comparison[metric]["train"] < LOW_SCORE_THRESHOLD
            and comparison[metric]["cv_mean"] < LOW_SCORE_THRESHOLD
        ):
            warnings.warn(
                f"Posible underfitting en '{metric}': "
                f"train={comparison[metric]['train']:.3f} y "
                f"cv={comparison[metric]['cv_mean']:.3f} son ambos bajos.",
                stacklevel=2,
            )

    return comparison
