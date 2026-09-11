"""Unit tests for src.model.model_validation."""

import warnings

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.model.model_training import split_train_test, train_model
from src.model.model_validation import compare_train_cv_test, cross_validate_model

EXPECTED_CV_FOLDS = 3

VALIDATION_CFG = OmegaConf.create(
    {
        "columns": {"target": "disease"},
        "model_columns": {
            "numeric": ["age", "chol"],
            "boolean": ["exang"],
            "categoric": ["sex", "thal"],
            "categoric_ordinal": "slope",
        },
        "valid_categories": {
            "sex": ["Male", "Female"],
            "thal": ["normal", "fixed", "reversable"],
            "slope": ["1", "2", "3"],
        },
        "train": {
            "test_size": 0.3,
            "random_state": 42,
            "random_forest": {
                "criterion": "entropy",
                "max_depth": 2,
                "max_features": 2,
                "random_state": 42,
            },
        },
        "validation": {"cv_folds": EXPECTED_CV_FOLDS},
    }
)


def sample_validation_df() -> pd.DataFrame:
    """A features dataframe large enough for a 3-fold stratified split."""
    n_per_class = 15
    n_total = n_per_class * 2

    def repeat_to_length(values: list[str]) -> list[str]:
        """Repeat a small list of category values until it reaches n_total."""
        repeats = n_total // len(values) + 1
        return (values * repeats)[:n_total]

    return pd.DataFrame(
        {
            "age": list(range(40, 40 + n_per_class)) * 2,
            "chol": [200.0 + i for i in range(n_per_class)] * 2,
            "exang": pd.array([False, True] * n_per_class, dtype="boolean"),
            "sex": pd.Categorical(["Male", "Female"] * n_per_class),
            "thal": pd.Categorical(repeat_to_length(["fixed", "normal", "reversable"])),
            "slope": pd.Categorical(
                repeat_to_length(["1", "2", "3"]),
                categories=["1", "2", "3"],
                ordered=True,
            ),
            "disease": [0] * n_per_class + [1] * n_per_class,
        }
    )


def test_cross_validate_model_returns_scores_per_fold() -> None:
    """cross_validate_model debe devolver EXPECTED_CV_FOLDS puntajes por métrica."""
    df = sample_validation_df()
    x_train, _x_test, y_train, _y_test = split_train_test(df, VALIDATION_CFG)

    cv_results = cross_validate_model(x_train, y_train, VALIDATION_CFG)

    assert len(cv_results["test_accuracy"]) == EXPECTED_CV_FOLDS


def test_compare_train_cv_test_returns_all_metrics() -> None:
    """compare_train_cv_test debe devolver train/cv_mean/cv_std/test por métrica."""
    df = sample_validation_df()
    x_train, x_test, y_train, y_test = split_train_test(df, VALIDATION_CFG)
    model = train_model(x_train, y_train, VALIDATION_CFG)
    cv_results = cross_validate_model(x_train, y_train, VALIDATION_CFG)

    comparison = compare_train_cv_test(model, (x_train, y_train), (x_test, y_test), cv_results)

    assert set(comparison["accuracy"].keys()) == {"train", "cv_mean", "cv_std", "test"}


def test_compare_train_cv_test_warns_on_overfitting() -> None:
    """Una brecha grande entre train y cv/test debe generar una advertencia."""
    df = sample_validation_df()
    x_train, x_test, y_train, y_test = split_train_test(df, VALIDATION_CFG)
    model = train_model(x_train, y_train, VALIDATION_CFG)
    cv_results = cross_validate_model(x_train, y_train, VALIDATION_CFG)
    # Fuerza un cv_results artificial con puntajes bajos para simular overfitting.
    fake_cv_results = {key: np.zeros_like(value) for key, value in cv_results.items()}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compare_train_cv_test(model, (x_train, y_train), (x_test, y_test), fake_cv_results)

    assert any("overfitting" in str(w.message) for w in caught)
