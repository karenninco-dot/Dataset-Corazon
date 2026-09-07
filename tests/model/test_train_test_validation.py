"""Unit tests for src.model.train_test_validation."""

import warnings

import pandas as pd
import pytest

from src.model.train_test_validation import (
    TrainTestSplitError,
    validate_train_test_split,
)


def valid_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """A small train/test split with no leakage and matching distributions."""
    x_train = pd.DataFrame(
        {
            "age": [40, 42, 44, 46, 48, 50, 52, 54, 56, 58],
            "sex": ["Male", "Female"] * 5,
        }
    )
    y_train = pd.Series([0, 1] * 5)

    x_test = pd.DataFrame(
        {
            "age": [41, 45, 49, 53, 57, 43],
            "sex": ["Male", "Female", "Male", "Female", "Male", "Female"],
        }
    )
    y_test = pd.Series([0, 1, 0, 1, 0, 1])

    return x_train, x_test, y_train, y_test


def test_validate_train_test_split_passes_with_valid_split() -> None:
    """Un split sin fuga y con distribuciones similares no debe lanzar nada."""
    x_train, x_test, y_train, y_test = valid_split()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_train_test_split(x_train, x_test, y_train, y_test)

    assert len(caught) == 0


def test_validate_train_test_split_raises_on_leakage() -> None:
    """Una fila duplicada entre train y test debe lanzar TrainTestSplitError."""
    x_train, x_test, y_train, y_test = valid_split()
    x_train_leak = pd.concat([x_train, x_test.iloc[[0]]], ignore_index=True)
    y_train_leak: pd.Series = pd.concat([y_train, y_test.iloc[0:1]], ignore_index=True)

    with pytest.raises(TrainTestSplitError):
        validate_train_test_split(x_train_leak, x_test, y_train_leak, y_test)


def test_validate_train_test_split_warns_on_target_imbalance() -> None:
    """Una diferencia grande en la proporción del target debe generar una advertencia."""
    x_train, x_test, _y_train, _y_test = valid_split()
    y_train_imbalanced = pd.Series([0] * 9 + [1])
    y_test_imbalanced = pd.Series([1] * 5 + [0])

    with pytest.warns(UserWarning, match="proporción"):
        validate_train_test_split(x_train, x_test, y_train_imbalanced, y_test_imbalanced)


def test_validate_train_test_split_warns_on_feature_drift() -> None:
    """Un cambio fuerte en la distribución de las columnas debe generar una advertencia."""
    x_train = pd.DataFrame(
        {
            "age": [40, 42, 44, 46, 48, 50, 52, 54, 56, 58],
            "chol": [200, 205, 195, 210, 190, 200, 205, 195, 210, 190],
        }
    )
    y_train = pd.Series([0, 1] * 5)

    x_test = pd.DataFrame(
        {
            "age": [80, 82, 84, 86, 88, 90],
            "chol": [400, 405, 395, 410, 390, 400],
        }
    )
    y_test = pd.Series([0, 1, 0, 1, 0, 1])

    with pytest.warns(UserWarning, match="drift"):
        validate_train_test_split(x_train, x_test, y_train, y_test)
