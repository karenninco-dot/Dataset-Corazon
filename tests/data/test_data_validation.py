"""Unit tests for src.data.data_validation."""

import pandas as pd
import pytest

from src.data.data_preparation import convert_dtypes, select_features
from src.data.data_validation import DataValidationError, validate_features
from tests.data.test_data_preparation import CFG, sample_raw_df


def valid_features_df() -> pd.DataFrame:
    """A small, fully valid features dataframe (2 rows, no nulls)."""
    typed_df = convert_dtypes(sample_raw_df(), CFG)
    return select_features(typed_df, CFG)


def test_validate_features_passes_with_valid_data() -> None:
    """Datos válidos deben pasar la validación sin lanzar ningún error."""
    df = valid_features_df()

    result = validate_features(df)

    assert len(result) == len(df)


def test_validate_features_rejects_out_of_range_value() -> None:
    """Una edad fuera de rango (0-120) debe ser rechazada."""
    df = valid_features_df()
    df.loc[0, "age"] = 200.0

    with pytest.raises(DataValidationError):
        validate_features(df)


def test_validate_features_rejects_invalid_category() -> None:
    """Una categoría no permitida en 'sex' debe ser rechazada."""
    df = valid_features_df()
    df["sex"] = df["sex"].cat.add_categories(["Otro"])
    df.loc[0, "sex"] = "Otro"

    with pytest.raises(DataValidationError):
        validate_features(df)


def test_validate_features_rejects_duplicate_rows() -> None:
    """Filas completamente duplicadas deben ser rechazadas."""
    df = valid_features_df()
    df_with_duplicate = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError):
        validate_features(df_with_duplicate)


def test_validate_features_rejects_excessive_nulls() -> None:
    """Un porcentaje de nulos superior al permitido debe ser rechazado."""
    df = valid_features_df()
    df.loc[0, "chol"] = None

    with pytest.raises(DataValidationError):
        validate_features(df)
