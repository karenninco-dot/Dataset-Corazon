"""Data validation for the Corazon features dataset, using Pandera.

Defines the quality contract the final features dataframe must satisfy
before it is persisted by the feature pipeline: expected types, valid
ranges, maximum null percentage per column, valid categories, and
absence of fully duplicated rows.
"""

from __future__ import annotations

import pandas as pd
from pandera.errors import SchemaErrors
from pandera.pandas import Check, Column, DataFrameSchema

# Maximum fraction of missing values tolerated per column, based on the
# null rates observed in the EDA (notebooks 02-04). Sourced data quality
# issues above this threshold should be investigated before training.
MAX_NULL_RATIO = {
    "age": 0.30,
    "sex": 0.30,
    "chest_pain": 0.30,
    "chol": 0.30,
    "rest_ecg": 0.30,
    "max_hr": 0.30,
    "exang": 0.30,
    "old_peak": 0.30,
    "slope": 0.30,
    "ca": 0.30,
    "thal": 0.30,
}


def _max_null_ratio_check(column: str, max_ratio: float) -> Check:
    """Build a Pandera Check that fails when a column has too many nulls.

    Note: Pandera drops null values before running custom checks unless
    ``ignore_na=False`` is set, so it is required here to actually see
    the nulls we want to count.
    """
    return Check(
        lambda s: s.isna().mean() <= max_ratio,
        element_wise=False,
        ignore_na=False,
        name=f"{column}_max_null_ratio_{max_ratio}",
        error=f"'{column}' supera el {max_ratio:.0%} máximo de valores nulos permitido",
    )


FEATURES_SCHEMA = DataFrameSchema(  # type: ignore[no-untyped-call]
    {
        "age": Column(
            float,
            [Check.in_range(0, 120), _max_null_ratio_check("age", MAX_NULL_RATIO["age"])],
            nullable=True,
        ),
        "sex": Column(
            "category",
            [Check.isin(["Male", "Female"]), _max_null_ratio_check("sex", MAX_NULL_RATIO["sex"])],
            nullable=True,
        ),
        "chest_pain": Column(
            "category",
            [
                Check.isin(["typical", "asymptomatic", "nonanginal", "nontypical"]),
                _max_null_ratio_check("chest_pain", MAX_NULL_RATIO["chest_pain"]),
            ],
            nullable=True,
        ),
        "chol": Column(
            float,
            [Check.in_range(0, 700), _max_null_ratio_check("chol", MAX_NULL_RATIO["chol"])],
            nullable=True,
        ),
        "rest_ecg": Column(
            "category",
            [
                Check.isin(["normal", "left ventricular hypertrophy", "ST-T wave abnormality"]),
                _max_null_ratio_check("rest_ecg", MAX_NULL_RATIO["rest_ecg"]),
            ],
            nullable=True,
        ),
        "max_hr": Column(
            "Int16",
            [Check.in_range(60, 220), _max_null_ratio_check("max_hr", MAX_NULL_RATIO["max_hr"])],
            nullable=True,
        ),
        "exang": Column(
            "boolean",
            _max_null_ratio_check("exang", MAX_NULL_RATIO["exang"]),
            nullable=True,
        ),
        "old_peak": Column(
            float,
            [Check.in_range(0, 10), _max_null_ratio_check("old_peak", MAX_NULL_RATIO["old_peak"])],
            nullable=True,
        ),
        "slope": Column(
            "category",
            [Check.isin(["1", "2", "3"]), _max_null_ratio_check("slope", MAX_NULL_RATIO["slope"])],
            nullable=True,
        ),
        "ca": Column(
            "Int8",
            [Check.in_range(0, 3), _max_null_ratio_check("ca", MAX_NULL_RATIO["ca"])],
            nullable=True,
        ),
        "thal": Column(
            "category",
            [
                Check.isin(["normal", "fixed", "reversable"]),
                _max_null_ratio_check("thal", MAX_NULL_RATIO["thal"]),
            ],
            nullable=True,
        ),
        "disease": Column(int, Check.isin([0, 1]), nullable=False),
    },
    checks=Check(
        lambda df: ~df.duplicated(),
        name="no_duplicate_rows",
        error="El dataset contiene filas completamente duplicadas",
    ),
    strict=False,
)


class DataValidationError(Exception):
    """Raised when the features dataframe fails the Pandera schema."""


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the features dataframe against ``FEATURES_SCHEMA``.

    Args:
        df: Features dataframe, as returned by
            :func:`src.data.data_preparation.select_features`.

    Returns:
        The same dataframe, unchanged, if it passes validation.

    Raises:
        DataValidationError: If any validation rule fails. The message
            includes the full table of failure cases so the problem is
            easy to diagnose.
    """
    try:
        return FEATURES_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        raise DataValidationError(
            "Los features no pasaron la validación de calidad "
            f"({len(exc.failure_cases)} fallas encontradas):\n"
            f"{exc.failure_cases.to_string()}"
        ) from exc
