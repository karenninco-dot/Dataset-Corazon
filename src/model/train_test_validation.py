"""Train/test split validation utilities.

Verifies that a train/test split does not leak information between
the two sets and is representative of the expected distribution of
the problem, per the course's Train/Test Split Checks guidance:
https://joserzapata.github.io/courses/ciencia-datos-en-produccion/data-validation/train_test-checks/
"""

from __future__ import annotations

import warnings

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

TARGET_COLUMN = "disease"
MAX_TARGET_PROPORTION_DIFF = 0.1
MAX_DRIFTED_COLUMNS_SHARE = 0.3
DRIFT_P_VALUE_THRESHOLD = 0.05


class TrainTestSplitError(Exception):
    """Raised when a train/test split has a data leakage problem."""


def _check_no_leakage(x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
    """Raise if any row appears in both train and test.

    Args:
        x_train: Training features.
        x_test: Test features.

    Raises:
        TrainTestSplitError: If one or more rows are identical between
            train and test, which would let the model "see" test data
            during training and invalidate the evaluation.
    """
    combined = pd.concat([x_train, x_test])
    is_duplicated = combined.duplicated(keep=False)
    if is_duplicated.any():
        n_leaked = int(is_duplicated.sum())
        raise TrainTestSplitError(
            f"Se encontraron {n_leaked} filas duplicadas entre train y test "
            "(fuga de información / data leakage)."
        )


def _check_target_balance(y_train: pd.Series, y_test: pd.Series) -> None:
    """Warn if the target proportion differs too much between train and test.

    Args:
        y_train: Training target.
        y_test: Test target.
    """
    train_ratio = float(y_train.mean())
    test_ratio = float(y_test.mean())
    diff = abs(train_ratio - test_ratio)
    if diff > MAX_TARGET_PROPORTION_DIFF:
        warnings.warn(
            f"La proporción de '{TARGET_COLUMN}' difiere entre train ({train_ratio:.2%}) "
            f"y test ({test_ratio:.2%}) por más de {MAX_TARGET_PROPORTION_DIFF:.0%}.",
            stacklevel=2,
        )


def _build_drift_dataset(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cast features + target into plain dtypes accepted by Evidently."""
    df = x.copy()
    df[TARGET_COLUMN] = y.to_numpy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]
    categoric_cols = [col for col in df.columns if col not in [*numeric_cols, TARGET_COLUMN]]
    categoric_cols.append(TARGET_COLUMN)

    for col in numeric_cols:
        df[col] = df[col].astype("float64")
    for col in categoric_cols:
        df[col] = df[col].astype(object).where(df[col].notna(), None).astype(str)

    return df


def _check_feature_drift(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    """Warn if too many columns show significant drift between train and test.

    Uses Evidently's ``DataDriftPreset``, which runs an appropriate
    statistical test per column (Kolmogorov-Smirnov for numeric,
    chi-square/Z-test for categorical) and reports a p-value: a value
    below ``DRIFT_P_VALUE_THRESHOLD`` means the column's distribution
    changed significantly between train and test.

    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training target (the target's own drift is checked too).
        y_test: Test target.
    """
    train_df = _build_drift_dataset(x_train, y_train)
    test_df = _build_drift_dataset(x_test, y_test)

    numeric_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
    categoric_cols = [c for c in train_df.columns if c not in numeric_cols]
    definition = DataDefinition(numerical_columns=numeric_cols, categorical_columns=categoric_cols)

    train_ds = Dataset.from_pandas(train_df, data_definition=definition)
    test_ds = Dataset.from_pandas(test_df, data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(test_ds, train_ds).dict()

    drift_summary = next(
        (m for m in result["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")),
        None,
    )
    if drift_summary is None:
        return

    share = drift_summary["value"]["share"]
    if share > MAX_DRIFTED_COLUMNS_SHARE:
        drifted_columns = [
            m["config"]["column"]
            for m in result["metrics"]
            if m["metric_name"].startswith("ValueDrift") and m["value"] < DRIFT_P_VALUE_THRESHOLD
        ]
        warnings.warn(
            f"{share:.0%} de las columnas muestran drift significativo entre "
            f"train y test: {drifted_columns}.",
            stacklevel=2,
        )


def validate_train_test_split(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    """Validate a train/test split for leakage and representativeness.

    Args:
        x_train: Training features.
        x_test: Test features.
        y_train: Training target.
        y_test: Test target.

    Raises:
        TrainTestSplitError: If data leakage is detected between train
            and test (duplicated rows). Other, less severe problems
            (target imbalance, feature drift) only raise a warning.
    """
    _check_no_leakage(x_train, x_test)
    _check_target_balance(y_train, y_test)
    _check_feature_drift(x_train, x_test, y_train, y_test)
