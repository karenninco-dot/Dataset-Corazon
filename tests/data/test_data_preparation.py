"""Unit tests for src.data.data_preparation."""

import pandas as pd
from omegaconf import OmegaConf

from src.data.data_preparation import convert_dtypes, select_features

CFG = OmegaConf.create(
    {
        "columns": {
            "numeric_continuous": ["age", "rest_bp", "chol", "old_peak"],
            "categoric": ["sex", "chest_pain", "rest_ecg", "thal"],
            "categoric_ordinal": "slope",
            "integer_int16": "max_hr",
            "integer_int8": "ca",
            "boolean": ["fbs", "exang", "disease"],
            "target": "disease",
            "drop_low_value": ["fbs", "rest_bp"],
        },
        "valid_categories": {
            "sex": ["Male", "Female"],
            "chest_pain": ["typical", "asymptomatic", "nonanginal", "nontypical"],
            "rest_ecg": [
                "normal",
                "left ventricular hypertrophy",
                "ST-T wave abnormality",
            ],
            "thal": ["normal", "fixed", "reversable"],
            "fbs": ["0.0", "1.0"],
            "exang": ["0", "1"],
            "slope": ["1", "2", "3"],
            "ca": ["0.0", "1.0", "2.0", "3.0"],
            "disease": ["0", "1"],
        },
    }
)


def sample_raw_df() -> pd.DataFrame:
    """A tiny raw dataframe shaped like the Corazon dataset."""
    return pd.DataFrame(
        {
            "age": ["63", "67"],
            "sex": ["Male", "Male"],
            "chest_pain": ["typical", "asymptomatic"],
            "rest_bp": ["145", "160"],
            "chol": ["233", "286"],
            "fbs": ["1.0", "0.0"],
            "rest_ecg": [
                "left ventricular hypertrophy",
                "left ventricular hypertrophy",
            ],
            "max_hr": ["150", "108"],
            "exang": ["0", "1"],
            "old_peak": ["2.3", "1.5"],
            "slope": ["3", "2"],
            "ca": ["0.0", "3.0"],
            "thal": ["fixed", "normal"],
            "disease": ["0", "1"],
        }
    )


def test_convert_dtypes_sets_expected_types() -> None:
    df = convert_dtypes(sample_raw_df(), CFG)

    assert df["age"].dtype == "float64"
    assert str(df["sex"].dtype) == "category"
    assert df["slope"].cat.ordered
    assert df["max_hr"].dtype == "Int16"
    assert df["ca"].dtype == "Int8"
    assert df["fbs"].dtype == "boolean"


def test_convert_dtypes_flags_invalid_category_as_missing() -> None:
    raw_df = sample_raw_df()
    raw_df.loc[0, "sex"] = "unknown"

    df = convert_dtypes(raw_df, CFG)

    assert pd.isna(df.loc[0, "sex"])


def test_select_features_drops_target_na_duplicates_and_low_value_columns() -> None:
    df = convert_dtypes(sample_raw_df(), CFG)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate a row

    result = select_features(df, CFG)

    assert "fbs" not in result.columns
    assert "rest_bp" not in result.columns
    assert result["disease"].dtype.kind == "i"
    assert not result.duplicated().any()


def test_select_features_drops_rows_with_missing_target() -> None:
    raw_df = sample_raw_df()
    raw_df.loc[0, "disease"] = None

    df = convert_dtypes(raw_df, CFG)
    result = select_features(df, CFG)

    assert len(result) == 1
