"""Unit tests for src.pipelines.feature_pipeline.feature_pipeline."""

from pathlib import Path

import pandas as pd

from src.pipelines.feature_pipeline.feature_pipeline import (
    build_features,
    extract_data,
    save_features,
)
from tests.data.test_data_preparation import CFG, sample_raw_df


def test_extract_data_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "raw.csv"
    expected_df = sample_raw_df()
    expected_df.to_csv(csv_path, index=False)

    df = extract_data(str(csv_path))

    assert len(df) == len(expected_df)
    assert list(df.columns) == list(expected_df.columns)


def test_build_features_returns_clean_dataframe() -> None:
    df = build_features(sample_raw_df(), CFG)

    assert "fbs" not in df.columns
    assert "rest_bp" not in df.columns
    assert df["disease"].isna().sum() == 0


def test_save_features_writes_parquet(tmp_path: Path) -> None:
    df = build_features(sample_raw_df(), CFG)
    output_path = tmp_path / "features" / "corazon_features.parquet"

    save_features(df, str(output_path))

    assert output_path.exists()
    saved_df = pd.read_parquet(output_path)
    assert len(saved_df) == len(df)
