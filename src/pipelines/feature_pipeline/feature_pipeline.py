"""Feature pipeline for the Corazon (heart disease) project.

Reads the raw dataset, converts it into properly typed data, selects
and cleans the model features, and persists the result as a parquet
file ready to be consumed by the training pipeline.

This script performs extraction, transformation and loading (ETL) of
the static Corazon dataset.

Usage:
    uv run python -m src.pipelines.feature_pipeline.feature_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.data.data_preparation import convert_dtypes, select_features

logger = logging.getLogger(__name__)


def extract_data(raw_path: str) -> pd.DataFrame:
    """Read the raw Corazon dataset from a CSV file.

    Args:
        raw_path: Path to the raw CSV file.

    Returns:
        The raw dataframe, as-is (no type conversion yet).
    """
    logger.info("Reading raw data from %s", raw_path)
    return pd.read_csv(raw_path, low_memory=False)


def build_features(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Transform raw data into the features used for modeling.

    Args:
        df: Raw dataframe, as returned by :func:`extract_data`.
        cfg: Pipeline configuration (see ``conf/config.yaml``).

    Returns:
        A cleaned, properly typed dataframe with the selected feature
        columns and the target column.
    """
    typed_df = convert_dtypes(df, cfg)
    features_df = select_features(typed_df, cfg)
    return features_df


def save_features(df: pd.DataFrame, feature_path: str) -> None:
    """Persist the features dataframe as parquet.

    Args:
        df: Features dataframe to persist.
        feature_path: Destination path for the parquet file. Parent
            directories are created if they do not exist.
    """
    output_path = Path(feature_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %s feature rows to %s", len(df), output_path)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """Run the feature pipeline end to end: extract, transform, load."""
    logging.basicConfig(level=logging.INFO)

    raw_df = extract_data(cfg.data.raw)
    features_df = build_features(raw_df, cfg)
    save_features(features_df, cfg.data.feature)

    logger.info("Feature pipeline completed.")


if __name__ == "__main__":  # pragma: no cover
    run()
