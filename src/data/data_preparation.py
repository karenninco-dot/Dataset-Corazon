"""Data preparation utilities shared by the FTI pipelines.

These functions convert the raw Corazon dataset into properly typed
columns and select/clean the columns used as model features. They are
kept independent of Hydra so they can be unit tested in isolation with
a plain configuration object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def convert_dtypes(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Convert raw (string) columns to their proper dtype.

    Replicates the type conversion performed in notebook
    02.Exploracion_inicial: numeric coercion, category cleanup against
    a whitelist of valid values, an ordered category for ``slope``,
    and boolean mapping for yes/no columns.

    Args:
        df: Raw dataframe, as read from the source CSV.
        cfg: Configuration with the ``columns`` and ``valid_categories``
            sections (see ``conf/config.yaml``).

    Returns:
        A new dataframe with corrected dtypes. The input is not
        mutated.
    """
    df = df.copy()

    numeric_continuous_cols = list(cfg.columns.numeric_continuous)
    categoric_cols = list(cfg.columns.categoric)
    boolean_cols = list(cfg.columns.boolean)

    numeric_cols = [*numeric_continuous_cols, cfg.columns.integer_int16]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, valid_values in cfg.valid_categories.items():
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].where(df[col].isin(list(valid_values)), np.nan)

    # OmegaConf lists (ListConfig) don't work reliably as pandas column
    # indexers, so they are converted to plain Python lists above.
    df[categoric_cols] = df[categoric_cols].astype("category")

    ordinal_col = cfg.columns.categoric_ordinal
    df[ordinal_col] = pd.Categorical(
        df[ordinal_col],
        categories=list(cfg.valid_categories[ordinal_col]),
        ordered=True,
    )

    df[numeric_continuous_cols] = df[numeric_continuous_cols].astype("float")
    df[cfg.columns.integer_int16] = df[cfg.columns.integer_int16].astype("Int16")
    df[cfg.columns.integer_int8] = pd.to_numeric(
        df[cfg.columns.integer_int8], errors="coerce"
    ).astype("Int8")

    bool_map = {"0": False, "1": True, "0.0": False, "1.0": True}
    for col in boolean_cols:
        df[col] = df[col].map(bool_map).astype("boolean")

    return df


def select_features(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Select and clean the columns used to train the model.

    Drops rows without a target value, removes the low-information
    columns identified during the EDA (``fbs``, ``rest_bp``),
    deduplicates records, and casts the target to ``int`` (0/1).

    Args:
        df: Typed dataframe, as returned by :func:`convert_dtypes`.
        cfg: Configuration with the ``columns`` section.

    Returns:
        A new dataframe ready to be persisted as model features.
    """
    df = df.dropna(subset=[cfg.columns.target])
    df = df.drop_duplicates()
    df = df.drop(columns=list(cfg.columns.drop_low_value))
    df = df.drop_duplicates()
    df[cfg.columns.target] = df[cfg.columns.target].astype("int")
    return df
