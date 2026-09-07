"""Model training utilities: train/test split, preprocessing and training.

Reproduces the preprocessing and model choices made during the POC
(notebook 06.ModelSelection): a ColumnTransformer with per-type
sub-pipelines, and a Random Forest classifier with the hyperparameters
already selected via GridSearchCV (fixed here, not re-searched, since
model selection is a research-time decision, not a production one).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def _boolean_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas nullable boolean columns to float (True/False/<NA> -> 1.0/0.0/NaN).

    Needed because SimpleImputer does not understand pandas' nullable
    "boolean" dtype. A named function (not a lambda) is required here
    so the fitted pipeline can be pickled with joblib.
    """
    return df.astype("float64")


def _sanitize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace pandas' nullable NA markers with plain numpy NaN.

    scikit-learn's imputers do not understand pandas' extension NA
    markers (``pd.NA``) used by "category" and nullable dtypes, and
    raise a confusing ``TypeError: boolean value of NA is ambiguous``.
    Converting to plain ``object``/``float`` dtype with ``np.nan``
    avoids that, without changing which values are considered missing.
    """
    df = df.astype(object)
    return df.where(df.notna(), np.nan)


def split_train_test(
    df: pd.DataFrame, cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the features dataframe into train and test sets.

    Args:
        df: Features dataframe, as produced by the feature pipeline.
        cfg: Configuration with the ``columns.target`` and ``train``
            (test_size, random_state) sections.

    Returns:
        A tuple ``(x_train, x_test, y_train, y_test)``. The split is
        stratified on the target to preserve the class balance in
        both sets.
    """
    x_features = df.drop(columns=[cfg.columns.target])
    y_target = df[cfg.columns.target]
    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y_target,
        test_size=cfg.train.test_size,
        random_state=cfg.train.random_state,
        stratify=y_target,
    )
    return x_train, x_test, y_train, y_test


def build_preprocessor(cfg: DictConfig) -> ColumnTransformer:
    """Build the ColumnTransformer used to preprocess features for training.

    Reproduces the preprocessing pipeline defined in notebook
    06.ModelSelection: median imputation + scaling for numeric columns,
    mode imputation for the boolean column, mode imputation + one-hot
    encoding for nominal categorical columns, and mode imputation +
    ordinal encoding (respecting category order) for ``slope``.

    Args:
        cfg: Configuration with the ``model_columns`` and
            ``valid_categories`` sections.

    Returns:
        An unfitted ColumnTransformer.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # pandas' nullable "boolean" dtype (used for exang) is not
    # understood by SimpleImputer, so it is cast to float first
    # (True/False/<NA> -> 1.0/0.0/NaN).
    boolean_pipe = Pipeline(
        steps=[
            ("to_float", FunctionTransformer(_boolean_to_float)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categoric_pipe = Pipeline(
        steps=[
            ("sanitize", FunctionTransformer(_sanitize_missing)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ordinal_col = cfg.model_columns.categoric_ordinal
    ordinal_categories = list(cfg.valid_categories[ordinal_col])
    categoric_ord_pipe = Pipeline(
        steps=[
            ("sanitize", FunctionTransformer(_sanitize_missing)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ordinal_categories],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, list(cfg.model_columns.numeric)),
            ("boolean", boolean_pipe, list(cfg.model_columns.boolean)),
            ("categoric", categoric_pipe, list(cfg.model_columns.categoric)),
            ("categoric_ordinal", categoric_ord_pipe, [ordinal_col]),
        ]
    )


def train_model(x_train: pd.DataFrame, y_train: pd.Series, cfg: DictConfig) -> Pipeline:
    """Train the Random Forest classification pipeline.

    Uses the hyperparameters already selected via GridSearchCV in
    notebook 06.ModelSelection (see ``conf/config.yaml``, section
    ``train.random_forest``), so no hyperparameter search is repeated
    in production.

    Args:
        x_train: Training features.
        y_train: Training target.
        cfg: Configuration with the ``train.random_forest`` section.

    Returns:
        A fitted Pipeline combining the preprocessor and the model.
    """
    preprocessor = build_preprocessor(cfg)
    model = RandomForestClassifier(**cfg.train.random_forest)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)
    return pipeline
