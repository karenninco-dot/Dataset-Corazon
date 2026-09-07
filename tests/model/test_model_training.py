"""Unit tests for src.model.model_training."""

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.model.model_training import build_preprocessor, split_train_test, train_model

EXPECTED_TEST_SIZE = 4

MODEL_CFG = OmegaConf.create(
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
            "test_size": 0.5,
            "random_state": 42,
            "random_forest": {
                "criterion": "entropy",
                "max_depth": 2,
                "max_features": 2,
                "random_state": 42,
            },
        },
    }
)


def sample_model_df() -> pd.DataFrame:
    """A tiny, already-typed features dataframe for model tests."""
    return pd.DataFrame(
        {
            "age": [63.0, 67.0, 45.0, 54.0, 39.0, 61.0, 50.0, 58.0],
            "chol": [233.0, 286.0, np.nan, 250.0, 210.0, 260.0, 240.0, 200.0],
            "exang": pd.array(
                [False, True, False, True, False, True, False, True], dtype="boolean"
            ),
            "sex": pd.Categorical(
                ["Male", "Male", "Female", None, "Male", "Female", "Male", "Female"]
            ),
            "thal": pd.Categorical(
                [
                    "fixed",
                    "normal",
                    "normal",
                    "reversable",
                    "fixed",
                    "normal",
                    "reversable",
                    "fixed",
                ]
            ),
            "slope": pd.Categorical(
                ["3", "2", "1", "2", "3", "1", "2", "3"],
                categories=["1", "2", "3"],
                ordered=True,
            ),
            "disease": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_split_train_test_is_stratified_and_sized() -> None:
    """split_train_test debe dividir respetando test_size y estratificando por target."""
    df = sample_model_df()

    x_train, x_test, _y_train, _y_test = split_train_test(df, MODEL_CFG)

    assert len(x_train) + len(x_test) == len(df)
    assert len(x_test) == EXPECTED_TEST_SIZE
    assert "disease" not in x_train.columns


def test_build_preprocessor_returns_column_transformer() -> None:
    """build_preprocessor debe devolver un ColumnTransformer sin ajustar."""
    preprocessor = build_preprocessor(MODEL_CFG)

    assert isinstance(preprocessor, ColumnTransformer)


def test_train_model_returns_fitted_pipeline_that_predicts() -> None:
    """train_model debe devolver un Pipeline ajustado capaz de predecir."""
    df = sample_model_df()
    x_train, x_test, y_train, _y_test = split_train_test(df, MODEL_CFG)

    model = train_model(x_train, y_train, MODEL_CFG)

    assert isinstance(model, Pipeline)
    preds = model.predict(x_test)
    assert len(preds) == len(x_test)
