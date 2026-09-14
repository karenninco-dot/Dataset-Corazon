"""Inference (prediction) pipeline for the Corazon (heart disease) project.

Loads the trained pipeline (preprocessor + model), reads new patient data
from a file, applies the same transformations used during training (the
preprocessing step is part of the persisted pipeline itself, so calling
``predict`` reproduces it), and stores the resulting predictions.

Usage:
    uv run python -m src.pipelines.inference_pipeline.inference_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from joblib import load
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Pipeline:
    """Load the trained pipeline (preprocessor + model) with joblib.

    Args:
        model_path: Path to the persisted joblib file, as saved by
            :func:`src.pipelines.training_pipeline.train_pipeline.save_model`.

    Returns:
        The fitted Pipeline.
    """
    logger.info("Loading model from %s", model_path)
    model: Pipeline = load(model_path)
    return model


def extract_new_data(data_path: str) -> pd.DataFrame:
    """Read new patient data (without a target column) from a CSV file.

    Args:
        data_path: Path to the CSV file with new patients.

    Returns:
        The new patients dataframe.
    """
    logger.info("Reading new patient data from %s", data_path)
    return pd.read_csv(data_path)


def generate_predictions(model: Pipeline, new_data: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for new data using the trained pipeline.

    The pipeline's preprocessor (imputation, scaling, encoding) is applied
    automatically by ``predict``, reproducing the exact transformations
    used during training.

    Args:
        model: Fitted pipeline (preprocessor + model).
        new_data: New patient features, with the same columns used
            during training.

    Returns:
        A copy of ``new_data`` with an added ``disease_prediction`` column.
    """
    predictions = new_data.copy()
    predictions["disease_prediction"] = model.predict(new_data)
    return predictions


def save_predictions(predictions: pd.DataFrame, predictions_path: str) -> None:
    """Persist the predictions as a CSV file.

    Args:
        predictions: Output of :func:`generate_predictions`.
        predictions_path: Destination path for the CSV file. Parent
            directories are created if they do not exist.
    """
    output_path = Path(predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("Saved predictions to %s", output_path)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """Run the inference pipeline end to end: load, read, predict, save."""
    logging.basicConfig(level=logging.INFO)

    model = load_model(cfg.data.model)
    new_data = extract_new_data(cfg.data.new_patients)

    logger.info("Generating predictions for %s new patients", len(new_data))
    predictions = generate_predictions(model, new_data)
    logger.info(
        "Predicted disease counts: %s",
        predictions["disease_prediction"].value_counts().to_dict(),
    )

    save_predictions(predictions, cfg.data.predictions)

    logger.info("Inference pipeline completed.")


if __name__ == "__main__":  # pragma: no cover
    run()
