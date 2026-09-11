"""Training pipeline for the Corazon (heart disease) project.

Reads the processed features from the feature pipeline, splits them
into train/test sets, trains a Random Forest classification pipeline
(preprocessing + model) with the hyperparameters already selected in
the POC (notebook 06.ModelSelection), evaluates it on the test set,
and persists both the trained pipeline and the evaluation metrics.

Usage:
    uv run python -m src.pipelines.training_pipeline.train_pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from joblib import dump
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

from src.model.model_evaluation import evaluate_model
from src.model.model_training import split_train_test, train_model
from src.model.model_validation import compare_train_cv_test, cross_validate_model
from src.model.train_test_validation import validate_train_test_split

logger = logging.getLogger(__name__)


def extract_data(feature_path: str) -> pd.DataFrame:
    """Read the processed features produced by the feature pipeline.

    Args:
        feature_path: Path to the features parquet file.

    Returns:
        The features dataframe.
    """
    logger.info("Reading features from %s", feature_path)
    return pd.read_parquet(feature_path)


def save_model(model: Pipeline, model_path: str) -> None:
    """Persist the trained pipeline (preprocessor + model) with joblib.

    Args:
        model: Fitted Pipeline to persist.
        model_path: Destination path for the joblib file. Parent
            directories are created if they do not exist.
    """
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, output_path)
    logger.info("Saved trained model to %s", output_path)


def save_metrics(metrics: dict[str, float], metrics_path: str) -> None:
    """Persist the evaluation metrics as a JSON file.

    Args:
        metrics: Dict of evaluation metrics (see
            :func:`src.model.model_evaluation.evaluate_model`).
        metrics_path: Destination path for the JSON file. Parent
            directories are created if they do not exist.
    """
    output_path = Path(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved evaluation metrics to %s", output_path)


def save_validation_results(
    validation_results: dict[str, dict[str, float]], validation_path: str
) -> None:
    """Persist the train/CV/test comparison as a JSON file.

    Args:
        validation_results: Output of
            :func:`src.model.model_validation.compare_train_cv_test`.
        validation_path: Destination path for the JSON file. Parent
            directories are created if they do not exist.
    """
    output_path = Path(validation_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(validation_results, indent=2))
    logger.info("Saved validation results to %s", output_path)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """Run the training pipeline end to end: split, train, evaluate, save."""
    logging.basicConfig(level=logging.INFO)

    features_df = extract_data(cfg.data.feature)
    x_train, x_test, y_train, y_test = split_train_test(features_df, cfg)

    logger.info("Validating train/test split")
    validate_train_test_split(x_train, x_test, y_train, y_test)

    logger.info("Training Random Forest on %s rows", len(x_train))
    model = train_model(x_train, y_train, cfg)

    metrics = evaluate_model(model, x_test, y_test)
    logger.info("Evaluation metrics: %s", metrics)

    logger.info("Running %s-fold cross-validation on train", cfg.validation.cv_folds)
    cv_results = cross_validate_model(x_train, y_train, cfg)
    validation_results = compare_train_cv_test(
        model, (x_train, y_train), (x_test, y_test), cv_results
    )
    logger.info("Train/CV/test comparison: %s", validation_results)

    save_model(model, cfg.data.model)
    save_metrics(metrics, cfg.data.train_metrics)
    save_validation_results(validation_results, cfg.data.validation_metrics)

    logger.info("Training pipeline completed.")


if __name__ == "__main__":  # pragma: no cover
    run()
