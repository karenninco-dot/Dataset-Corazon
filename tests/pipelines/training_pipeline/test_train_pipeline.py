"""Unit tests for src.pipelines.training_pipeline.train_pipeline."""

import json
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.dummy import DummyClassifier

from src.pipelines.training_pipeline.train_pipeline import (
    extract_data,
    save_metrics,
    save_model,
    save_validation_results,
)


def test_extract_data_reads_parquet(tmp_path: Path) -> None:
    """extract_data debe leer un parquet y devolver el mismo dataframe."""
    parquet_path = tmp_path / "features.parquet"
    expected_df = pd.DataFrame({"age": [63.0, 67.0], "disease": [0, 1]})
    expected_df.to_parquet(parquet_path, index=False)

    df = extract_data(str(parquet_path))

    assert len(df) == len(expected_df)
    assert list(df.columns) == list(expected_df.columns)


def test_save_model_writes_loadable_joblib_file(tmp_path: Path) -> None:
    """save_model debe guardar un modelo que se pueda recargar y usar."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(pd.DataFrame({"x": [1, 2, 3]}), pd.Series([0, 1, 0]))
    output_path = tmp_path / "models" / "dummy.joblib"

    save_model(model, str(output_path))

    assert output_path.exists()
    loaded_model = load(output_path)
    assert list(loaded_model.predict(pd.DataFrame({"x": [1]}))) == [0]


def test_save_metrics_writes_json_file(tmp_path: Path) -> None:
    """save_metrics debe guardar un JSON legible con las métricas."""
    metrics = {"accuracy": 0.86, "recall": 0.85}
    output_path = tmp_path / "metrics" / "train_metrics.json"

    save_metrics(metrics, str(output_path))

    assert output_path.exists()
    saved_metrics = json.loads(output_path.read_text())
    assert saved_metrics == metrics


def test_save_validation_results_writes_json_file(tmp_path: Path) -> None:
    """save_validation_results debe guardar un JSON legible con la comparación."""
    validation_results = {
        "accuracy": {"train": 0.98, "cv_mean": 0.81, "cv_std": 0.04, "test": 0.86}
    }
    output_path = tmp_path / "metrics" / "validation_metrics.json"

    save_validation_results(validation_results, str(output_path))

    assert output_path.exists()
    saved_results = json.loads(output_path.read_text())
    assert saved_results == validation_results
