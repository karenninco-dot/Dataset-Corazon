"""Unit tests for src.pipelines.inference_pipeline.inference_pipeline."""

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.dummy import DummyClassifier

from src.pipelines.inference_pipeline.inference_pipeline import (
    extract_new_data,
    generate_predictions,
    load_model,
    save_predictions,
)


def test_load_model_reads_a_fitted_pipeline(tmp_path: Path) -> None:
    """load_model debe cargar un modelo joblib que pueda predecir."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(pd.DataFrame({"x": [1, 2, 3]}), pd.Series([0, 1, 0]))
    model_path = tmp_path / "dummy_model.joblib"
    dump(model, model_path)

    loaded_model = load_model(str(model_path))

    assert list(loaded_model.predict(pd.DataFrame({"x": [1]}))) == [0]


def test_extract_new_data_reads_csv(tmp_path: Path) -> None:
    """extract_new_data debe leer un CSV y devolver el mismo dataframe."""
    csv_path = tmp_path / "new_patients.csv"
    expected_df = pd.DataFrame({"age": [50.0, 60.0], "chol": [200.0, 250.0]})
    expected_df.to_csv(csv_path, index=False)

    new_data = extract_new_data(str(csv_path))

    assert list(new_data.columns) == list(expected_df.columns)
    assert len(new_data) == len(expected_df)


def test_generate_predictions_adds_prediction_column() -> None:
    """generate_predictions debe agregar una columna disease_prediction."""
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(pd.DataFrame({"x": [1, 2, 3]}), pd.Series([0, 1, 0]))
    new_data = pd.DataFrame({"x": [10, 20, 30]})

    predictions = generate_predictions(model, new_data)

    assert list(predictions["disease_prediction"]) == [1, 1, 1]
    assert list(predictions["x"]) == list(new_data["x"])


def test_save_predictions_writes_csv_file(tmp_path: Path) -> None:
    """save_predictions debe guardar un CSV legible con las predicciones."""
    predictions = pd.DataFrame({"age": [50.0], "disease_prediction": [1]})
    output_path = tmp_path / "predictions" / "predicciones.csv"

    save_predictions(predictions, str(output_path))

    assert output_path.exists()
    saved_predictions = pd.read_csv(output_path)
    assert saved_predictions["disease_prediction"].tolist() == [1]
