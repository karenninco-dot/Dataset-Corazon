"""Streamlit demo of the Corazon (heart disease) classification model.

Individual prediction: lets the user enter one patient's data through a
web form (in Spanish, with friendly category labels) and shows whether
the model predicts heart disease.

How to run locally:
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

MODEL_PATH = "data/06_models/corazon_classification_model.joblib"
HEART_IMAGE_PATH = "notebooks/7-deploy/images/corazon.jpg"

# Mapas de opciones amigables -> valor real esperado por el modelo.
SEX_OPTIONS = {"Masculino": "Male", "Femenino": "Female"}
CHEST_PAIN_OPTIONS = {
    "Típico": "typical",
    "Asintomático": "asymptomatic",
    "No anginoso": "nonanginal",
    "No típico": "nontypical",
}
REST_ECG_OPTIONS = {
    "Normal": "normal",
    "Hipertrofia ventricular izquierda": "left ventricular hypertrophy",
    "Anomalía de la onda ST-T": "ST-T wave abnormality",
}
THAL_OPTIONS = {
    "Normal": "normal",
    "Defecto fijo": "fixed",
    "Defecto reversible": "reversable",
}
SLOPE_OPTIONS = {"Ascendente": "1", "Plana": "2", "Descendente": "3"}
YES_NO_OPTIONS = {"Sí": True, "No": False}


@st.cache_resource
def load_model(model_path: str) -> Pipeline:
    """Load the trained pipeline (preprocessor + model) with joblib.

    Cached by Streamlit (``st.cache_resource``) so the model is loaded
    only once per session, not on every rerun.

    Args:
        model_path: Path to the persisted joblib file.

    Returns:
        The fitted Pipeline.
    """
    with st.spinner("Cargando modelo..."):
        model = load(model_path)
    return model


def build_patient_dataframe(patient_data: dict[str, float | int | str | bool]) -> pd.DataFrame:
    """Build a one-row dataframe with a patient's data, ready for the model.

    Kept separate from the Streamlit widgets so it can be tested with
    plain Python values.

    Args:
        patient_data: One value per feature expected by the model
            (age, sex, chest_pain, chol, rest_ecg, max_hr, exang,
            old_peak, slope, ca, thal).

    Returns:
        A single-row dataframe with those columns.
    """
    return pd.DataFrame([patient_data])


def get_patient_data() -> pd.DataFrame:
    """Render the input form and collect one patient's data.

    Categorical fields are shown with friendly Spanish labels and mapped
    back to the raw values the model expects.

    Returns:
        A single-row dataframe with the patient's data, matching the
        columns expected by the trained model.
    """
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Edad:", min_value=1, max_value=120, value=50, step=1)
        chol = st.number_input("Colesterol (mg/dl):", min_value=0, max_value=600, value=200, step=1)
        max_hr = st.number_input(
            "Frecuencia cardíaca máxima:", min_value=60, max_value=220, value=150, step=1
        )
    with col_b:
        old_peak = st.number_input(
            "Depresión del segmento ST inducida por ejercicio:",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )
        ca = st.selectbox("Número de vasos principales afectados:", options=[0, 1, 2, 3])
        exang_label = st.radio(
            "¿Angina inducida por ejercicio?", options=list(YES_NO_OPTIONS.keys()), horizontal=True
        )

    col1, col2 = st.columns(2)
    with col1:
        sex_label = st.radio("Sexo:", options=list(SEX_OPTIONS.keys()), horizontal=True)
        chest_pain_label = st.selectbox(
            "Tipo de dolor de pecho:", options=list(CHEST_PAIN_OPTIONS.keys())
        )
    with col2:
        rest_ecg_label = st.selectbox(
            "Resultado del electrocardiograma en reposo:", options=list(REST_ECG_OPTIONS.keys())
        )
        thal_label = st.selectbox(
            "Resultado de la prueba de esfuerzo con talio:", options=list(THAL_OPTIONS.keys())
        )

    slope_label = st.select_slider(
        "Pendiente del segmento ST en el pico del ejercicio:", options=list(SLOPE_OPTIONS.keys())
    )

    patient_data: dict[str, float | int | str | bool] = {
        "age": float(age),
        "sex": SEX_OPTIONS[sex_label],
        "chest_pain": CHEST_PAIN_OPTIONS[chest_pain_label],
        "chol": float(chol),
        "rest_ecg": REST_ECG_OPTIONS[rest_ecg_label],
        "max_hr": max_hr,
        "exang": YES_NO_OPTIONS[exang_label],
        "old_peak": old_peak,
        "slope": SLOPE_OPTIONS[slope_label],
        "ca": ca,
        "thal": THAL_OPTIONS[thal_label],
    }
    return build_patient_dataframe(patient_data)


def predict_disease(model: Pipeline, patient: pd.DataFrame) -> int:
    """Predict heart disease (0/1) for a single patient.

    Args:
        model: Fitted pipeline (preprocessor + model).
        patient: A single-row dataframe with the patient's data.

    Returns:
        ``1`` if the model predicts heart disease, ``0`` otherwise.
    """
    prediction: int = model.predict(patient)[0]
    return prediction


def render_prediction(prediction: int) -> None:
    """Show the prediction result to the user.

    Args:
        prediction: Output of :func:`predict_disease`.
    """
    st.write("")
    if prediction == 0:
        st.success("El modelo predice: bajo riesgo de enfermedad cardíaca. 😀")
    else:
        st.error(
            "El modelo predice: riesgo de enfermedad cardíaca. Se recomienda evaluación médica. 😕"
        )


def individual_prediction_tab(model: Pipeline) -> None:
    """Render the individual prediction tab: form, button and result.

    Args:
        model: Fitted pipeline (preprocessor + model).
    """
    st.subheader("Predicción individual")
    patient = get_patient_data()

    if st.button("Predecir"):
        prediction = predict_disease(model, patient)
        render_prediction(prediction)


def main() -> None:
    """Configure the page, load the model and render the app."""
    st.set_page_config(page_title="Predicción de enfermedad cardíaca", page_icon="🫀")

    _col_img1, col_img2, _col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(HEART_IMAGE_PATH, use_container_width=True)

    st.header("¿Tiene el paciente riesgo de enfermedad cardíaca? ❤️")
    st.caption(
        "Esta es una predicción estadística basada en un modelo de Machine Learning "
        "(recall ≈ 0.85 sobre datos de prueba), no un diagnóstico médico."
    )

    model = load_model(MODEL_PATH)
    individual_prediction_tab(model)


if __name__ == "__main__":
    main()
