"""Demo funcional del modelo de clasificación de enfermedad cardíaca.

HOW TO RUN THE APP:
    streamlit run notebooks/7-deploy/corazon-streamlit.py
"""

import os

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline

# Mapas de opciones amigables -> valor real esperado por el modelo
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

SLOPE_OPTIONS = {
    "Ascendente": "1",
    "Plana": "2",
    "Descendente": "3",
}

THAL_OPTIONS = {
    "Normal": "normal",
    "Defecto fijo": "fixed",
    "Defecto reversible": "reversable",
}

SEX_OPTIONS = {
    "Masculino": "Male",
    "Femenino": "Female",
}

YES_NO_OPTIONS = {
    "Sí": True,
    "No": False,
}


def get_user_data() -> pd.DataFrame:
    """Get the data provided by the user through the form.

    :return: DataFrame with the raw user input, ready to feed the model pipeline.
    """
    user_data = {}

    col_a, col_b = st.columns(2)
    with col_a:
        user_data["age"] = st.number_input(
            label="Edad:", min_value=18, max_value=100, value=54, step=1
        )
        user_data["chol"] = st.number_input(
            label="Colesterol (mg/dl):", min_value=100, max_value=600, value=247, step=1
        )
        user_data["max_hr"] = st.number_input(
            label="Frecuencia cardíaca máxima:",
            min_value=60,
            max_value=220,
            value=150,
            step=1,
        )
    with col_b:
        user_data["old_peak"] = st.number_input(
            label="Depresión del segmento ST inducida por ejercicio:",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )
        ca_label = st.radio(
            label="Número de vasos principales afectados:",
            options=["0", "1", "2", "3"],
            horizontal=True,
        )
        user_data["ca"] = int(ca_label)

    col1, col2, col3 = st.columns(3)

    with col1:
        sex_label = st.radio(label="Sexo:", options=list(SEX_OPTIONS.keys()), horizontal=False)
        user_data["sex"] = SEX_OPTIONS[sex_label]

        exang_label = st.radio(
            label="¿Angina inducida por ejercicio?",
            options=list(YES_NO_OPTIONS.keys()),
            horizontal=False,
        )
        user_data["exang"] = YES_NO_OPTIONS[exang_label]
    with col2:
        chest_pain_label = st.radio(
            label="Tipo de dolor de pecho:",
            options=list(CHEST_PAIN_OPTIONS.keys()),
            horizontal=False,
        )
        user_data["chest_pain"] = CHEST_PAIN_OPTIONS[chest_pain_label]

        slope_label = st.radio(
            label="Pendiente del segmento ST en el pico del ejercicio:",
            options=list(SLOPE_OPTIONS.keys()),
            horizontal=False,
        )
        user_data["slope"] = SLOPE_OPTIONS[slope_label]
    with col3:
        rest_ecg_label = st.radio(
            label="Resultado del electrocardiograma en reposo:",
            options=list(REST_ECG_OPTIONS.keys()),
            horizontal=False,
        )
        user_data["rest_ecg"] = REST_ECG_OPTIONS[rest_ecg_label]

        thal_label = st.radio(
            label="Resultado de la prueba de esfuerzo con talio:",
            options=list(THAL_OPTIONS.keys()),
            horizontal=False,
        )
        user_data["thal"] = THAL_OPTIONS[thal_label]

    df = pd.DataFrame.from_dict(user_data, orient="index").T

    return df


@st.cache_resource
def load_model(model_file_path: str) -> Pipeline:
    """Load a model in joblib format from the models directory.

    Args:
        model_file_path (str): The path where the trained model is stored.

    Returns:
        Pipeline: The trained model, a scikit-learn Pipeline object.
    """
    with st.spinner("Cargando modelo..."):
        model = load(model_file_path)

    return model


def main() -> None:
    model_name = "corazon_classification-model-v1.joblib"

    this_file_path = os.path.abspath(__file__)
    project_path = "/".join(this_file_path.split("/")[:-3])

    _col_img1, col_img2, _col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image("notebooks/7-deploy/images/corazon.jpg", use_container_width=True)

    st.header(body="¿Tiene el paciente riesgo de enfermedad cardíaca? ❤️")
    st.caption(
        "Esta es una predicción estadística basada en un modelo de Machine Learning "
        "(recall ≈ 0.85 sobre datos de prueba), no un diagnóstico médico."
    )

    model = load_model(model_file_path=project_path + "/models/" + model_name)

    df_user_data = get_user_data()

    if st.button("Predecir"):
        prediction = model.predict(df_user_data)[0]

        st.write("")
        if prediction == 0:
            st.success("El modelo predice: bajo riesgo de enfermedad cardíaca. 😀")
        else:
            st.error(
                "El modelo predice: riesgo de enfermedad cardíaca. "
                "Se recomienda evaluación médica. 😕"
            )


if __name__ == "__main__":
    main()
