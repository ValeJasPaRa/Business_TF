import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Predicción de Sueño",
    page_icon="🌙",
    layout="centered"
)

# ESTILOS
st.markdown("""
<style>
body {
    background-color: #f4f0fa;
}

div.stNumberInput, div.stSelectbox, div.stSlider, div.stRadio {
    background-color: #fdf7ff;
    border: 2px solid #4b0082;
    border-radius: 12px;
    padding: 10px;
}

.stButton > button {
    background-color: #7b2cbf;
    color: white;
    border-radius: 12px;
    border: 2px solid #ff9de2;
    font-size: 18px;
    padding: 10px 20px;
}

.stButton > button:hover {
    background-color: #ff9de2;
    color: black;
}

h1, h2, h3 {
    color: #4b0082;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TÍTULO

st.title("🌙 Predicción de Calidad de Sueño")
st.markdown("App basada en **XGBoost** para predecir la calidad del sueño")

st.divider()

# -------------------------------
# CARGA DE DATASET TRANSFORMADO

df = pd.read_csv("df_final_Transformados.csv")

# -------------------------------
# DEFINICIÓN DE VARIABLES

feature_columns = [
    'Caffeine_mg',
    'Stress_Level_Num',
    'Physical_Activity_Hours',
    'Smoking',
    'Alcohol_Consumption'
]

X = df[feature_columns]
y = df['Sleep_Quality']

# -------------------------------
# ENTRENAMIENTO DEL MODELO

xgb = XGBClassifier(
    n_estimators=100,
    random_state=0,
    max_depth=6,
    learning_rate=0.1
)

xgb.fit(X, y)

# -------------------------------
# FORMULARIO DE USUARIO

st.subheader("📝 Ingreso de Datos del Usuario")

col1, col2 = st.columns(2)

with col1:
    caffeine_mg = st.number_input("☕ Cafeína (mg)", min_value=0.0, max_value=1000.0, value=250.0)

    stress_level = st.selectbox(
        "pip  Nivel de Estrés",
        ["Low", "Medium", "High"]
    )

    physical_activity = st.slider(
        "🏃 Horas de Actividad Física",
        0.0, 24.0, 5.0
    )

with col2:
    smoking = st.radio(
        "🚬 Fumador",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí"
    )

    alcohol = st.radio(
        "🍺 Consumo de Alcohol",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí"
    )

# -------------------------------
# MAPEO DEL STRESS
stress_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

stress_level_num = stress_map[stress_level]

# -------------------------------
# DATAFRAME DE ENTRADA

input_data = pd.DataFrame([[
    caffeine_mg,
    stress_level_num,
    physical_activity,
    smoking,
    alcohol
]], columns=feature_columns)

# -------------------------------

# PREDICCIÓN xgboost
st.divider()

sleep_quality_map = {
    0: "Poor",
    1: "Fair",
    2: "Good",
    3: "Excellent"
}

if st.button("🔮 Predecir Calidad de Sueño"):
    prediction_num = xgb.predict(input_data)[0]
    probabilities = xgb.predict_proba(input_data)

    prediction_text = sleep_quality_map[prediction_num]

    st.success(f"✅ **Calidad de Sueño Predicha:** {prediction_text}")

    st.subheader("📊 Probabilidades por Clase")
    
    class_labels = [sleep_quality_map[c] for c in xgb.classes_]

    proba_df = pd.DataFrame(
        probabilities,
        columns=class_labels
    )
    st.dataframe(proba_df)

    st.balloons()

    #streamlit run App_Business_2.py
