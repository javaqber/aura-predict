import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CARGAMOS LA IA DIRECTAMENTE EN STREAMLIT ---


@st.cache_resource
def cargar_modelo():
    # Buscamos el modelo en la carpeta models
    ruta_modelo = os.path.join(os.path.dirname(
        __file__), '../models/rodamientos_mecanizado.joblib')

    if os.path.exists(ruta_modelo):
        return joblib.load(ruta_modelo)
    else:
        # Si la ruta falla, intentamos una ruta relativa simple
        ruta_alternativa = "models/rodamientos_mecanizado.joblib"
        if os.path.exists(ruta_alternativa):
            return joblib.load(ruta_alternativa)
        return None


modelo_ia = cargar_modelo()

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AuraPredict", page_icon="⚙️", layout="centered")

# --- TÍTULO PERSONALIZADO ---
st.markdown("""
    <div style="text-align: center; padding-top: 20px; padding-bottom: 20px;">
        <h1 style="font-size: 4rem; font-weight: 800; margin-bottom: 0; color: #0A192F; font-family: 'Arial', sans-serif;">
            <span style="color: #0B3D91;">Aura</span>Predict <span style="font-size: 2.5rem;"></span>
        </h1>
        <p style="font-size: 1.2rem; color: #666666; font-weight: 400; margin-top: -10px;">
            Monitorización Predictiva para Activos Industriales Críticos
        </p>
    </div>
""", unsafe_allow_html=True)

# --- SELECTOR CURSOR: POINTER ---
st.markdown("""
    <style>
    div[data-baseweb="select"] > div { cursor: pointer !important; }
    button { cursor: pointer !important; }
    </style>
""", unsafe_allow_html=True)

# --- SELECTOR DE MÁQUINA ---
tipo_maquina = st.selectbox(
    "Selecciona el activo industrial a monitorizar:",
    ("Torno CNC / Fresadora (Vibración de Rodamiento)",
     "Prensa de Extrusión de Aluminio")
)

st.markdown("---")

# --- LÓGICA DE LA INTERFAZ SEGÚN LA MÁQUINA ---
if tipo_maquina == "Torno CNC / Fresadora (Vibración de Rodamiento)":
    st.subheader("📡 Sensor de Análisis de Vibración")
    col1, col2 = st.columns(2)
    with col1:
        rms_val = st.number_input("RMS (Fuerza)", value=0.07, format="%.4f")
        p2p_val = st.number_input("Peak-to-Peak", value=0.84, format="%.4f")
    with col2:
        kurt_val = st.number_input(
            "Kurtosis (Impactos)", value=0.62, format="%.4f")
        skew_val = st.number_input("Skewness", value=0.08, format="%.4f")

elif tipo_maquina == "Prensa de Extrusión de Aluminio":
    st.subheader("🏗️ Monitorización de Fatiga Estructural e Hidráulica")
    st.write("*Sensores IoT independientes del HMI del operario*")
    col1, col2, col3 = st.columns(3)
    with col1:
        columnas = st.number_input("Desviación Columnas (µε)", value=2.5,
                                   format="%.1f", help="Diferencia de tensión entre columnas")
    with col2:
        bomba = st.number_input("Firma Acústica Bomba (dB)", value=65.0,
                                format="%.1f", help="Ultrasonidos en bomba principal")
    with col3:
        aceite = st.number_input("Partículas Aceite (ISO)", value=14,
                                 format="%d", help="Contaminación sólida en tanque")

st.markdown("---")

# --- BOTÓN DE EJECUCIÓN ---
if st.button("🔍 Analizar Estado del Activo", use_container_width=True):

    # 1. TORNO MECANIZADO (Usa la IA de Scikit-Learn)
    if tipo_maquina == "Torno CNC / Fresadora (Vibración de Rodamiento)":
        if modelo_ia is None:
            st.error(
                "🚨 ERROR: No se ha encontrado el archivo models/isolation_forest.joblib. Revisa la ruta en GitHub.")
        else:
            df_entrada = pd.DataFrame([{
                "RMS": rms_val,
                "Peak_to_Peak": p2p_val,
                "Kurtosis": kurt_val,
                "Skewness": skew_val
            }])

            # Predicción instantánea
            prediccion = modelo_ia.predict(df_entrada)[0]

            if prediccion == 1:
                st.success("✅ ESTADO: OK - Sano")
                st.info("🟢 RIESGO: Bajo")
            else:
                st.error("🚨 ESTADO: NOK - Anomalía Detectada")
                st.warning("⚠️ RIESGO: CRÍTICO - Parar Máquina")

    # 2. CASO PRENSA (Usa la Lógica de Reglas de Negocio Simulada)
    elif tipo_maquina == "Prensa de Extrusión de Aluminio":
        # Evaluamos las reglas directamente aquí
        if columnas > 15.0:
            st.error("🚨 ESTADO: NOK - Desalineación Estructural")
            st.warning("⚠️ RIESGO: CRÍTICO - Peligro de rotura de columna")
        elif bomba > 85.0 or aceite > 18:
            st.error("🚨 ESTADO: NOK - Desgaste Hidráulico")
            st.warning("⚠️ RIESGO: ALTO - Filtrar aceite / Revisar bomba")
        else:
            st.success("✅ ESTADO: OK - Estructura y Sistema Hidráulico Sanos")
            st.info("🟢 RIESGO: Bajo")
