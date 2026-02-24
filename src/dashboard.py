import streamlit as st
import requests

st.set_page_config(page_title="AuraPredict", page_icon="⚙️", layout="centered")
# st.title("AuraPredict: Mantenimiento Predictivo")
# st.write("Monitorización con IA para Activos Industriales Críticos")

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
    /* cursor pointer en el selectbox y botones */
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    button {
        cursor: pointer !important;
    }
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

    endpoint = "http://127.0.0.1:8000/predict/bearing"
    payload = {"RMS": rms_val, "Peak_to_Peak": p2p_val,
               "Kurtosis": kurt_val, "Skewness": skew_val}

elif tipo_maquina == "Prensa de Extrusión de Aluminio":
    st.subheader("🏗️ Monitorización de Fatiga Estructural e Hidráulica")
    st.write("*Sensores IoT independientes del HMI del operario*")
    col1, col2, col3 = st.columns(3)
    with col1:
        columnas = st.number_input("Desviación Columnas (µε)", value=2.5, format="%.1f",
                                   help="Diferencia de tensión entre columnas (Galgas extensiométricas)")
    with col2:
        bomba = st.number_input("Firma Acústica Bomba (dB)", value=65.0,
                                format="%.1f", help="Ultrasonidos en bomba principal")
    with col3:
        aceite = st.number_input("Partículas Aceite (ISO)", value=14,
                                 format="%d", help="Contaminación sólida en tanque")

    endpoint = "http://127.0.0.1:8000/predict/extrusion_press"
    payload = {"Desviacion_Columnas_uE": columnas,
               "Vibracion_Bomba_AltaFrec_dB": bomba, "Particulas_Aceite_ISO": aceite}

st.markdown("---")

# --- BOTÓN DE EJECUCIÓN ---
if st.button("🔍 Analizar Estado del Activo", use_container_width=True):
    try:
        respuesta = requests.post(endpoint, json=payload)

        if respuesta.status_code == 200:
            resultado = respuesta.json()

            if resultado["estado_maquina"].startswith("OK"):
                st.success(f"✅ ESTADO: {resultado['estado_maquina']}")
            else:
                st.error(f"🚨 ESTADO: {resultado['estado_maquina']}")
                st.warning(f"⚠️ RIESGO: {resultado['nivel_riesgo']}")
        else:
            st.error("Error al comunicarse con la API de AuraPredict.")

    except requests.exceptions.ConnectionError:
        st.error("🚨 ERROR FATAL: No puedo encontrar el cerebro (API).")
