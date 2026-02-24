from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="AuraPredict API",
              description="Predictive Maintenance AI", version="1.1.0")

# --- CARGA DEL MODELO DE RODAMIENTOS ---
MODEL_PATH = os.path.join(os.path.dirname(
    __file__), '../models/isolation_forest.joblib')
try:
    modelo_ia = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# --- 1. ESQUEMAS DE DATOS (CONTRATOS) ---


class DatosVibracion(BaseModel):
    RMS: float
    Peak_to_Peak: float
    Kurtosis: float
    Skewness: float


class DatosPrensaExtrusion(BaseModel):
    # Micro-strain: Diferencia de tensión entre las columnas
    Desviacion_Columnas_uE: float
    Vibracion_Bomba_AltaFrec_dB: float  # Ultrasonido de la bomba principal
    Particulas_Aceite_ISO: int  # Contaminación del aceite hidráulico

# --- 2. ENDPOINTS (RUTAS) ---


@app.post("/predict/bearing")
def predecir_rodamiento(datos: DatosVibracion):
    df_entrada = pd.DataFrame([datos.model_dump()])
    prediccion = modelo_ia.predict(df_entrada)[0]

    estado = "OK - Sano" if prediccion == 1 else "NOK - Anomalía Detectada"
    riesgo = "Bajo" if prediccion == 1 else "CRÍTICO - Parar Máquina"
    return {"estado_maquina": estado, "nivel_riesgo": riesgo}


@app.post("/predict/extrusion_press")
def predecir_prensa(datos: DatosPrensaExtrusion):
    # Lógica simulada de PdM real
    if datos.Desviacion_Columnas_uE > 15.0:
        return {"estado_maquina": "NOK - Desalineación Estructural", "nivel_riesgo": "CRÍTICO - Peligro de rotura de columna"}
    elif datos.Vibracion_Bomba_AltaFrec_dB > 85.0 or datos.Particulas_Aceite_ISO > 18:
        return {"estado_maquina": "NOK - Desgaste Hidráulico", "nivel_riesgo": "ALTO - Filtrar aceite / Revisar bomba"}

    return {"estado_maquina": "OK - Estructura y Sistema Hidráulico Sanos", "nivel_riesgo": "Bajo"}


@app.get("/")
def read_root():
    return {"mensaje": "AuraPredict API está ONLINE"}
