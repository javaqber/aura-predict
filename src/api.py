from alertas import enviar_alerta
from diagnostico import diagnosticar_rodamiento
from database import (
    guardar_lectura_rodamiento, guardar_lectura_prensa,
    obtener_emails_maquina, obtener_usuario_por_email
)
from auth import crear_token, verificar_token, verificar_password
import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, Depends, HTTPException, status
from dotenv import load_dotenv
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))


app = FastAPI(
    title="AuraPredict API",
    description="Predictive Maintenance AI — API para sensores IoT y Raspberry Pi",
    version="1.4.0"
)

# --- CARGA DEL MODELO ---
MODEL_PATH = os.path.join(os.path.dirname(
    __file__), '../models/rodamientos_mecanizado.joblib')
try:
    modelo_ia = joblib.load(MODEL_PATH)
    print("✅ Modelo de rodamientos cargado correctamente.")
except Exception as e:
    modelo_ia = None
    print(f"❌ Error al cargar el modelo: {e}")

# --- SEGURIDAD ---
security = HTTPBearer()


def get_usuario_actual(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifica el token JWT en cada petición protegida."""
    token = credentials.credentials
    payload = verificar_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado"
        )
    return payload

# --- ESQUEMAS ---


class DatosLogin(BaseModel):
    email:    str
    password: str


class DatosVibracion(BaseModel):
    maquina:      str = "Torno_CNC_1"
    RMS:          float
    Peak_to_Peak: float
    Kurtosis:     float
    Skewness:     float


class DatosPrensaExtrusion(BaseModel):
    maquina:                     str = "Prensa_1"
    Desviacion_Columnas_uE:      float
    Vibracion_Bomba_AltaFrec_dB: float
    Particulas_Aceite_ISO:       int

# --- ENDPOINTS ---


@app.get("/")
def read_root():
    return {"mensaje": "AuraPredict API está ONLINE", "version": "1.4.0"}


@app.post("/login")
def login(datos: DatosLogin):
    usuario = obtener_usuario_por_email(datos.email)
    if not usuario:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    id_, email, password_hash, nombre, rol, empresa_id, activo = usuario

    if not activo:
        raise HTTPException(status_code=403, detail="Usuario desactivado")

    if not verificar_password(datos.password, password_hash):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    token = crear_token({
        "sub":        email,
        "nombre":     nombre,
        "rol":        rol,
        "empresa_id": empresa_id
    })

    return {
        "access_token": token,
        "token_type":   "bearer",
        "nombre":       nombre,
        "rol":          rol,
        "empresa_id":   empresa_id
    }


@app.post("/predict/bearing")
def predecir_rodamiento(datos: DatosVibracion,
                        current_user: dict = Depends(get_usuario_actual)):
    if modelo_ia is None:
        return {"error": "Modelo no disponible."}

    df = pd.DataFrame([{
        "RMS":          datos.RMS,
        "Peak_to_Peak": datos.Peak_to_Peak,
        "Kurtosis":     datos.Kurtosis,
        "Skewness":     datos.Skewness
    }])
    prediccion = modelo_ia.predict(df)[0]

    estado = "OK - Sano" if prediccion == 1 else "NOK - Anomalía Detectada"
    riesgo = "Bajo" if prediccion == 1 else "CRÍTICO - Parar Máquina"

    diagnostico = diagnosticar_rodamiento(
        rms=datos.RMS,
        peak_to_peak=datos.Peak_to_Peak,
        kurtosis=datos.Kurtosis,
        skewness=datos.Skewness
    )

    guardar_lectura_rodamiento(
        maquina=datos.maquina,
        rms=datos.RMS,
        peak_to_peak=datos.Peak_to_Peak,
        kurtosis=datos.Kurtosis,
        skewness=datos.Skewness,
        resultado=estado,
        nivel_riesgo=riesgo,
        diagnostico=diagnostico["tipo_fallo"]
    )

    if prediccion != 1:
        emails_cliente = obtener_emails_maquina(datos.maquina)
        enviar_alerta(
            maquina=datos.maquina,
            estado=estado,
            riesgo=riesgo,
            diagnostico=diagnostico,
            valores={
                "RMS":          datos.RMS,
                "Peak_to_Peak": datos.Peak_to_Peak,
                "Kurtosis":     datos.Kurtosis,
                "Skewness":     datos.Skewness
            },
            destinatarios_extra=emails_cliente
        )

    return {
        "maquina":        datos.maquina,
        "estado_maquina": estado,
        "nivel_riesgo":   riesgo,
        "diagnostico":    diagnostico
    }


@app.post("/predict/extrusion_press")
def predecir_prensa(datos: DatosPrensaExtrusion,
                    current_user: dict = Depends(get_usuario_actual)):
    if datos.Desviacion_Columnas_uE > 15.0:
        estado = "NOK - Desalineación Estructural"
        riesgo = "CRÍTICO - Peligro de rotura de columna"
    elif datos.Vibracion_Bomba_AltaFrec_dB > 85.0 or datos.Particulas_Aceite_ISO > 18:
        estado = "NOK - Desgaste Hidráulico"
        riesgo = "ALTO - Filtrar aceite / Revisar bomba"
    else:
        estado = "OK - Estructura y Sistema Hidráulico Sanos"
        riesgo = "Bajo"

    guardar_lectura_prensa(
        maquina=datos.maquina,
        desviacion=datos.Desviacion_Columnas_uE,
        vibracion=datos.Vibracion_Bomba_AltaFrec_dB,
        particulas=datos.Particulas_Aceite_ISO,
        resultado=estado,
        nivel_riesgo=riesgo
    )

    return {
        "maquina":        datos.maquina,
        "estado_maquina": estado,
        "nivel_riesgo":   riesgo
    }
