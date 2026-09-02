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


# ─── ENDPOINTS v2 (Sistema Edge — Fases 2B-2D) ───────────────────────────────
# Estos endpoints exponen datos de lecturas_cnc_v2, health_scores y alert_log.
# Son ADITIVOS: no modifican ni afectan a /predict/bearing ni /predict/extrusion_press.
#
# AISLAMIENTO MULTIEMPRESA (condición arquitectónica obligatoria — Fase 3):
# Cada endpoint verifica que la máquina solicitada pertenece a la empresa_id
# del token JWT antes de devolver cualquier dato.

def _verificar_maquina_empresa(maquina_id: int, empresa_id: int) -> None:
    """
    Raise HTTP 403 if maquina_id does not belong to empresa_id.
    Raise HTTP 404 if maquina_id does not exist.
    This enforces strict multicompany isolation for all v2 endpoints.
    """
    try:
        import sys as _sys
        _src = os.path.join(os.path.dirname(__file__), '..')
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from database_v2.repositories import obtener_empresa_id_de_maquina
        maquina_empresa = obtener_empresa_id_de_maquina(maquina_id)
        if maquina_empresa is None:
            raise HTTPException(status_code=404, detail="Máquina no encontrada")
        if maquina_empresa != empresa_id:
            raise HTTPException(
                status_code=403,
                detail="Acceso denegado: la máquina no pertenece a su empresa"
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error de base de datos: {exc}")


@app.get("/v2/maquinas/{maquina_id}/health")
def get_health_v2(
    maquina_id:   int,
    current_user: dict = Depends(get_usuario_actual),
):
    """
    Return the latest health score and trend for a CNC machine.

    Requires JWT. The machine must belong to the authenticated user's company.
    Returns 404 if no health data exists yet for this machine.
    """
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        import sys as _sys
        _src = os.path.join(os.path.dirname(__file__), '..')
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from database_v2.repositories import obtener_historial_health
        historial = obtener_historial_health(maquina_id, dias=1)
        if not historial:
            raise HTTPException(
                status_code=404,
                detail="Sin datos de health score para esta máquina"
            )
        ultimo = historial[0]
        return {
            "maquina_id":  maquina_id,
            "health_score": ultimo.get("score"),
            "trend":        ultimo.get("trend"),
            "slope":        ultimo.get("slope"),
            "timestamp":    str(ultimo.get("timestamp", "")),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error de base de datos: {exc}")


@app.get("/v2/maquinas/{maquina_id}/historial")
def get_historial_v2(
    maquina_id:   int,
    limite:       int  = 50,
    current_user: dict = Depends(get_usuario_actual),
):
    """
    Return recent CNC readings (lecturas_cnc_v2) for a machine.

    Requires JWT. The machine must belong to the authenticated user's company.
    Returns newest readings first. Limit capped at 200 for performance.
    """
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    _verificar_maquina_empresa(maquina_id, empresa_id)

    limite = min(max(1, limite), 200)  # 1–200

    try:
        import sys as _sys
        _src = os.path.join(os.path.dirname(__file__), '..')
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from database_v2.repositories import obtener_historial_cnc
        lecturas = obtener_historial_cnc(maquina_id, limite=limite)
        return {
            "maquina_id": maquina_id,
            "total":      len(lecturas),
            "lecturas":   [
                {k: str(v) if hasattr(v, "isoformat") else v for k, v in r.items()}
                for r in lecturas
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error de base de datos: {exc}")


@app.get("/v2/maquinas/{maquina_id}/anomalias")
def get_anomalias_v2(
    maquina_id:   int,
    dias:         int  = 7,
    current_user: dict = Depends(get_usuario_actual),
):
    """
    Return recent anomaly readings for a machine (resultado != 'OK - Sano').

    Requires JWT. The machine must belong to the authenticated user's company.
    Filters readings from the last N days. Capped at 30 days for performance.
    """
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    _verificar_maquina_empresa(maquina_id, empresa_id)

    dias = min(max(1, dias), 30)

    try:
        import sys as _sys
        _src = os.path.join(os.path.dirname(__file__), '..')
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from database_v2.repositories import obtener_historial_cnc
        todas = obtener_historial_cnc(maquina_id, limite=200)
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=dias)
        anomalias = []
        for r in todas:
            ts = r.get("timestamp")
            if ts:
                if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    continue
            if r.get("resultado", "").startswith("OK"):
                continue
            anomalias.append(
                {k: str(v) if hasattr(v, "isoformat") else v for k, v in r.items()}
            )
        return {
            "maquina_id": maquina_id,
            "dias":       dias,
            "total":      len(anomalias),
            "anomalias":  anomalias,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error de base de datos: {exc}")


@app.get("/v2/maquinas/resumen")
def get_resumen_v2(current_user: dict = Depends(get_usuario_actual)):
    """
    Return health status summary for all machines belonging to the
    authenticated user's company. Used by the dashboard multi-machine map.
    """
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")
    try:
        import sys as _sys
        _src = os.path.join(os.path.dirname(__file__), '..')
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from database_v2.repositories import obtener_todas_maquinas_con_health
        maquinas = obtener_todas_maquinas_con_health(empresa_id=empresa_id)
        return {
            "empresa_id": empresa_id,
            "total":      len(maquinas),
            "maquinas":   [
                {k: str(v) if hasattr(v, "isoformat") else v for k, v in m.items()}
                for m in maquinas
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error de base de datos: {exc}")
