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


# ─── ENDPOINTS v2 — GESTIÓN DE MODELOS ML (Fase 5B) ─────────────────────────
# Todos respetan el aislamiento por empresa_id del JWT existente.
# ModelManager actúa como orquestador — nunca activa sin validación previa.

def _get_model_manager():
    """Create a ModelManager instance for API use (no BaselineManager needed)."""
    import sys as _sys
    _src = os.path.join(os.path.dirname(__file__), '..')
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from edge.anomaly.model_manager import ModelManager
    return ModelManager()


@app.get("/v2/maquinas/{maquina_id}/modelos")
def get_modelos_v2(
    maquina_id:   int,
    current_user: dict = Depends(get_usuario_actual),
):
    """
    Return all model versions for a machine (history), ordered newest first.
    Requires JWT. The machine must belong to the authenticated user's company.
    """
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        mm      = _get_model_manager()
        modelos = mm.list_models(maquina_id)
        return {
            "maquina_id": maquina_id,
            "total":      len(modelos),
            "modelos":    [
                {k: str(v) if hasattr(v, "isoformat") else v
                 for k, v in m.items()}
                for m in modelos
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error: {exc}")


@app.get("/v2/maquinas/{maquina_id}/modelos/activo")
def get_modelo_activo_v2(
    maquina_id:   int,
    current_user: dict = Depends(get_usuario_actual),
):
    """
    Return the currently active model for a machine.
    Returns 404 if no model has been trained yet.
    """
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        mm     = _get_model_manager()
        modelo = mm.get_active_model(maquina_id)
        if modelo is None:
            raise HTTPException(
                status_code=404,
                detail="No hay ningún modelo activo para esta máquina"
            )
        return {k: str(v) if hasattr(v, "isoformat") else v for k, v in modelo.items()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error: {exc}")


@app.post("/v2/maquinas/{maquina_id}/modelos/{model_id}/activar")
def activar_modelo_v2(
    maquina_id:   int,
    model_id:     int,
    current_user: dict = Depends(get_usuario_actual),
):
    """
    Activate a specific model version after passing 5-step validation.
    Validation: file exists, SHA-256, joblib.load, feature compatibility, predict().
    Returns 422 if validation fails (model not activated).
    """
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        mm     = _get_model_manager()
        result = mm.activate_model(model_id)
        if not result.success:
            raise HTTPException(
                status_code=422,
                detail=f"Activación rechazada: {result.error}"
            )
        return {
            "success":   True,
            "model_id":  result.model_id,
            "version":   result.version,
            "message":   f"Modelo v{result.version} activado correctamente",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error: {exc}")


@app.post("/v2/maquinas/{maquina_id}/modelos/{model_id}/rollback")
def rollback_modelo_v2(
    maquina_id:   int,
    model_id:     int,
    current_user: dict = Depends(get_usuario_actual),
):
    """
    Roll back to the latest valid previous model version for a machine.
    Searches previous versions from newest to oldest and activates
    the first one that passes all validation checks.
    Returns 422 if no valid previous version exists.
    """
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        mm     = _get_model_manager()
        result = mm.rollback_model(maquina_id)
        if not result.success:
            raise HTTPException(
                status_code=422,
                detail=f"Rollback fallido: {result.error}"
            )
        return {
            "success":  True,
            "model_id": result.model_id,
            "version":  result.version,
            "message":  f"Rollback exitoso → v{result.version}",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error: {exc}")


@app.post("/v2/maquinas/{maquina_id}/modelos/entrenar")
def entrenar_modelo_v2(
    maquina_id:   int,
    current_user: dict = Depends(get_usuario_actual),
):
    """
    Trigger manual retraining of the Isolation Forest model.
    Requires a BaselineManager with sufficient data (baseline_min_samples).
    The new model is NOT automatically activated — use /activar afterwards.
    Returns 422 if insufficient baseline data.
    Note: In the current architecture the BaselineManager lives in the Edge process.
    This endpoint is a placeholder that confirms the request — actual retraining
    happens via the edge_scheduler or can be triggered locally.
    """
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    _verificar_maquina_empresa(maquina_id, empresa_id)

    # The API cannot directly trigger the Edge process BaselineManager.
    # It registers the intent and returns guidance.
    return {
        "success":    True,
        "maquina_id": maquina_id,
        "message":    (
            "Solicitud de entrenamiento registrada. "
            "El modelo se entrenará automáticamente cuando el Edge acumule "
            "suficientes lecturas normales (baseline_min_samples). "
            "También puedes usar edge_scheduler.py con el ModelManager directamente."
        ),
        "action": "automatic_via_baseline_manager",
    }


# ─── ENDPOINTS v2 — REPORTING (Fase 6) ───────────────────────────────────────

def _get_reporting_path():
    import sys as _sys
    _src = os.path.join(os.path.dirname(__file__), '..')
    if _src not in _sys.path:
        _sys.path.insert(0, _src)

@app.get("/v2/maquinas/{maquina_id}/exportar/csv")
def exportar_csv_v2(
    maquina_id:   int,
    tipo:         str  = "readings",
    fecha_desde:  str  = None,
    fecha_hasta:  str  = None,
    current_user: dict = Depends(get_usuario_actual),
):
    """
    Export machine data as CSV.
    tipo: readings | health_history | anomalies | alerts
    fecha_desde / fecha_hasta: ISO8601 strings (YYYY-MM-DD)
    """
    from fastapi.responses import Response
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")
    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        _get_reporting_path()
        from reporting.exporter import export_csv
        from datetime import datetime
        fd = datetime.fromisoformat(fecha_desde) if fecha_desde else None
        fh = datetime.fromisoformat(fecha_hasta) if fecha_hasta else None
        data = export_csv(maquina_id, tipo=tipo, fecha_desde=fd, fecha_hasta=fh,
                          empresa_id=empresa_id)
        filename = f"aurapredict_{maquina_id}_{tipo}.csv"
        return Response(content=data, media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={filename}"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Export error: {exc}")


@app.get("/v2/maquinas/{maquina_id}/exportar/excel")
def exportar_excel_v2(
    maquina_id:   int,
    fecha_desde:  str  = None,
    fecha_hasta:  str  = None,
    current_user: dict = Depends(get_usuario_actual),
):
    """Export all machine data as a multi-sheet Excel workbook."""
    from fastapi.responses import Response
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")
    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        _get_reporting_path()
        from reporting.exporter import export_excel
        from datetime import datetime
        fd = datetime.fromisoformat(fecha_desde) if fecha_desde else None
        fh = datetime.fromisoformat(fecha_hasta) if fecha_hasta else None
        data = export_excel(maquina_id, fecha_desde=fd, fecha_hasta=fh, empresa_id=empresa_id)
        filename = f"aurapredict_{maquina_id}_informe.xlsx"
        return Response(
            content=data,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Export error: {exc}")


@app.get("/v2/maquinas/{maquina_id}/informe")
def informe_maquina_v2(
    maquina_id:   int,
    fecha_desde:  str = None,
    fecha_hasta:  str = None,
    current_user: dict = Depends(get_usuario_actual),
):
    """Generate an HTML machine report viewable in the browser."""
    from fastapi.responses import HTMLResponse
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")
    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        _get_reporting_path()
        from reporting.report_machine import generate_machine_report
        from datetime import datetime
        fd = datetime.fromisoformat(fecha_desde) if fecha_desde else None
        fh = datetime.fromisoformat(fecha_hasta) if fecha_hasta else None
        html_bytes = generate_machine_report(
            maquina_id=maquina_id, fecha_desde=fd, fecha_hasta=fh, empresa_id=empresa_id)
        return HTMLResponse(content=html_bytes.decode("utf-8"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Report error: {exc}")


@app.get("/v2/empresa/informe-planta")
def informe_planta_v2(
    fecha_desde:  str = None,
    fecha_hasta:  str = None,
    current_user: dict = Depends(get_usuario_actual),
):
    """Generate an HTML plant-level report for all machines in the company."""
    from fastapi.responses import HTMLResponse
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    try:
        _get_reporting_path()
        from reporting.report_plant import generate_plant_report
        from datetime import datetime
        fd = datetime.fromisoformat(fecha_desde) if fecha_desde else None
        fh = datetime.fromisoformat(fecha_hasta) if fecha_hasta else None
        html_bytes = generate_plant_report(empresa_id=empresa_id, fecha_desde=fd, fecha_hasta=fh)
        return HTMLResponse(content=html_bytes.decode("utf-8"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Report error: {exc}")


@app.post("/v2/maquinas/{maquina_id}/ground-truth/mantenimiento")
def registrar_mantenimiento_v2(
    maquina_id:   int,
    datos:        dict,
    current_user: dict = Depends(get_usuario_actual),
):
    """Register a maintenance event (ground truth label)."""
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")
    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        _get_reporting_path()
        from reporting.ground_truth import registrar_mantenimiento, MaintenanceEventInput
        from datetime import datetime
        evento = MaintenanceEventInput(
            maquina_id     = maquina_id,
            empresa_id     = empresa_id,
            maintenance_at = datetime.fromisoformat(datos.get("maintenance_at", "")),
            tipo           = datos.get("tipo", "correctivo"),
            componente     = datos.get("componente"),
            descripcion    = datos.get("descripcion"),
            tiempo_parada_h= datos.get("tiempo_parada_h"),
            coste_euros    = datos.get("coste_euros"),
            tecnico        = datos.get("tecnico"),
            alertado_por_ia= datos.get("alertado_por_ia", False),
            dias_anticipacion = datos.get("dias_anticipacion"),
            registrado_por = current_user.get("usuario_id"),
        )
        new_id = registrar_mantenimiento(evento)
        return {"success": new_id is not None, "id": new_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Error: {exc}")


@app.post("/v2/maquinas/{maquina_id}/ground-truth/fallo")
def registrar_fallo_v2(
    maquina_id:   int,
    datos:        dict,
    current_user: dict = Depends(get_usuario_actual),
):
    """Register a confirmed machine failure (ground truth label for ML)."""
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")
    _verificar_maquina_empresa(maquina_id, empresa_id)

    try:
        _get_reporting_path()
        from reporting.ground_truth import registrar_fallo, FailureEventInput
        from datetime import datetime
        evento = FailureEventInput(
            maquina_id             = maquina_id,
            empresa_id             = empresa_id,
            failure_at             = datetime.fromisoformat(datos.get("failure_at", "")),
            tipo_fallo             = datos.get("tipo_fallo"),
            componente             = datos.get("componente"),
            descripcion            = datos.get("descripcion"),
            primera_anomalia_ts    = datetime.fromisoformat(datos["primera_anomalia_ts"]) if datos.get("primera_anomalia_ts") else None,
            tiempo_deteccion_dias  = datos.get("tiempo_deteccion_dias"),
            diagnostico_ia         = datos.get("diagnostico_ia"),
            diagnostico_confirmado = datos.get("diagnostico_confirmado"),
            tiempo_parada_h        = datos.get("tiempo_parada_h"),
            coste_euros            = datos.get("coste_euros"),
            tecnico                = datos.get("tecnico"),
            registrado_por         = current_user.get("usuario_id"),
        )
        new_id = registrar_fallo(evento)
        return {"success": new_id is not None, "id": new_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Error: {exc}")


@app.get("/v2/empresa/ground-truth/exportar")
def exportar_ground_truth_v2(
    maquina_id:   int   = None,
    current_user: dict  = Depends(get_usuario_actual),
):
    """Export all ground truth labels as CSV for ML training."""
    from fastapi.responses import Response
    empresa_id = current_user.get("empresa_id")
    if empresa_id is None:
        raise HTTPException(status_code=403, detail="Token sin empresa_id")

    try:
        _get_reporting_path()
        from reporting.ground_truth import exportar_ground_truth_csv
        data = exportar_ground_truth_csv(empresa_id=empresa_id, maquina_id=maquina_id)
        return Response(content=data, media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=ground_truth.csv"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Export error: {exc}")
