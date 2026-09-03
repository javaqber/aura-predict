"""
AuraPredict — alertas_v2.py  (Fase 3)
=======================================
Bridge between EdgePipeline (Fases 2B-2D) and the existing alertas.py SMTP system.

Responsibilities:
  1. Check cooldown using alert_log in PostgreSQL (persistent across restarts).
     Replaces the in-memory _ultimo_envio dict in alertas.py.
  2. Register every alert attempt in alert_log regardless of email outcome.
  3. If EMAIL_ACTIVO=true: send email via alertas.enviar_alerta() (Gmail SMTP,
     no new dependencies).
  4. If BD is offline: fall back to alertas.py's in-memory cooldown so the
     Edge never silently drops an alert because the DB is down.

What this module does NOT do:
  - Implement SMTP (delegated to alertas.py unchanged).
  - Read sensors or compute anomaly scores (delegated to EdgePipeline).
  - Modify alertas.py, scheduler.py, or any legacy module.

Usage (called from EdgePipeline._maybe_send_alert in Fase 3 Step 2):

    from alertas_v2 import maybe_enviar_alerta_cnc
    maybe_enviar_alerta_cnc(
        maquina_id    = 42,
        empresa_id    = 1,
        machine_name  = "Torno_CNC_1",
        anomaly_result= ar,
        feature_set   = fs,
        cooldown_hours= 1.0,
        destinatarios = ["responsable@empresa.com"],
    )
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Add src/ to path so sibling modules are importable
_SRC = os.path.dirname(__file__)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if TYPE_CHECKING:
    from edge.anomaly.anomaly_detector import AnomalyResult
    from edge.pipeline.models import FeatureSet


# ── Mapping from nivel_riesgo to alertas.py "nivel_urgencia" ─────────────────
# alertas.py uses: verde / amarillo / naranja / rojo
# EdgePipeline uses: Bajo / Medio / Alto / CRÍTICO
_RIESGO_TO_URGENCIA: dict[str, str] = {
    "Bajo":    "verde",
    "Medio":   "amarillo",
    "Alto":    "naranja",
    "CRÍTICO": "rojo",
}

# Only these risk levels trigger an alert
_ALERT_LEVELS = {"Alto", "CRÍTICO"}


def maybe_enviar_alerta_cnc(
    maquina_id:    int,
    empresa_id:    int,
    machine_name:  str,
    anomaly_result: "AnomalyResult",
    feature_set:   "FeatureSet",
    cooldown_hours: float = 1.0,
    destinatarios:  Optional[list[str]] = None,
    *,
    # Injectable for tests — None = real implementations
    puede_enviar_fn=None,
    registrar_fn=None,
) -> bool:
    """
    Evaluate and optionally send an alert for a CNC anomaly event.

    Flow:
      1. Guard: skip if cold start, low risk, or no anomaly result.
      2. Check cooldown:
           - Primary:  puede_enviar_alerta() from alert_log in BD (persistent).
           - Fallback: alertas._puede_enviar() in-memory if BD is offline.
      3. Register in alert_log (always, before attempting SMTP).
      4. Send email via alertas.enviar_alerta() if EMAIL_ACTIVO=true.
      5. Update alert_log.enviado based on SMTP outcome.

    Args:
        maquina_id:     Integer PK in maquinas table.
        empresa_id:     Integer PK in empresas table.
        machine_name:   Human-readable machine name for email body.
        anomaly_result: AnomalyResult from the anomaly detection engine.
        feature_set:    FeatureSet from this acquisition cycle (for sensor values).
        cooldown_hours: Minimum hours between alerts for this machine.
        destinatarios:  Email recipients. Merged with EMAIL_DESTINO from .env.
        puede_enviar_fn: Injectable for tests (replaces puede_enviar_alerta).
        registrar_fn:   Injectable for tests (replaces registrar_alerta).

    Returns:
        True if an alert was processed (cooldown passed), False otherwise.
    """
    ar = anomaly_result

    # ── Guard conditions ──────────────────────────────────────────────────────
    if ar is None:
        return False
    if ar.is_cold_start:
        return False
    if ar.nivel_riesgo not in _ALERT_LEVELS:
        return False
    if ar.health_score is None:
        return False  # SENSOR_ERROR — not actionable

    # ── Cooldown check ────────────────────────────────────────────────────────
    puede = _check_cooldown(maquina_id, cooldown_hours, puede_enviar_fn)
    if not puede:
        return False

    # ── Prepare alert content ─────────────────────────────────────────────────
    tipo       = "CRITICAL" if ar.nivel_riesgo == "CRÍTICO" else "WARNING"
    urgencia   = _RIESGO_TO_URGENCIA.get(ar.nivel_riesgo, "naranja")
    destinatario_primary = (destinatarios[0] if destinatarios else
                            os.getenv("EMAIL_DESTINO", ""))
    asunto = (f"AuraPredict {tipo}: {machine_name} "
              f"— health={ar.health_score} ({ar.nivel_riesgo})")

    diagnostico = _build_diagnostico(ar, urgencia)
    valores     = _build_valores(feature_set)

    # ── Register in alert_log BEFORE attempting SMTP ──────────────────────────
    alert_id = _registrar(
        maquina_id, empresa_id, tipo, destinatario_primary, asunto,
        enviado=False,
        error_msg=None,
        registrar_fn=registrar_fn,
    )

    # ── Send email via existing alertas.py ────────────────────────────────────
    email_activo = os.getenv("EMAIL_ACTIVO", "false").lower() == "true"
    enviado      = False
    error_msg    = None

    if email_activo:
        try:
            from alertas import enviar_alerta as _enviar_legacy
            _enviar_legacy(
                maquina    = machine_name,
                estado     = ar.resultado,
                riesgo     = ar.nivel_riesgo,
                diagnostico= diagnostico,
                valores    = valores,
                destinatarios_extra=destinatarios,
            )
            enviado   = True
        except Exception as exc:
            error_msg = str(exc)[:200]
            logger.error("SMTP failed for %s: %s", machine_name, exc)
    else:
        error_msg = "EMAIL_ACTIVO=false — alert logged but not emailed"

    # ── Update alert_log with final outcome ───────────────────────────────────
    if alert_id is not None:
        _update_alert_sent(alert_id, enviado, error_msg)

    return True


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_cooldown(
    maquina_id:     int,
    cooldown_hours: float,
    puede_enviar_fn,
) -> bool:
    """
    Check whether enough time has passed since the last alert.

    Primary: puede_enviar_alerta() from BD (survives restarts).
    Fallback: alertas._puede_enviar() in-memory if BD is unavailable.
    """
    if puede_enviar_fn is not None:
        # Injected (tests)
        return puede_enviar_fn(maquina_id, cooldown_hours=cooldown_hours)

    try:
        from database_v2.repositories import puede_enviar_alerta
        return puede_enviar_alerta(maquina_id, cooldown_hours=cooldown_hours)
    except Exception:
        # BD offline — fall back to alertas.py in-memory cooldown
        try:
            from alertas import _puede_enviar
            return _puede_enviar(str(maquina_id))
        except Exception:
            logger.info(
        "Alert processed for maquina_id=%d: tipo=%s, enviado=%s",
        maquina_id, tipo, enviado,
    )
    return True  # if both fail, allow the alert (safe default)


def _registrar(
    maquina_id:     int,
    empresa_id:     int,
    tipo:           str,
    destinatario:   str,
    asunto:         str,
    enviado:        bool,
    error_msg:      Optional[str],
    registrar_fn,
) -> Optional[int]:
    """Insert a row into alert_log. Returns the new id or None."""
    if registrar_fn is not None:
        return registrar_fn(maquina_id, empresa_id, tipo, destinatario,
                            asunto, enviado, error_msg)
    try:
        from database_v2.repositories import registrar_alerta
        return registrar_alerta(
            maquina_id   = maquina_id,
            empresa_id   = empresa_id,
            tipo_alerta  = tipo,
            destinatario = destinatario,
            asunto       = asunto,
            enviado      = enviado,
            error_msg    = error_msg,
        )
    except Exception as exc:
        logger.warning("Could not register alert in BD: %s", exc)
        return None


def _update_alert_sent(
    alert_id:  int,
    enviado:   bool,
    error_msg: Optional[str],
) -> None:
    """Update alert_log.enviado after SMTP attempt."""
    try:
        _src = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from database_v2.repositories import get_conn
        conn = get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("""
                UPDATE alert_log
                SET enviado=%s, error_msg=%s
                WHERE id=%s
            """, (enviado, error_msg, alert_id))
            conn.commit()
        finally:
            cur.close()
            conn.close()
    except Exception:
        pass  # BD offline — log entry stays with enviado=False


def _build_diagnostico(ar: "AnomalyResult", urgencia: str) -> dict:
    """
    Build the diagnostico dict expected by alertas._construir_email().
    Maps AnomalyResult fields to the legacy alertas.py schema.
    """
    return {
        "tipo_fallo":          ar.resultado,
        "componente_afectado": "Sistema CNC v2",
        "descripcion":         ar.diagnostico or f"Health score: {ar.health_score}/100",
        "consecuencias":       _consequence_for(ar.nivel_riesgo),
        "accion_recomendada":  _action_for(ar.nivel_riesgo),
        "pieza_referencia":    "—",
        "ventana_actuacion":   _window_for(ar.nivel_riesgo),
        "nivel_urgencia":      urgencia,
        "confianza":           f"Anomaly score: {ar.anomaly_score:.2f}",
    }


def _build_valores(fs: "FeatureSet") -> dict:
    """
    Build the valores dict expected by alertas._construir_email().
    Extracts primary axis time-domain features from the FeatureSet.
    """
    try:
        vf = fs.multiaxis.get_axis(fs.primary_axis)
        if vf:
            return {
                "RMS":          round(vf.time.rms, 4),
                "Peak_to_Peak": round(vf.time.peak_to_peak, 4),
                "Kurtosis":     round(vf.time.kurtosis, 4),
                "Skewness":     round(vf.time.skewness, 4),
            }
    except Exception:
        pass
    return {"RMS": "—", "Peak_to_Peak": "—", "Kurtosis": "—", "Skewness": "—"}


# ── Risk-level text tables ────────────────────────────────────────────────────

def _consequence_for(nivel_riesgo: str) -> str:
    return {
        "CRÍTICO": "Riesgo inminente de fallo. Intervención urgente necesaria.",
        "Alto":    "Degradación significativa. Planificar inspección próxima.",
    }.get(nivel_riesgo, "Supervisar evolución del sistema.")


def _action_for(nivel_riesgo: str) -> str:
    return {
        "CRÍTICO": "Parar máquina e inspeccionar inmediatamente.",
        "Alto":    "Programar revisión técnica en las próximas 24-48 horas.",
    }.get(nivel_riesgo, "Continuar monitorización.")


def _window_for(nivel_riesgo: str) -> str:
    return {
        "CRÍTICO": "Inmediato — menos de 8 horas",
        "Alto":    "24-48 horas",
    }.get(nivel_riesgo, "Sin urgencia inmediata")
