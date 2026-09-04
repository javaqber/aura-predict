"""
AuraPredict — GroundTruth  (Fase 6)
======================================
Infrastructure for labelling real confirmed faults and maintenance events.

Purpose:
  - Technicians confirm or correct the system's automatic diagnoses
  - Confirmed labels become "ground truth" for future supervised ML training
  - The system tracks AI prediction accuracy over time

Database tables used (V2_007 and V2_008 — already exist):
  maintenance_events : all maintenance interventions performed
  failure_events     : actual machine failures that occurred

No new migrations are needed — these tables were designed in Fase 2A
specifically with supervised learning in mind.

Column mapping:
  maintenance_events.alertado_por_ia    : was the AI alert correct?
  maintenance_events.dias_anticipacion  : how many days of advance warning?
  failure_events.tipo_fallo             : confirmed fault type
  failure_events.primera_anomalia_ts    : when AI first detected something
  failure_events.tiempo_deteccion_dias  : AI detection lead time (days)
  failure_events.diagnostico_ia         : what the AI diagnosed
  failure_events.diagnostico_confirmado : what the technician confirms
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

@dataclass
class MaintenanceEventInput:
    """Input for registering a maintenance intervention."""
    maquina_id:        int
    empresa_id:        int
    maintenance_at:    datetime
    tipo:              str                # preventivo | correctivo | predictivo
    componente:        Optional[str]      = None
    descripcion:       Optional[str]      = None
    tiempo_parada_h:   Optional[float]    = None
    coste_euros:       Optional[float]    = None
    tecnico:           Optional[str]      = None
    alertado_por_ia:   bool               = False
    dias_anticipacion: Optional[int]      = None
    registrado_por:    Optional[int]      = None   # usuario_id


@dataclass
class FailureEventInput:
    """Input for registering a confirmed machine failure."""
    maquina_id:             int
    empresa_id:             int
    failure_at:             datetime
    tipo_fallo:             Optional[str]    = None  # bearing_fault|imbalance|lubrication|other
    componente:             Optional[str]    = None
    descripcion:            Optional[str]    = None
    # AI context
    primera_anomalia_ts:    Optional[datetime] = None
    tiempo_deteccion_dias:  Optional[int]    = None  # AI advance warning (days)
    diagnostico_ia:         Optional[str]    = None  # what the system diagnosed
    diagnostico_confirmado: Optional[str]    = None  # what the technician confirms
    # Impact
    tiempo_parada_h:        Optional[float]  = None
    coste_euros:            Optional[float]  = None
    tecnico:                Optional[str]    = None
    registrado_por:         Optional[int]    = None


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def registrar_mantenimiento(event: MaintenanceEventInput) -> Optional[int]:
    """
    Register a maintenance intervention in maintenance_events.

    Returns:
        New event id, or None if the DB call fails.
    """
    try:
        from database_v2.repositories import get_conn
        conn = get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO maintenance_events
                    (maquina_id, empresa_id, maintenance_at, tipo, componente,
                     descripcion, tiempo_parada_h, coste_euros, tecnico,
                     alertado_por_ia, dias_anticipacion, registrado_por)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (
                event.maquina_id, event.empresa_id,
                event.maintenance_at, event.tipo, event.componente,
                event.descripcion, event.tiempo_parada_h, event.coste_euros,
                event.tecnico, event.alertado_por_ia, event.dias_anticipacion,
                event.registrado_por,
            ))
            conn.commit()
            row = cur.fetchone()
            new_id = row[0] if row else None
            logger.info("Maintenance event registered: id=%s, maquina=%d, tipo=%s",
                        new_id, event.maquina_id, event.tipo)
            return new_id
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("Failed to register maintenance event: %s", exc)
        return None


def registrar_fallo(event: FailureEventInput) -> Optional[int]:
    """
    Register a confirmed machine failure in failure_events.
    This is the ground truth label used for supervised ML validation.

    The diagnostico_confirmado field is the most important for training:
    it records what the technician actually found vs what the AI predicted.

    Returns:
        New event id, or None if the DB call fails.
    """
    try:
        from database_v2.repositories import get_conn
        conn = get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO failure_events
                    (maquina_id, empresa_id, failure_at, tipo_fallo,
                     componente, descripcion,
                     primera_anomalia_ts, tiempo_deteccion_dias,
                     diagnostico_ia, diagnostico_confirmado,
                     tiempo_parada_h, coste_euros, tecnico, registrado_por)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (
                event.maquina_id, event.empresa_id,
                event.failure_at, event.tipo_fallo,
                event.componente, event.descripcion,
                event.primera_anomalia_ts, event.tiempo_deteccion_dias,
                event.diagnostico_ia, event.diagnostico_confirmado,
                event.tiempo_parada_h, event.coste_euros,
                event.tecnico, event.registrado_por,
            ))
            conn.commit()
            row = cur.fetchone()
            new_id = row[0] if row else None
            logger.info("Failure event registered: id=%s, maquina=%d, tipo=%s",
                        new_id, event.maquina_id, event.tipo_fallo)
            return new_id
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("Failed to register failure event: %s", exc)
        return None


def obtener_historial_mantenimiento(
    maquina_id:  int,
    empresa_id:  Optional[int] = None,
    limite:      int = 50,
) -> list[dict]:
    """
    Return maintenance history for a machine, newest first.
    Includes alertado_por_ia to measure AI prediction accuracy.
    """
    try:
        from database_v2.repositories import get_conn
        conn = get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("""
                SELECT id, maintenance_at, tipo, componente, descripcion,
                       tiempo_parada_h, coste_euros, tecnico,
                       alertado_por_ia, dias_anticipacion
                FROM maintenance_events
                WHERE maquina_id = %s
                  AND (%s IS NULL OR empresa_id = %s)
                ORDER BY maintenance_at DESC
                LIMIT %s
            """, (maquina_id, empresa_id, empresa_id, limite))
            cols = ["id","maintenance_at","tipo","componente","descripcion",
                    "tiempo_parada_h","coste_euros","tecnico",
                    "alertado_por_ia","dias_anticipacion"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("obtener_historial_mantenimiento failed: %s", exc)
        return []


def obtener_historial_fallos(
    maquina_id:  int,
    empresa_id:  Optional[int] = None,
    limite:      int = 50,
) -> list[dict]:
    """
    Return failure events for a machine (ground truth labels), newest first.
    """
    try:
        from database_v2.repositories import get_conn
        conn = get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("""
                SELECT id, failure_at, tipo_fallo, componente, descripcion,
                       primera_anomalia_ts, tiempo_deteccion_dias,
                       diagnostico_ia, diagnostico_confirmado,
                       tiempo_parada_h, coste_euros, tecnico
                FROM failure_events
                WHERE maquina_id = %s
                  AND (%s IS NULL OR empresa_id = %s)
                ORDER BY failure_at DESC
                LIMIT %s
            """, (maquina_id, empresa_id, empresa_id, limite))
            cols = ["id","failure_at","tipo_fallo","componente","descripcion",
                    "primera_anomalia_ts","tiempo_deteccion_dias",
                    "diagnostico_ia","diagnostico_confirmado",
                    "tiempo_parada_h","coste_euros","tecnico"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("obtener_historial_fallos failed: %s", exc)
        return []


def exportar_ground_truth_csv(
    empresa_id: int,
    maquina_id: Optional[int] = None,
) -> bytes:
    """
    Export all failure events as a CSV for ML training.
    This is the labelled dataset combining AI diagnoses with technician confirmations.

    Columns:
        failure_at, maquina_id, tipo_fallo, componente,
        primera_anomalia_ts, tiempo_deteccion_dias (AI advance warning),
        diagnostico_ia (what the AI predicted),
        diagnostico_confirmado (what the technician found — ground truth label)
    """
    import pandas as pd

    try:
        from database_v2.repositories import get_conn
        conn = get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("""
                SELECT fe.id, fe.failure_at, fe.maquina_id,
                       m.nombre AS maquina_nombre,
                       fe.tipo_fallo, fe.componente, fe.descripcion,
                       fe.primera_anomalia_ts, fe.tiempo_deteccion_dias,
                       fe.diagnostico_ia, fe.diagnostico_confirmado,
                       fe.tiempo_parada_h, fe.coste_euros, fe.tecnico
                FROM failure_events fe
                JOIN maquinas m ON m.id = fe.maquina_id
                WHERE fe.empresa_id = %s
                  AND (%s IS NULL OR fe.maquina_id = %s)
                ORDER BY fe.failure_at DESC
            """, (empresa_id, maquina_id, maquina_id))
            cols = ["id","failure_at","maquina_id","maquina_nombre",
                    "tipo_fallo","componente","descripcion",
                    "primera_anomalia_ts","tiempo_deteccion_dias",
                    "diagnostico_ia","diagnostico_confirmado",
                    "tiempo_parada_h","coste_euros","tecnico"]
            df = pd.DataFrame(cur.fetchall(), columns=cols)
            return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("exportar_ground_truth_csv failed: %s", exc)
        return b"id,error\n0,Error al exportar ground truth\n"


def calcular_metricas_ia(empresa_id: int) -> dict:
    """
    Calculate AI prediction accuracy metrics from ground truth labels.

    Returns:
        {
          'total_fallos':          int,
          'detectados_por_ia':     int,
          'tasa_deteccion':        float (0-1),
          'anticipacion_media_dias': float,
          'tipos_fallo':           {tipo: count},
        }
    """
    try:
        from database_v2.repositories import get_conn
        conn = get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE tiempo_deteccion_dias IS NOT NULL
                                        AND tiempo_deteccion_dias > 0) AS detectados,
                       AVG(tiempo_deteccion_dias) FILTER (WHERE tiempo_deteccion_dias > 0) AS avg_dias,
                       tipo_fallo
                FROM failure_events
                WHERE empresa_id = %s
                GROUP BY tipo_fallo
            """, (empresa_id,))
            rows = cur.fetchall()
            if not rows:
                return {"total_fallos": 0, "detectados_por_ia": 0,
                        "tasa_deteccion": 0.0, "anticipacion_media_dias": 0.0, "tipos_fallo": {}}

            total      = sum(r[0] for r in rows)
            detectados = sum(r[1] or 0 for r in rows)
            avg_dias   = sum((r[2] or 0) * (r[1] or 0) for r in rows) / max(detectados, 1)
            tipos      = {(r[3] or "unknown"): int(r[0]) for r in rows}

            return {
                "total_fallos":           total,
                "detectados_por_ia":      detectados,
                "tasa_deteccion":         round(detectados / max(total, 1), 3),
                "anticipacion_media_dias": round(avg_dias, 1),
                "tipos_fallo":            tipos,
            }
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("calcular_metricas_ia failed: %s", exc)
        return {}
