"""
AuraPredict — DataExporter  (Fase 6)
======================================
Exports machine data to CSV or Excel format for client reporting.

Supported export types:
  - health_history   : health score + trend + slope over time
  - readings         : full CNC readings with features and diagnoses
  - anomalies        : only anomalous readings (resultado != 'OK - Sano')
  - alerts           : alert_log entries
  - combined         : all of the above in one Excel workbook (multiple sheets)

All exports respect fecha_desde / fecha_hasta date range filters.
Multi-company isolation: empresa_id is always applied when provided.

Dependencies:
  pandas    — already in requirements.txt
  openpyxl  — already in requirements.txt (Excel write engine)
"""

from __future__ import annotations

import io
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def export_csv(
    maquina_id:   int,
    tipo:         str  = "readings",   # health_history | readings | anomalies | alerts
    fecha_desde:  Optional[datetime] = None,
    fecha_hasta:  Optional[datetime] = None,
    empresa_id:   Optional[int]      = None,
) -> bytes:
    """
    Export data to CSV bytes.

    Args:
        maquina_id:  Machine to export.
        tipo:        Dataset type (see module docstring).
        fecha_desde: Start of date range (inclusive). None = no lower bound.
        fecha_hasta: End of date range (inclusive). None = now.
        empresa_id:  If provided, verifies the machine belongs to this company.

    Returns:
        UTF-8 encoded CSV bytes ready to write to a file or HTTP response.
    """
    import pandas as pd
    df = _load_dataframe(maquina_id, tipo, fecha_desde, fecha_hasta, empresa_id)
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def export_excel(
    maquina_id:   int,
    fecha_desde:  Optional[datetime] = None,
    fecha_hasta:  Optional[datetime] = None,
    empresa_id:   Optional[int]      = None,
    include_all:  bool               = True,
) -> bytes:
    """
    Export all data types to a single Excel workbook (multiple sheets).

    Sheets:
      - Health Score       : health history
      - Lecturas           : CNC readings with features
      - Anomalías          : anomalous readings only
      - Alertas            : alert_log
      - Resumen            : summary statistics

    Returns:
        Excel file bytes (.xlsx).
    """
    import pandas as pd

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Health Score
        df_health = _load_dataframe(maquina_id, "health_history",
                                     fecha_desde, fecha_hasta, empresa_id)
        _write_sheet(writer, df_health, "Health Score")

        # Sheet 2: Lecturas completas
        df_read = _load_dataframe(maquina_id, "readings",
                                   fecha_desde, fecha_hasta, empresa_id)
        _write_sheet(writer, df_read, "Lecturas")

        # Sheet 3: Solo anomalías
        df_anom = _load_dataframe(maquina_id, "anomalies",
                                   fecha_desde, fecha_hasta, empresa_id)
        _write_sheet(writer, df_anom, "Anomalías")

        # Sheet 4: Alertas
        df_alerts = _load_dataframe(maquina_id, "alerts",
                                     fecha_desde, fecha_hasta, empresa_id)
        _write_sheet(writer, df_alerts, "Alertas")

        # Sheet 5: Resumen estadístico
        df_summary = _build_summary(maquina_id, df_health, df_read, df_anom, df_alerts)
        _write_sheet(writer, df_summary, "Resumen")

    return buf.getvalue()


def export_plant_excel(
    empresa_id: int,
    fecha_desde: Optional[datetime] = None,
    fecha_hasta: Optional[datetime] = None,
) -> bytes:
    """
    Export a plant-level summary Excel with one row per machine.

    Includes: machine name, current health score, trend, anomaly count,
    alert count, last reading timestamp.
    """
    import pandas as pd

    machines = _get_all_machines(empresa_id)
    if not machines:
        return _empty_excel("Sin máquinas registradas para esta empresa")

    rows = []
    for m in machines:
        mid  = m["maquina_id"]
        health = _load_dataframe(mid, "health_history", fecha_desde, fecha_hasta, empresa_id)
        reads  = _load_dataframe(mid, "readings",       fecha_desde, fecha_hasta, empresa_id)
        anoms  = _load_dataframe(mid, "anomalies",      fecha_desde, fecha_hasta, empresa_id)
        alerts = _load_dataframe(mid, "alerts",         fecha_desde, fecha_hasta, empresa_id)

        last_score = health["score"].iloc[0] if not health.empty else None
        last_trend = health["trend"].iloc[0] if not health.empty else None
        last_ts    = reads["timestamp"].iloc[0] if not reads.empty else None

        rows.append({
            "Máquina":           m.get("nombre", str(mid)),
            "Tipo":              m.get("tipo", "—"),
            "Health Score":      last_score,
            "Tendencia":         last_trend or "—",
            "Última lectura":    str(last_ts)[:19] if last_ts else "—",
            "Total lecturas":    len(reads),
            "Anomalías":         len(anoms),
            "Alertas":           len(alerts),
            "Prioridad":         _priority_label(last_score),
        })

    # Sort by priority (worst first)
    priority_order = {"🔴 CRÍTICO": 0, "🟠 Alto riesgo": 1,
                       "🟡 Vigilar": 2, "🟢 Sano": 3, "—": 4}
    rows.sort(key=lambda r: priority_order.get(r["Prioridad"], 99))

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Resumen de Planta",
                                     index=False)
    return buf.getvalue()


# ─── INTERNAL DATA LOADERS ────────────────────────────────────────────────────

def _load_dataframe(
    maquina_id:  int,
    tipo:        str,
    fecha_desde: Optional[datetime],
    fecha_hasta: Optional[datetime],
    empresa_id:  Optional[int],
):
    import pandas as pd

    try:
        if tipo == "health_history":
            return _df_health(maquina_id, fecha_desde, fecha_hasta)
        elif tipo == "readings":
            return _df_readings(maquina_id, fecha_desde, fecha_hasta)
        elif tipo == "anomalies":
            return _df_anomalies(maquina_id, fecha_desde, fecha_hasta)
        elif tipo == "alerts":
            return _df_alerts(maquina_id, fecha_desde, fecha_hasta)
        else:
            logger.warning("Unknown export type: %s", tipo)
            return pd.DataFrame()
    except Exception as exc:
        logger.error("Export failed for maquina_id=%d tipo=%s: %s", maquina_id, tipo, exc)
        return pd.DataFrame()


def _df_health(maquina_id: int, fecha_desde, fecha_hasta):
    import pandas as pd
    from database_v2.repositories import get_conn
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT timestamp, score, trend, slope
            FROM health_scores
            WHERE maquina_id = %s
              AND (%s IS NULL OR timestamp >= %s)
              AND (%s IS NULL OR timestamp <= %s)
            ORDER BY timestamp DESC
        """, (maquina_id, fecha_desde, fecha_desde, fecha_hasta, fecha_hasta))
        cols = ["timestamp", "score", "trend", "slope"]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        cur.close()
        conn.close()


def _df_readings(maquina_id: int, fecha_desde, fecha_hasta):
    import pandas as pd
    from database_v2.repositories import get_conn
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT timestamp, resultado, nivel_riesgo, health_score, anomaly_score,
                   rms_x, rms_y, rms_z,
                   kurtosis_x, kurtosis_y, kurtosis_z,
                   crest_factor_x, crest_factor_y, crest_factor_z,
                   peak_to_peak_x, peak_to_peak_y, peak_to_peak_z,
                   dominant_freq_hz, band_low_energy, band_mid_energy, band_high_energy,
                   diagnostico, signal_quality_score, algorithm
            FROM lecturas_cnc_v2
            WHERE maquina_id = %s
              AND (%s IS NULL OR timestamp >= %s)
              AND (%s IS NULL OR timestamp <= %s)
            ORDER BY timestamp DESC
        """, (maquina_id, fecha_desde, fecha_desde, fecha_hasta, fecha_hasta))
        cols = ["timestamp","resultado","nivel_riesgo","health_score","anomaly_score",
                "rms_x","rms_y","rms_z","kurtosis_x","kurtosis_y","kurtosis_z",
                "crest_factor_x","crest_factor_y","crest_factor_z",
                "peak_to_peak_x","peak_to_peak_y","peak_to_peak_z",
                "dominant_freq_hz","band_low_energy","band_mid_energy","band_high_energy",
                "diagnostico","signal_quality_score","algorithm"]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        cur.close()
        conn.close()


def _df_anomalies(maquina_id: int, fecha_desde, fecha_hasta):
    import pandas as pd
    df = _df_readings(maquina_id, fecha_desde, fecha_hasta)
    if df.empty:
        return df
    return df[~df["resultado"].str.startswith("OK", na=False)].copy()


def _df_alerts(maquina_id: int, fecha_desde, fecha_hasta):
    import pandas as pd
    from database_v2.repositories import get_conn
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT sent_at, tipo_alerta, asunto, enviado, error_msg
            FROM alert_log
            WHERE maquina_id = %s
              AND (%s IS NULL OR sent_at >= %s)
              AND (%s IS NULL OR sent_at <= %s)
            ORDER BY sent_at DESC
        """, (maquina_id, fecha_desde, fecha_desde, fecha_hasta, fecha_hasta))
        cols = ["sent_at","tipo_alerta","asunto","enviado","error_msg"]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        cur.close()
        conn.close()


def _get_all_machines(empresa_id: int) -> list[dict]:
    try:
        from database_v2.repositories import obtener_todas_maquinas_con_health
        return obtener_todas_maquinas_con_health(empresa_id=empresa_id)
    except Exception:
        return []


def _build_summary(maquina_id, df_health, df_read, df_anom, df_alerts):
    import pandas as pd
    rows = [
        ("Máquina ID",           maquina_id),
        ("Total lecturas",       len(df_read)),
        ("Total anomalías",      len(df_anom)),
        ("Total alertas",        len(df_alerts)),
        ("Health Score actual",  df_health["score"].iloc[0]  if not df_health.empty else "—"),
        ("Tendencia actual",     df_health["trend"].iloc[0]  if not df_health.empty else "—"),
        ("Health Score mínimo",  df_health["score"].min()    if not df_health.empty else "—"),
        ("Health Score máximo",  df_health["score"].max()    if not df_health.empty else "—"),
        ("Health Score medio",   round(df_health["score"].mean(), 1) if not df_health.empty else "—"),
        ("RMS X medio",          round(df_read["rms_x"].mean(), 4)   if not df_read.empty and "rms_x" in df_read else "—"),
        ("Kurtosis X medio",     round(df_read["kurtosis_x"].mean(), 3) if not df_read.empty and "kurtosis_x" in df_read else "—"),
        ("Primera lectura",      str(df_read["timestamp"].iloc[-1])[:19] if not df_read.empty else "—"),
        ("Última lectura",       str(df_read["timestamp"].iloc[0])[:19]  if not df_read.empty else "—"),
        ("Exportado",            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
    ]
    return pd.DataFrame(rows, columns=["Indicador", "Valor"])


def _write_sheet(writer, df, sheet_name: str) -> None:
    import pandas as pd
    if df.empty:
        pd.DataFrame([["Sin datos en este período"]]).to_excel(
            writer, sheet_name=sheet_name, index=False, header=False
        )
    else:
        # Excel does not support timezone-aware datetimes — strip tz before writing
        df_out = df.copy()
        for col in df_out.select_dtypes(include=["datetimetz"]).columns:
            df_out[col] = df_out[col].dt.tz_localize(None)
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)
        # Auto-size columns
        worksheet = writer.sheets[sheet_name]
        for col_cells in worksheet.columns:
            max_len = max((len(str(c.value or "")) for c in col_cells), default=10)
            worksheet.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)


def _empty_excel(message: str) -> bytes:
    import pandas as pd
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([[message]]).to_excel(writer, index=False, header=False)
    return buf.getvalue()


def _priority_label(score) -> str:
    if score is None: return "—"
    if score >= 75:   return "🟢 Sano"
    if score >= 50:   return "🟡 Vigilar"
    if score >= 25:   return "🟠 Alto riesgo"
    return "🔴 CRÍTICO"
