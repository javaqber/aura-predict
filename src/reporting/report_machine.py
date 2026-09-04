"""
AuraPredict — MachineReport  (Fase 6)
=======================================
Generates a self-contained HTML report for a single machine.

The report includes:
  - Current status and health score
  - Trend and maintenance recommendation
  - Health score evolution chart (inline SVG via data URIs)
  - Feature summary (RMS, Kurtosis)
  - Anomalies detected in the period
  - Fault diagnoses
  - Alert history
  - Analysis period metadata

No external dependencies beyond what is already installed.
The HTML is fully self-contained (no external CSS/JS) and can be:
  - Opened in a browser
  - Printed to PDF via browser print dialog
  - Attached to an email
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def generate_machine_report(
    maquina_id:   int,
    machine_name: str             = "",
    fecha_desde:  Optional[datetime] = None,
    fecha_hasta:  Optional[datetime] = None,
    empresa_id:   Optional[int]   = None,
    empresa_name: str             = "",
) -> bytes:
    """
    Generate a complete HTML machine report.

    Args:
        maquina_id:   Machine to report on.
        machine_name: Human-readable machine name (e.g. 'Torno CNC 1').
        fecha_desde:  Start of analysis period.
        fecha_hasta:  End of analysis period (defaults to now).
        empresa_id:   Company ID for isolation.
        empresa_name: Company name for the report header.

    Returns:
        UTF-8 encoded HTML bytes.
    """
    from reporting.exporter import _load_dataframe, _priority_label

    if fecha_hasta is None:
        fecha_hasta = datetime.now(timezone.utc)

    df_health  = _load_dataframe(maquina_id, "health_history", fecha_desde, fecha_hasta, empresa_id)
    df_read    = _load_dataframe(maquina_id, "readings",       fecha_desde, fecha_hasta, empresa_id)
    df_anom    = _load_dataframe(maquina_id, "anomalies",      fecha_desde, fecha_hasta, empresa_id)
    df_alerts  = _load_dataframe(maquina_id, "alerts",         fecha_desde, fecha_hasta, empresa_id)

    # ── Key metrics ────────────────────────────────────────────────────────────
    current_score = int(df_health["score"].iloc[0])  if not df_health.empty else None
    current_trend = df_health["trend"].iloc[0]        if not df_health.empty else None
    current_slope = df_health["slope"].iloc[0]        if not df_health.empty else None
    priority      = _priority_label(current_score)

    # Health color
    color = ("#2E7D32" if (current_score or 0) >= 75
             else "#F9A825" if (current_score or 0) >= 50
             else "#E65100" if (current_score or 0) >= 25
             else "#C62828")

    # Period string
    from_str = fecha_desde.strftime("%d/%m/%Y") if fecha_desde else "Inicio"
    to_str   = fecha_hasta.strftime("%d/%m/%Y")
    period   = f"{from_str} – {to_str}"

    # Maintenance recommendation
    recommendation = _recommendation_text(current_score, current_trend, current_slope)

    # ── Fault summary ──────────────────────────────────────────────────────────
    fault_counts: dict[str, int] = {}
    if not df_anom.empty and "diagnostico" in df_anom.columns:
        for diag in df_anom["diagnostico"].dropna():
            if diag:
                key = str(diag)[:60]
                fault_counts[key] = fault_counts.get(key, 0) + 1

    # ── Health chart (simple inline ASCII → replaced with numeric table) ───────
    health_table_rows = ""
    if not df_health.empty:
        for _, row in df_health.head(30).iterrows():
            ts    = str(row.get("timestamp", ""))[:16]
            sc    = row.get("score")
            trend = row.get("trend") or "—"
            slope = row.get("slope")
            bar   = _score_bar(sc)
            health_table_rows += (
                f"<tr><td>{ts}</td><td><b>{sc}</b></td>"
                f"<td>{bar}</td><td>{trend}</td>"
                f"<td>{round(slope, 2) if slope is not None else '—'}</td></tr>\n"
            )

    # ── Anomaly table ──────────────────────────────────────────────────────────
    anomaly_rows = ""
    if not df_anom.empty:
        for _, row in df_anom.head(20).iterrows():
            ts    = str(row.get("timestamp", ""))[:16]
            res   = row.get("resultado", "")
            risk  = row.get("nivel_riesgo", "")
            hs    = row.get("health_score", "—")
            score = round(row.get("anomaly_score", 0) or 0, 3)
            diag  = (str(row.get("diagnostico", "")) or "")[:80]
            risk_color = "#C62828" if risk == "CRÍTICO" else "#E65100" if risk == "Alto" else "#F9A825"
            anomaly_rows += (
                f"<tr><td>{ts}</td><td>{res}</td>"
                f"<td style='color:{risk_color};font-weight:bold'>{risk}</td>"
                f"<td>{hs}</td><td>{score}</td><td>{diag}</td></tr>\n"
            )

    # ── Alert table ────────────────────────────────────────────────────────────
    alert_rows = ""
    if not df_alerts.empty:
        for _, row in df_alerts.head(10).iterrows():
            ts      = str(row.get("sent_at", ""))[:16]
            tipo    = row.get("tipo_alerta", "")
            asunto  = (str(row.get("asunto", "")) or "")[:80]
            enviado = "✅" if row.get("enviado") else "⚠️"
            alert_rows += f"<tr><td>{ts}</td><td><b>{tipo}</b></td><td>{asunto}</td><td>{enviado}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AuraPredict — Informe {machine_name or f'Máquina {maquina_id}'}</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 13px; color: #222; margin: 0; padding: 20px 40px; }}
  h1 {{ color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 6px; }}
  h2 {{ color: #1976D2; margin-top: 28px; font-size: 15px; border-left: 4px solid #1976D2; padding-left: 8px; }}
  .header-meta {{ color: #555; font-size: 12px; margin-bottom: 24px; }}
  .kpi-grid {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
  .kpi {{ background: #f5f5f5; border-radius: 8px; padding: 14px 20px; min-width: 140px; }}
  .kpi .val {{ font-size: 28px; font-weight: bold; }}
  .kpi .lbl {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .rec {{ background: #E3F2FD; border-left: 4px solid #1976D2; padding: 10px 16px; border-radius: 4px; margin: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12px; }}
  th {{ background: #1565C0; color: white; text-align: left; padding: 6px 8px; }}
  td {{ padding: 5px 8px; border-bottom: 1px solid #eee; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  .bar {{ display: inline-block; height: 10px; border-radius: 3px; vertical-align: middle; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; color: white; }}
  .footer {{ margin-top: 40px; color: #aaa; font-size: 11px; border-top: 1px solid #eee; padding-top: 12px; }}
  @media print {{ body {{ padding: 10px 20px; }} h1 {{ font-size: 18px; }} }}
</style>
</head>
<body>

<h1>🔮 AuraPredict — Informe de Máquina</h1>
<div class="header-meta">
  <b>{machine_name or f'Máquina ID {maquina_id}'}</b>
  {f' | {empresa_name}' if empresa_name else ''}
  | Período analizado: <b>{period}</b>
  | Generado: {datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M UTC')}
</div>

<h2>Estado actual</h2>
<div class="kpi-grid">
  <div class="kpi">
    <div class="val" style="color:{color}">{current_score if current_score is not None else '—'}<small style="font-size:14px">/100</small></div>
    <div class="lbl">Health Score</div>
  </div>
  <div class="kpi">
    <div class="val">{priority}</div>
    <div class="lbl">Estado</div>
  </div>
  <div class="kpi">
    <div class="val">{current_trend or '—'}</div>
    <div class="lbl">Tendencia</div>
  </div>
  <div class="kpi">
    <div class="val">{round(current_slope, 2) if current_slope is not None else '—'}<small style="font-size:12px"> pt/día</small></div>
    <div class="lbl">Slope</div>
  </div>
  <div class="kpi">
    <div class="val">{len(df_read)}</div>
    <div class="lbl">Lecturas</div>
  </div>
  <div class="kpi">
    <div class="val" style="color:{'#C62828' if len(df_anom) > 0 else '#2E7D32'}">{len(df_anom)}</div>
    <div class="lbl">Anomalías</div>
  </div>
  <div class="kpi">
    <div class="val">{len(df_alerts)}</div>
    <div class="lbl">Alertas</div>
  </div>
</div>

<div class="rec">📋 <b>Recomendación de mantenimiento:</b> {recommendation}</div>

<h2>Evolución del Health Score (últimas 30 lecturas)</h2>
{'<p style="color:#888">Sin datos de health score en el período seleccionado.</p>' if not health_table_rows else f'''
<table>
  <thead><tr><th>Timestamp</th><th>Health Score</th><th>Barra</th><th>Tendencia</th><th>Slope (pt/día)</th></tr></thead>
  <tbody>{health_table_rows}</tbody>
</table>'''}

<h2>Anomalías detectadas (últimas 20)</h2>
{'<p style="color:#2E7D32">✅ Sin anomalías detectadas en el período analizado.</p>' if not anomaly_rows else f'''
<table>
  <thead><tr><th>Timestamp</th><th>Resultado</th><th>Riesgo</th><th>Health</th><th>Score</th><th>Diagnóstico</th></tr></thead>
  <tbody>{anomaly_rows}</tbody>
</table>'''}

<h2>Alertas enviadas (últimas 10)</h2>
{'<p style="color:#888">Sin alertas en el período seleccionado.</p>' if not alert_rows else f'''
<table>
  <thead><tr><th>Timestamp</th><th>Tipo</th><th>Asunto</th><th>Enviado</th></tr></thead>
  <tbody>{alert_rows}</tbody>
</table>'''}

<div class="footer">
  AuraPredict v2 — Sistema de Mantenimiento Predictivo Industrial |
  Este informe se genera automáticamente a partir de datos del sensor de vibración.
  Los diagnósticos son orientativos y deben ser confirmados por un técnico cualificado.
</div>
</body>
</html>"""

    return html.encode("utf-8")


def _recommendation_text(score, trend, slope) -> str:
    if score is None:
        return "Sin datos suficientes para generar una recomendación."
    if score >= 75:
        if trend == "degrading" and slope is not None and slope < -1.5:
            return "Estado saludable pero con tendencia decreciente. Monitorizar evolución en los próximos días."
        return "Estado saludable. Continuar con el plan de mantenimiento preventivo habitual."
    if score >= 50:
        if slope is not None and slope < -1.0:
            return "Degradación activa. Planificar inspección técnica en los próximos 7-14 días."
        return "Zona de vigilancia. Programar inspección en la próxima parada programada."
    if score >= 25:
        return "Prioridad alta. Realizar inspección técnica antes de continuar la producción."
    return "Estado crítico. Se recomienda detener la operación e inspeccionar inmediatamente."


def _score_bar(score) -> str:
    if score is None:
        return "—"
    color = ("#2E7D32" if score >= 75 else "#F9A825" if score >= 50
             else "#E65100" if score >= 25 else "#C62828")
    width = max(4, score)
    return (f'<div class="bar" style="width:{width}px;background:{color}"></div>'
            f'<span style="margin-left:4px;font-size:11px">{score}</span>')
