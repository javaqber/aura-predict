"""
AuraPredict — PlantReport  (Fase 6)
=====================================
Generates a plant-level HTML status report summarising all machines
belonging to a company.

Sorted by priority (worst health first) so the maintenance team sees
the most urgent machines at the top.
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


def generate_plant_report(
    empresa_id:   int,
    empresa_name: str             = "",
    fecha_desde:  Optional[datetime] = None,
    fecha_hasta:  Optional[datetime] = None,
) -> bytes:
    """
    Generate a plant-level HTML status report.

    Returns:
        UTF-8 encoded HTML bytes.
    """
    from reporting.exporter import _load_dataframe, _get_all_machines, _priority_label

    if fecha_hasta is None:
        fecha_hasta = datetime.now(timezone.utc)

    from_str = fecha_desde.strftime("%d/%m/%Y") if fecha_desde else "Inicio"
    to_str   = fecha_hasta.strftime("%d/%m/%Y")
    period   = f"{from_str} – {to_str}"

    machines = _get_all_machines(empresa_id)
    generated_at = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    if not machines:
        return f"""<!DOCTYPE html><html lang="es"><body>
        <h1>AuraPredict — Informe de Planta</h1>
        <p>Sin máquinas registradas para {empresa_name or f'empresa {empresa_id}'}.</p>
        </body></html>""".encode("utf-8")

    # ── Build machine rows ─────────────────────────────────────────────────────
    machine_data = []
    for m in machines:
        mid   = m["maquina_id"]
        name  = m.get("nombre", str(mid))
        tipo  = m.get("tipo", "—")

        df_h  = _load_dataframe(mid, "health_history", fecha_desde, fecha_hasta, empresa_id)
        df_r  = _load_dataframe(mid, "readings",       fecha_desde, fecha_hasta, empresa_id)
        df_a  = _load_dataframe(mid, "anomalies",      fecha_desde, fecha_hasta, empresa_id)
        df_al = _load_dataframe(mid, "alerts",         fecha_desde, fecha_hasta, empresa_id)

        score   = int(df_h["score"].iloc[0])  if not df_h.empty else None
        trend   = df_h["trend"].iloc[0]       if not df_h.empty else None
        slope   = df_h["slope"].iloc[0]       if not df_h.empty else None
        last_ts = str(df_r["timestamp"].iloc[0])[:16] if not df_r.empty else "—"

        priority = _priority_label(score)
        color    = _score_color(score)

        machine_data.append({
            "id":        mid,
            "name":      name,
            "tipo":      tipo,
            "score":     score,
            "trend":     trend or "—",
            "slope":     slope,
            "priority":  priority,
            "color":     color,
            "n_reads":   len(df_r),
            "n_anom":    len(df_a),
            "n_alerts":  len(df_al),
            "last_ts":   last_ts,
        })

    # Sort: worst first
    priority_order = {"🔴 CRÍTICO": 0, "🟠 Alto riesgo": 1, "🟡 Vigilar": 2, "🟢 Sano": 3, "—": 4}
    machine_data.sort(key=lambda x: (priority_order.get(x["priority"], 99),
                                      -(x["score"] or 100)))

    # ── Summary KPIs ───────────────────────────────────────────────────────────
    total      = len(machine_data)
    critical   = sum(1 for m in machine_data if (m["score"] or 100) < 25)
    high_risk  = sum(1 for m in machine_data if 25 <= (m["score"] or 100) < 50)
    watch      = sum(1 for m in machine_data if 50 <= (m["score"] or 100) < 75)
    healthy    = sum(1 for m in machine_data if (m["score"] or 0) >= 75)
    total_anom = sum(m["n_anom"] for m in machine_data)

    # ── Machine table rows ─────────────────────────────────────────────────────
    machine_rows = ""
    for m in machine_data:
        sc    = m["score"]
        bar   = _bar_html(sc, m["color"])
        slope_str = (f"{m['slope']:+.2f}" if m["slope"] is not None else "—")
        anom_color = "#C62828" if m["n_anom"] > 0 else "#2E7D32"
        machine_rows += (
            f"<tr>"
            f"<td><b>{m['name']}</b><br><small style='color:#888'>{m['tipo']}</small></td>"
            f"<td style='color:{m['color']};font-weight:bold'>{sc if sc is not None else '—'}</td>"
            f"<td>{bar}</td>"
            f"<td>{m['priority']}</td>"
            f"<td>{m['trend']}</td>"
            f"<td>{slope_str}</td>"
            f"<td style='color:{anom_color}'>{m['n_anom']}</td>"
            f"<td>{m['n_alerts']}</td>"
            f"<td>{m['n_reads']}</td>"
            f"<td>{m['last_ts']}</td>"
            f"</tr>\n"
        )

    # ── Priority list (critical machines) ─────────────────────────────────────
    priority_list = ""
    urgent = [m for m in machine_data if (m["score"] or 100) < 50]
    if urgent:
        for i, m in enumerate(urgent[:5], 1):
            rec = _short_recommendation(m["score"])
            priority_list += (
                f"<li><b>{m['name']}</b> — Health Score: "
                f"<span style='color:{m['color']}'>{m['score']}</span> "
                f"({m['priority']}) — {rec}</li>\n"
            )
    else:
        priority_list = "<li style='color:#2E7D32'>✅ Todas las máquinas en estado saludable.</li>"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>AuraPredict — Informe de Planta</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 13px; color: #222; margin: 0; padding: 20px 40px; }}
  h1 {{ color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 6px; }}
  h2 {{ color: #1976D2; margin-top: 28px; font-size: 15px; border-left: 4px solid #1976D2; padding-left: 8px; }}
  .header-meta {{ color: #555; font-size: 12px; margin-bottom: 24px; }}
  .kpi-grid {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
  .kpi {{ background: #f5f5f5; border-radius: 8px; padding: 12px 18px; min-width: 120px; text-align: center; }}
  .kpi .val {{ font-size: 26px; font-weight: bold; }}
  .kpi .lbl {{ font-size: 11px; color: #666; text-transform: uppercase; }}
  .priority-box {{ background: #FFF3E0; border-left: 4px solid #FF6F00; padding: 10px 16px; border-radius: 4px; margin: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 12px; }}
  th {{ background: #1565C0; color: white; text-align: left; padding: 6px 8px; }}
  td {{ padding: 5px 8px; border-bottom: 1px solid #eee; vertical-align: middle; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  .bar {{ display: inline-block; height: 10px; border-radius: 3px; vertical-align: middle; }}
  .footer {{ margin-top: 40px; color: #aaa; font-size: 11px; border-top: 1px solid #eee; padding-top: 12px; }}
  @media print {{ body {{ padding: 10px 20px; }} }}
</style>
</head>
<body>

<h1>🏭 AuraPredict — Informe de Estado de Planta</h1>
<div class="header-meta">
  <b>{empresa_name or f'Empresa ID {empresa_id}'}</b>
  | Período: <b>{period}</b>
  | Generado: {generated_at}
</div>

<h2>Resumen ejecutivo</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="val">{total}</div><div class="lbl">Máquinas</div></div>
  <div class="kpi"><div class="val" style="color:#C62828">{critical}</div><div class="lbl">Críticas</div></div>
  <div class="kpi"><div class="val" style="color:#E65100">{high_risk}</div><div class="lbl">Alto riesgo</div></div>
  <div class="kpi"><div class="val" style="color:#F9A825">{watch}</div><div class="lbl">Vigilar</div></div>
  <div class="kpi"><div class="val" style="color:#2E7D32">{healthy}</div><div class="lbl">Sanas</div></div>
  <div class="kpi"><div class="val">{total_anom}</div><div class="lbl">Anomalías totales</div></div>
</div>

<h2>Prioridades de mantenimiento</h2>
<div class="priority-box">
  <b>Máquinas que requieren atención inmediata o próxima:</b>
  <ol style="margin: 8px 0 0 0; padding-left: 20px;">
    {priority_list}
  </ol>
</div>

<h2>Estado de todas las máquinas</h2>
<table>
  <thead>
    <tr>
      <th>Máquina</th><th>Health</th><th>Barra</th><th>Estado</th>
      <th>Tendencia</th><th>Slope</th><th>Anomalías</th>
      <th>Alertas</th><th>Lecturas</th><th>Última lectura</th>
    </tr>
  </thead>
  <tbody>
    {machine_rows}
  </tbody>
</table>

<div class="footer">
  AuraPredict v2 — Sistema de Mantenimiento Predictivo Industrial |
  Los datos se obtienen del sensor de vibración. Los diagnósticos son orientativos.
</div>
</body>
</html>"""

    return html.encode("utf-8")


def _score_color(score) -> str:
    if score is None: return "#888888"
    if score >= 75:   return "#2E7D32"
    if score >= 50:   return "#F9A825"
    if score >= 25:   return "#E65100"
    return "#C62828"


def _bar_html(score, color) -> str:
    if score is None:
        return "—"
    w = max(4, score)
    return (f'<div class="bar" style="width:{w}px;background:{color}"></div>'
            f' <span style="font-size:11px">{score}</span>')


def _short_recommendation(score) -> str:
    if score is None: return "Sin datos"
    if score < 25:    return "Detener y revisar"
    if score < 50:    return "Inspección urgente"
    return "Planificar revisión próxima"
