import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ─── CONFIGURACION ────────────────────────────────────────────────────────────
EMAIL_ORIGEN = os.getenv("EMAIL_ORIGEN", "")
EMAIL_CONTRASENA = os.getenv("EMAIL_CONTRASENA", "")
EMAIL_DESTINO = os.getenv("EMAIL_DESTINO", "")
EMAIL_ACTIVO = os.getenv("EMAIL_ACTIVO", "false").lower() == "true"

# Control de frecuencia: evita mandar el mismo email más de una vez
# por máquina en un intervalo de tiempo (en segundos)
INTERVALO_MINIMO = 3600   # 1 hora entre alertas de la misma máquina
_ultimo_envio = {}        # {nombre_maquina: timestamp_ultimo_envio}


def _puede_enviar(maquina):
    """Comprueba si ha pasado suficiente tiempo desde la última alerta."""
    ahora = datetime.now().timestamp()
    ultimo = _ultimo_envio.get(maquina, 0)
    return (ahora - ultimo) >= INTERVALO_MINIMO


def _construir_email(maquina, estado, riesgo, diagnostico, valores):
    """Construye el email HTML con toda la información del diagnóstico."""

    COLORES_URGENCIA = {
        "rojo":    "#C62828",
        "naranja": "#E65100",
        "amarillo": "#F9A825",
        "verde":   "#2E7D32"
    }
    color = COLORES_URGENCIA.get(diagnostico.get(
        "nivel_urgencia", "naranja"), "#666")
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    cuerpo_html = f"""
    <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px;">

        <div style="background:#0B3D91; padding:20px; border-radius:8px 8px 0 0;">
            <h1 style="color:white; margin:0; font-size:1.5rem;">
                ⚠️ AuraPredict — Alerta de Mantenimiento
            </h1>
        </div>

        <div style="border:1px solid #ddd; border-top:none; padding:20px; border-radius:0 0 8px 8px;">

            <div style="background:{color}15; border-left:4px solid {color};
                        padding:12px 16px; border-radius:4px; margin-bottom:20px;">
                <h2 style="color:{color}; margin:0 0 8px 0;">
                    {diagnostico.get('tipo_fallo', estado)}
                </h2>
                <p style="margin:0; color:#333;">
                    <strong>Máquina:</strong> {maquina} &nbsp;|&nbsp;
                    <strong>Detectado:</strong> {timestamp}
                </p>
            </div>

            <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
                <tr style="background:#F5F5F5;">
                    <td style="padding:10px; font-weight:bold; width:40%;">Componente afectado</td>
                    <td style="padding:10px;">{diagnostico.get('componente_afectado', '—')}</td>
                </tr>
                <tr>
                    <td style="padding:10px; font-weight:bold;">Descripción</td>
                    <td style="padding:10px;">{diagnostico.get('descripcion', '—')}</td>
                </tr>
                <tr style="background:#F5F5F5;">
                    <td style="padding:10px; font-weight:bold;">Consecuencias</td>
                    <td style="padding:10px;">{diagnostico.get('consecuencias', '—')}</td>
                </tr>
                <tr>
                    <td style="padding:10px; font-weight:bold;">Acción recomendada</td>
                    <td style="padding:10px; color:{color}; font-weight:bold;">
                        {diagnostico.get('accion_recomendada', '—')}
                    </td>
                </tr>
                <tr style="background:#F5F5F5;">
                    <td style="padding:10px; font-weight:bold;">Pieza / Referencia</td>
                    <td style="padding:10px;">{diagnostico.get('pieza_referencia', '—')}</td>
                </tr>
                <tr>
                    <td style="padding:10px; font-weight:bold;">Ventana de actuación</td>
                    <td style="padding:10px; color:{color}; font-weight:bold;">
                        {diagnostico.get('ventana_actuacion', '—')}
                    </td>
                </tr>
            </table>

            <div style="background:#F5F5F5; padding:12px 16px; border-radius:4px; margin-bottom:20px;">
                <h4 style="margin:0 0 8px 0;">📊 Valores del sensor en el momento del fallo</h4>
                <p style="margin:4px 0;">
                    RMS: <strong>{valores.get('RMS', '—')}</strong> &nbsp;|&nbsp;
                    Peak-to-Peak: <strong>{valores.get('Peak_to_Peak', '—')}</strong> &nbsp;|&nbsp;
                    Kurtosis: <strong>{valores.get('Kurtosis', '—')}</strong> &nbsp;|&nbsp;
                    Skewness: <strong>{valores.get('Skewness', '—')}</strong>
                </p>
            </div>

            <p style="color:#888; font-size:0.85em; border-top:1px solid #eee; padding-top:12px;">
                Este mensaje ha sido generado automáticamente por AuraPredict.<br>
                Confianza del diagnóstico: {diagnostico.get('confianza', '—')}
            </p>
        </div>

    </body></html>
    """
    return cuerpo_html


def enviar_alerta(maquina, estado, riesgo, diagnostico, valores):
    # Leer credenciales aquí, no al importar el módulo
    email_activo = os.getenv("EMAIL_ACTIVO", "false").lower() == "true"
    email_origen = os.getenv("EMAIL_ORIGEN", "")
    email_contrasena = os.getenv("EMAIL_CONTRASENA", "")
    email_destino = os.getenv("EMAIL_DESTINO", "")

    if not email_activo:
        return

    if not email_origen or not email_contrasena or not email_destino:
        print("⚠️  Alertas email: faltan credenciales en el archivo .env")
        return

    if diagnostico.get("nivel_urgencia") == "verde":
        return

    if not _puede_enviar(maquina):
        print(
            f"  📧 Alerta omitida para {maquina} (ya se envió hace menos de 1h)")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = (
            f"🚨 AuraPredict — {diagnostico.get('tipo_fallo', 'Anomalía')} "
            f"en {maquina}"
        )
        msg["From"] = email_origen
        msg["To"] = email_destino

        cuerpo_html = _construir_email(
            maquina, estado, riesgo, diagnostico, valores)
        msg.attach(MIMEText(cuerpo_html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(email_origen, email_contrasena)
            servidor.sendmail(email_origen, email_destino, msg.as_string())

        _ultimo_envio[maquina] = datetime.now().timestamp()
        print(f"  📧 Alerta enviada a {email_destino} para máquina {maquina}")

    except Exception as e:
        print(f"  ❌ Error al enviar email: {e}")
