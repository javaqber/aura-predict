from dotenv import load_dotenv
from datetime import datetime
import struct
import math
import requests
import schedule
import time
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# ─── CONFIGURACION ────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_EMAIL = os.getenv("SCHEDULER_EMAIL", "")
API_PASS = os.getenv("SCHEDULER_PASSWORD", "")
MAQUINA = os.getenv("MAQUINA_NOMBRE", "Torno_CNC_1")

# Horario de lecturas (formato 24h)
HORA_INICIO = 6     # 06:00 — inicio de jornada
HORA_FIN = 22    # 18:00 — fin de jornada

# Intervalo normal entre lecturas (en horas)
INTERVALO_NORMAL_H = 2

# Intervalo en caso de anomalía (en minutos)
INTERVALO_ANOMALIA_MIN = 5

# Número de muestras por ventana de análisis
MUESTRAS = 200
FREQ_HZ = 100

# Modo simulado (True en PC, False en Raspberry Pi con sensor real)
MODO_SIMULADO = True

# ─── ESTADO INTERNO ───────────────────────────────────────────────────────────
_token = None
_en_modo_anomalia = False


# ─── AUTENTICACION ────────────────────────────────────────────────────────────

def obtener_token():
    """Hace login en la API y guarda el token JWT."""
    global _token
    try:
        r = requests.post(
            f"{API_URL}/login",
            json={"email": API_EMAIL, "password": API_PASS},
            timeout=10
        )
        if r.status_code == 200:
            _token = r.json()["access_token"]
            print(f"  ✅ Login correcto como {API_EMAIL}")
            return True
        else:
            print(
                f"  ❌ Error de login: {r.json().get('detail', 'desconocido')}")
            return False
    except Exception as e:
        print(f"  ❌ No se pudo conectar con la API: {e}")
        return False


# ─── LECTURA DEL SENSOR ───────────────────────────────────────────────────────

def leer_sensor_simulado():
    return {
        "maquina":      MAQUINA,
        "RMS":          0.45,   # ← alto (normal ~0.07)
        "Peak_to_Peak": 2.5,
        "Kurtosis":     5.2,    # ← alto (normal ~0.6)
        "Skewness":     0.08
    }


def leer_sensor_real():
    """Lee el sensor ADXL345 via I2C (solo Raspberry Pi)."""
    try:
        import smbus2  # Libreria de Raspberry Pi
        import numpy as np
        from scipy import stats

        ADXL_ADDR = 0x53
        REG_POWER_CTL = 0x2D
        REG_DATA_FORMAT = 0x31
        REG_DATAX0 = 0x32
        SCALE_FACTOR = 0.0039

        bus = smbus2.SMBus(1)
        bus.write_byte_data(ADXL_ADDR, REG_DATA_FORMAT, 0x01)
        time.sleep(0.01)
        bus.write_byte_data(ADXL_ADDR, REG_POWER_CTL, 0x08)
        time.sleep(0.01)

        muestras = []
        intervalo = 1.0 / FREQ_HZ
        for _ in range(MUESTRAS):
            datos = bus.read_i2c_block_data(ADXL_ADDR, REG_DATAX0, 6)
            x = struct.unpack('<h', bytes([datos[0], datos[1]]))[
                0] * SCALE_FACTOR
            y = struct.unpack('<h', bytes([datos[2], datos[3]]))[
                0] * SCALE_FACTOR
            z = struct.unpack('<h', bytes([datos[4], datos[5]]))[
                0] * SCALE_FACTOR
            muestras.append(math.sqrt(x**2 + y**2 + z**2))
            time.sleep(intervalo)

        bus.close()
        ventana = np.array(muestras)

        return {
            "maquina":      MAQUINA,
            "RMS":          round(float(np.sqrt(np.mean(ventana**2))), 4),
            "Peak_to_Peak": round(float(np.max(ventana) - np.min(ventana)), 4),
            "Kurtosis":     round(float(stats.kurtosis(ventana)), 4),
            "Skewness":     round(float(stats.skew(ventana)), 4)
        }
    except Exception as e:
        print(f"  ❌ Error leyendo sensor: {e}")
        return None


# ─── ENVIO A LA API ───────────────────────────────────────────────────────────

def enviar_lectura(datos):
    """Envía los datos a la API con autenticación JWT."""
    global _token, _en_modo_anomalia

    if not _token:
        if not obtener_token():
            return

    try:
        r = requests.post(
            f"{API_URL}/predict/bearing",
            json=datos,
            headers={"Authorization": f"Bearer {_token}"},
            timeout=10
        )

        # Token expirado — hacer login de nuevo
        if r.status_code == 401:
            print("  🔄 Token expirado, renovando...")
            if obtener_token():
                enviar_lectura(datos)
            return

        if r.status_code == 200:
            resultado = r.json()
            estado = resultado.get("estado_maquina", "?")
            diagnostico = resultado.get("diagnostico", {})
            tipo_fallo = diagnostico.get("tipo_fallo", "—")

            es_ok = "NOK" not in estado  # ← fix: comprobar NOK primero

            simbolo = "✅" if es_ok else "🚨"
            print(f"  {simbolo} {estado}")

            if not es_ok:
                print(f"     → {tipo_fallo}")
                print(f"     → {diagnostico.get('accion_recomendada', '—')}")

            # Activar modo anomalía si hay fallo
            if not es_ok and not _en_modo_anomalia:
                _en_modo_anomalia = True
                programar_lecturas()
                print(
                    f"  ⚡ Modo anomalía activado: lectura cada {INTERVALO_ANOMALIA_MIN} min")

            elif es_ok and _en_modo_anomalia:
                _en_modo_anomalia = False
                programar_lecturas()
                print(
                    f"  ✅ Modo normal restaurado: lectura cada {INTERVALO_NORMAL_H}h")
        else:
            print(f"  ⚠️  Error API: {r.status_code}")

    except Exception as e:
        print(f"  ❌ Error de conexión: {e}")


# ─── TAREA PRINCIPAL ──────────────────────────────────────────────────────────

def tarea_lectura():
    """Tarea que se ejecuta según el scheduler."""
    ahora = datetime.now()
    hora = ahora.hour

    # Respetar horario de jornada laboral
    if not (HORA_INICIO <= hora < HORA_FIN):
        print(f"[{ahora.strftime('%H:%M')}] Fuera de jornada — lectura omitida")
        return

    timestamp = ahora.strftime("%H:%M:%S")
    print(f"\n[{timestamp}] Tomando lectura de {MAQUINA}...")

    datos = leer_sensor_simulado() if MODO_SIMULADO else leer_sensor_real()
    if datos:
        print(f"  RMS={datos['RMS']}  K={datos['Kurtosis']}  "
              f"P2P={datos['Peak_to_Peak']}  Skew={datos['Skewness']}", end="  →  ")
        enviar_lectura(datos)


# ─── PROGRAMACION DE LECTURAS ─────────────────────────────────────────────────

def programar_lecturas():
    """Reprograma las lecturas según el estado actual."""
    schedule.clear()

    if _en_modo_anomalia:
        schedule.every(INTERVALO_ANOMALIA_MIN).minutes.do(tarea_lectura)
        print(
            f"  📅 Lecturas programadas cada {INTERVALO_ANOMALIA_MIN} minutos (modo anomalía)")
    else:
        schedule.every(INTERVALO_NORMAL_H).hours.do(tarea_lectura)
        print(
            f"  📅 Lecturas programadas cada {INTERVALO_NORMAL_H} horas (modo normal)")


# ─── ARRANQUE ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🔮 AuraPredict — Scheduler de Adquisición")
    print("=" * 60)
    print(f"  Máquina   : {MAQUINA}")
    print(f"  Jornada   : {HORA_INICIO}:00 — {HORA_FIN}:00")
    print(f"  Intervalo : cada {INTERVALO_NORMAL_H}h (normal) / "
          f"{INTERVALO_ANOMALIA_MIN}min (anomalía)")
    print(f"  Modo      : {'SIMULADO' if MODO_SIMULADO else 'SENSOR REAL'}")
    print(f"  API       : {API_URL}")
    print("=" * 60)

    # Login inicial
    if not obtener_token():
        print("❌ No se pudo autenticar. Verifica EMAIL y PASSWORD en .env")
        return

    # Primera lectura inmediata al arrancar
    tarea_lectura()

    # Programar lecturas periódicas
    programar_lecturas()

    print("\n  Scheduler activo. Pulsa Ctrl+C para detener.\n")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Revisar cada 30 segundos
    except KeyboardInterrupt:
        print("\n\n  Scheduler detenido.")


if __name__ == "__main__":
    main()
