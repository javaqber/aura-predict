import time
import math
import requests
import struct
import numpy as np
from scipy import stats
from datetime import datetime

# smbus2 solo existe en Linux (Raspberry Pi)
try:
    import smbus2
    HARDWARE_DISPONIBLE = True
except ImportError:
    HARDWARE_DISPONIBLE = False
    print("⚠️  smbus2 no disponible — este script requiere Raspberry Pi.")

# ─── CONFIGURACION ────────────────────────────────────────────────────────────
API_URL = "http://TU_IP_DEL_SERVIDOR:8000/predict/bearing"
MAQUINA = "Torno Mecaniperz 1"   # Cambiar por el nombre registrado en el dashboard
INTERVALO = 10               # Segundos entre cada lectura al servidor

# ADXL345 — configuracion I2C
I2C_BUS = 1       # Bus I2C de Raspberry Pi (siempre es 1 en RPi Zero 2W)
ADXL_ADDR = 0x53    # Direccion I2C por defecto (SDO conectado a GND)
MUESTRAS = 200     # Numero de muestras por ventana de analisis
FREQ_HZ = 100     # Frecuencia de muestreo (muestras por segundo)

# Registros del ADXL345
REG_POWER_CTL = 0x2D
REG_DATA_FORMAT = 0x31
REG_DATAX0 = 0x32
SCALE_FACTOR = 0.0039  # g por LSB en rango ±2g (resolucion por defecto)

# ─── INICIALIZACION DEL SENSOR ────────────────────────────────────────────────


def inicializar_sensor(bus):
    """Configura el ADXL345 en modo medicion."""
    # Rango ±4g, resolucion completa
    bus.write_byte_data(ADXL_ADDR, REG_DATA_FORMAT, 0x01)
    time.sleep(0.01)
    # Activar modo medicion
    bus.write_byte_data(ADXL_ADDR, REG_POWER_CTL, 0x08)
    time.sleep(0.01)
    print("✅ ADXL345 inicializado correctamente.")


def leer_aceleracion(bus):
    """Lee los valores raw de aceleracion en los tres ejes."""
    datos = bus.read_i2c_block_data(ADXL_ADDR, REG_DATAX0, 6)
    x = struct.unpack('<h', bytes([datos[0], datos[1]]))[0] * SCALE_FACTOR
    y = struct.unpack('<h', bytes([datos[2], datos[3]]))[0] * SCALE_FACTOR
    z = struct.unpack('<h', bytes([datos[4], datos[5]]))[0] * SCALE_FACTOR
    # Modulo total de la aceleracion (combina los tres ejes)
    return math.sqrt(x**2 + y**2 + z**2)


# ─── PROCESAMIENTO DE SEÑAL ───────────────────────────────────────────────────

def capturar_ventana(bus):
    """
    Toma MUESTRAS lecturas del sensor a FREQ_HZ Hz
    y devuelve el array de aceleraciones.
    """
    muestras = []
    intervalo = 1.0 / FREQ_HZ

    for _ in range(MUESTRAS):
        muestras.append(leer_aceleracion(bus))
        time.sleep(intervalo)

    return np.array(muestras)


def calcular_features(ventana):
    """
    Calcula las 4 features que usa el modelo de Isolation Forest:
    RMS, Peak-to-Peak, Kurtosis y Skewness.
    """
    rms = float(np.sqrt(np.mean(ventana**2)))
    peak_to_peak = float(np.max(ventana) - np.min(ventana))
    kurtosis = float(stats.kurtosis(ventana))
    skewness = float(stats.skew(ventana))

    return {
        "maquina":      MAQUINA,
        "RMS":          round(rms, 4),
        "Peak_to_Peak": round(peak_to_peak, 4),
        "Kurtosis":     round(kurtosis, 4),
        "Skewness":     round(skewness, 4)
    }


# ─── ENVIO A LA API ───────────────────────────────────────────────────────────

def enviar_a_api(datos):
    try:
        respuesta = requests.post(API_URL, json=datos, timeout=5)
        if respuesta.status_code == 200:
            return respuesta.json()
        else:
            print(f"  ⚠️  Error HTTP {respuesta.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"  ❌  Sin conexion con la API ({API_URL})")
        return None
    except Exception as e:
        print(f"  ❌  Error: {e}")
        return None


# ─── BUCLE PRINCIPAL ──────────────────────────────────────────────────────────

def main():
    if not HARDWARE_DISPONIBLE:
        print("❌ Este script solo puede ejecutarse en la Raspberry Pi.")
        print("   En tu PC usa simulador.py para probar el pipeline.")
        return
    print("=" * 60)
    print("  🔮 AuraPredict — Lectura de Sensor ADXL345")
    print("=" * 60)
    print(f"  Maquina   : {MAQUINA}")
    print(
        f"  Muestras  : {MUESTRAS} a {FREQ_HZ}Hz ({MUESTRAS/FREQ_HZ}s/ventana)")
    print(f"  Intervalo : {INTERVALO}s entre envios")
    print(f"  API       : {API_URL}")
    print("=" * 60)

    try:
        bus = smbus2.SMBus(I2C_BUS)
        inicializar_sensor(bus)
    except Exception as e:
        print(f"❌ No se pudo inicializar el sensor I2C: {e}")
        print("   Comprueba el cableado y que I2C esta habilitado en raspi-config")
        return

    ciclo = 0
    try:
        while True:
            ciclo += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"\n[{timestamp}] Capturando ventana {ciclo} ({MUESTRAS} muestras)...")

            ventana = capturar_ventana(bus)
            datos = calcular_features(ventana)

            print(f"  RMS={datos['RMS']:.4f}  "
                  f"K={datos['Kurtosis']:.4f}  "
                  f"P2P={datos['Peak_to_Peak']:.4f}  "
                  f"Skew={datos['Skewness']:.4f}", end="  →  ")

            resultado = enviar_a_api(datos)
            if resultado:
                estado = resultado.get("estado_maquina", "?")
                simbolo = "✅" if "OK" in estado else "🚨"
                print(f"{simbolo} {estado}")

            # Esperamos el intervalo menos el tiempo que tardo la captura
            time.sleep(max(0, INTERVALO - (MUESTRAS / FREQ_HZ)))

    except KeyboardInterrupt:
        print(f"\n\n  Lectura detenida. Total de ciclos: {ciclo}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
