from datetime import datetime
import requests
import math
import random
import time
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


# ─── CONFIGURACION ────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/predict/bearing"
MAQUINA = "Torno Mecaniperz 2"      # Debe coincidir con el nombre en el dashboard
INTERVALO = 5                   # Segundos entre cada lectura
TOTAL_CICLOS = 0                   # 0 = infinito, N = parar tras N lecturas

# Modo de simulacion:
# "sano"      → genera datos de maquina en buen estado
# "degradado" → simula un rodamiento que se va deteriorando progresivamente
# "fallo"     → genera datos de fallo claro
MODO = "degradado"

# ─── GENERADOR DE DATOS ───────────────────────────────────────────────────────

ciclo_actual = 0


def generar_vibracion(ciclo):
    """
    Genera valores de vibracion sinteticos pero realistas.
    En modo degradado, los valores empeoran gradualmente con cada ciclo.
    """
    if MODO == "sano":
        rms = random.gauss(0.07, 0.005)
        peak = random.gauss(0.84, 0.05)
        kurtosis = random.gauss(0.62, 0.08)
        skewness = random.gauss(0.08, 0.02)

    elif MODO == "degradado":
        # Simulamos deterioro progresivo a lo largo de los ciclos
        factor = ciclo / 100.0          # va de 0 a 1 segun los ciclos
        ruido = random.gauss(0, 0.01)

        rms = 0.07 + factor * 0.30 + ruido
        peak = 0.84 + factor * 1.50 + ruido * 5
        kurtosis = 0.62 + factor * 3.50 + ruido * 2
        skewness = 0.08 + factor * 0.80 + ruido

    elif MODO == "fallo":
        rms = random.gauss(0.45, 0.05)
        peak = random.gauss(3.80, 0.30)
        kurtosis = random.gauss(5.20, 0.50)
        skewness = random.gauss(1.10, 0.15)

    else:
        raise ValueError(f"Modo desconocido: {MODO}")

    return {
        "maquina":       MAQUINA,
        "RMS":           round(rms, 4),
        "Peak_to_Peak":  round(peak, 4),
        "Kurtosis":      round(kurtosis, 4),
        "Skewness":      round(skewness, 4)
    }


def enviar_a_api(datos):
    """Envia los datos al endpoint de la API y devuelve la respuesta."""
    try:
        respuesta = requests.post(API_URL, json=datos, timeout=5)
        if respuesta.status_code == 200:
            return respuesta.json()
        else:
            print(
                f"  ⚠️  Error HTTP {respuesta.status_code}: {respuesta.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("  ❌  No se puede conectar con la API.")
        print(f"      Asegurate de que la API esta corriendo en {API_URL}")
        return None
    except Exception as e:
        print(f"  ❌  Error inesperado: {e}")
        return None


# ─── BUCLE PRINCIPAL ──────────────────────────────────────────────────────────

def main():
    global ciclo_actual

    print("=" * 60)
    print("  🔮 AuraPredict — Simulador de Sensores IoT")
    print("=" * 60)
    print(f"  Maquina   : {MAQUINA}")
    print(f"  Modo      : {MODO.upper()}")
    print(f"  Intervalo : {INTERVALO}s entre lecturas")
    print(f"  API       : {API_URL}")
    print("=" * 60)
    print("  Pulsa Ctrl+C para detener")
    print()

    try:
        while True:
            ciclo_actual += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            datos = generar_vibracion(ciclo_actual)
            print(f"[{timestamp}] Ciclo {ciclo_actual:04d} | "
                  f"RMS={datos['RMS']:.4f}  "
                  f"Kurtosis={datos['Kurtosis']:.4f}  "
                  f"P2P={datos['Peak_to_Peak']:.4f}", end="  →  ")

            resultado = enviar_a_api(datos)

            if resultado:
                estado = resultado.get("estado_maquina", "?")
                riesgo = resultado.get("nivel_riesgo", "?")
                simbolo = "✅" if "OK" in estado else "🚨"
                print(f"{simbolo} {estado} | Riesgo: {riesgo}")
            else:
                print("Sin respuesta")

            if TOTAL_CICLOS > 0 and ciclo_actual >= TOTAL_CICLOS:
                print(f"\n  Simulacion completada ({TOTAL_CICLOS} ciclos).")
                break

            time.sleep(INTERVALO)

    except KeyboardInterrupt:
        print(
            f"\n\n  Simulacion detenida. Total de lecturas enviadas: {ciclo_actual}")


if __name__ == "__main__":
    main()
