# AuraPredict — Hardware Setup: Raspberry Pi + ADXL345

> **Validado por software** — Este procedimiento ha sido probado mediante tests con hardware simulado.
> **Validación física pendiente** — Los pasos marcados con ⚙️ requieren Raspberry Pi + ADXL345 físico.

---

## 1. Requisitos de hardware

| Componente | Especificación mínima |
|---|---|
| Raspberry Pi | 3B+ o superior (recomendado: Pi 4 Model B, 2 GB RAM) |
| Sistema operativo | Raspberry Pi OS Bullseye 64-bit o Bookworm |
| Tarjeta SD | 32 GB Clase 10 (mínimo 16 GB) |
| Sensor | ADXL345 en módulo breakout (3.3V compatible) |
| Cables | Jumpers hembra-hembra de calidad |
| Fuente de alimentación | 5V / 3A oficial de Raspberry Pi |

---

## 2. Conexión ADXL345 ↔ Raspberry Pi (I²C)

El ADXL345 se comunica por I²C usando los pines GPIO 2 (SDA) y GPIO 3 (SCL).

```
ADXL345 pin    →    Raspberry Pi pin    GPIO
────────────────────────────────────────────
VCC            →    Pin 1  (3.3V)       —
GND            →    Pin 6  (GND)        —
SDA            →    Pin 3  (SDA)        GPIO 2
SCL            →    Pin 5  (SCL)        GPIO 3
CS             →    Pin 1  (3.3V)       —    ← fuerza modo I²C
SDO/ALT        →    Pin 1  (3.3V)       —    ← dirección 0x53
```

> ⚠️ **NUNCA conectar VCC a 5V** — el ADXL345 tolera máximo 3.6V.

### Selección de dirección I²C

| SDO/ALT pin | Dirección I²C | Valor en config YAML |
|---|---|---|
| Conectado a 3.3V (HIGH) | **0x53** (defecto) | `i2c_address: "0x53"` |
| Conectado a GND (LOW) | **0x1D** | `i2c_address: "0x1D"` |

### Resistencias pull-up

La mayoría de módulos breakout incluyen resistencias pull-up de 4.7kΩ en SDA y SCL. Si no las incluye, añadirlas externamente entre cada línea y 3.3V.

---

## 3. Configuración I²C en Raspberry Pi OS

### Habilitar I²C ⚙️

```bash
sudo raspi-config
# → Interface Options → I2C → Yes → Finish → Reboot
```

O directamente:

```bash
sudo sed -i 's/#dtparam=i2c_arm=on/dtparam=i2c_arm=on/' /boot/config.txt
echo "i2c-dev" | sudo tee -a /etc/modules
sudo reboot
```

### Verificar que el bus está activo ⚙️

```bash
ls /dev/i2c*
# Debe mostrar: /dev/i2c-1
```

### Instalar herramientas I²C ⚙️

```bash
sudo apt install -y i2c-tools python3-smbus2
# o con pip:
pip install smbus2
```

### Detectar el sensor ⚙️

```bash
sudo i2cdetect -y 1
```

Salida esperada con ADXL345 en 0x53:

```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
50: -- -- -- 53 -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- -- 
```

Si no aparece ningún dispositivo: comprobar cableado, pull-ups y que el sensor está alimentado.

---

## 4. Instalación del software AuraPredict

```bash
# En Raspberry Pi
cd /home/pi
git clone https://github.com/javaqber/aura-predict.git
cd aura-predict

# Entorno virtual
python3 -m venv venv
source venv/bin/activate

# Dependencias
pip install -r requirements.txt
pip install smbus2   # solo en Raspberry Pi

# Variables de entorno
cp .env.example .env
nano .env           # rellenar DATABASE_URL, SUPABASE_URL, etc.
```

---

## 5. Configuración del sensor

Copiar y editar el YAML de la máquina:

```bash
cp config/machines/example_cnc.yaml config/machines/torno_cnc_1.yaml
nano config/machines/torno_cnc_1.yaml
```

Sección de sensor para **ADXL345 físico** (cambiar de `mock` a `adxl345`):

```yaml
sensor:
  type: adxl345           # ← cambiar desde 'mock'
  i2c_address: "0x53"     # ← 0x53 o 0x1D según cableado
  i2c_bus: 1              # ← 1 en la mayoría de Raspberry Pi
  range_g: 2              # ← ±2g para máquinas con vibración baja/media
                          #   ±16g para maquinaria pesada
  odr_hz: 800.0           # ← ODR del registro del sensor
  sampling_rate_hz: 800.0 # ← Debe coincidir con odr_hz
  samples_per_window: 800 # ← 800 samples a 800 Hz = 1 segundo de ventana
  axes: [x, y, z]
```

> **Nota sobre sampling rate:** A 3200 Hz ODR, Python solo puede leer ~400-800 Hz en un loop simple. La frecuencia efectiva real se mide en cada adquisición y se registra en los logs. Recomendado: `odr_hz: 800` para evitar pérdida de muestras.

> **Para usar DATA_READY (opcional):** añadir `check_data_ready: true` en el YAML bajo `sensor:`. Reduce el riesgo de samples duplicados a altas frecuencias.

---

## 6. Verificación del sensor ⚙️

Ejecutar la herramienta de verificación antes de arrancar el scheduler:

```bash
cd /home/pi/aura-predict
source venv/bin/activate

# Verificación básica
python -m edge.sensors.adxl345_verify

# Con dirección alternativa
python -m edge.sensors.adxl345_verify --bus 1 --address 0x1D

# Salida JSON (para scripting)
python -m edge.sensors.adxl345_verify --json
```

Salida esperada si todo es correcto:

```
AuraPredict — ADXL345 Hardware Verification
Bus: 1 | Address: 0x53 | Range: ±2g

  ✅ Smbus2 Importable
  ✅ I2C Bus Open
  ✅ Devid Correct
  ✅ Data Format Rw
  ✅ Bw Rate Rw
  ✅ Measurement Mode
  ✅ Samples Readable
  ✅ Non Constant
  ✅ Physical Range
  ✅ Standby Ok

✅ PASS — 10/10 checks passed
```

---

## 7. Arranque del scheduler ⚙️

```bash
# Manual (primer arranque)
cd /home/pi/aura-predict
source venv/bin/activate
python src/edge_scheduler.py --config config/machines/torno_cnc_1.yaml
```

Buscar en los logs:

```
2025-01-15 10:00:00 [INFO    ] edge.sensors.adxl345_sensor: ADXL345 configured: addr=0x53, odr_reg=0x0D (800 Hz), range_reg=0x00 (scale=0.0039 g/LSB)
2025-01-15 10:00:00 [INFO    ] edge_scheduler: Pipeline started successfully
2025-01-15 10:02:00 [INFO    ] edge_scheduler: Cycle done in 1.2s. Next in 120min. (ok=1, err=0)
```

### Como servicio systemd (producción)

```bash
sudo cp config/systemd/aurapredict-edge.service /etc/systemd/system/
sudo nano /etc/systemd/system/aurapredict-edge.service  # ajustar rutas
sudo systemctl daemon-reload
sudo systemctl enable aurapredict-edge
sudo systemctl start aurapredict-edge
sudo journalctl -u aurapredict-edge -f   # ver logs en tiempo real
```

---

## 8. Diagnóstico de problemas comunes

### El sensor no aparece en i2cdetect

1. ¿I²C habilitado? → `ls /dev/i2c*`
2. ¿VCC a 3.3V? (no 5V)
3. ¿GND conectado?
4. ¿SDA → Pin 3, SCL → Pin 5?
5. ¿CS a 3.3V? (fuerza modo I²C, no SPI)
6. ¿Resistencias pull-up presentes?

### DEVID mismatch (0xE5 esperado, otro valor)

1. Probar la dirección alternativa: `--address 0x1D`
2. Comprobar que CS está a 3.3V (no flotante)
3. Verificar que no hay otro dispositivo en esa dirección

### Señal constante (flat signal)

1. Comprobar que POWER_CTL tiene el bit Measure activo
2. Verificar que los cables no están demasiado largos (>30cm → degradación)
3. Probar a reiniciar: cortar alimentación y reconectar

### Sampling rate mucho más baja que la configurada

Este es el comportamiento NORMAL en Python:
- A 3200 Hz ODR: Python alcanza ~400-800 Hz efectivos en Raspberry Pi 4
- A 800 Hz ODR: Python alcanza ~400-700 Hz efectivos
- El sistema mide la frecuencia real en cada ciclo y la usa para DSP

Reducir `odr_hz` a 800 Hz es el ajuste recomendado para Raspberry Pi.

### Error de I²C durante adquisición

```
ERROR edge.sensors.adxl345_sensor: ADXL345 I2C read error at sample 234/3200: ...
```

Causas probables:
- Interferencia electromagnética (alejar de motores, fuentes de alimentación)
- Cables demasiado largos o mala calidad
- Frecuencia I²C demasiado alta → añadir `dtparam=i2c_arm_baudrate=100000` en `/boot/config.txt`
- Bus I²C compartido con otro dispositivo que no responde

---

## 9. Pasos pendientes de validación física

Los siguientes aspectos solo pueden verificarse con Raspberry Pi + ADXL345 real en una máquina CNC:

- [ ] Frecuencia de muestreo efectiva real en RPi 4 a odr_hz=800
- [ ] Umbral de señal plana correcto para la máquina específica
- [ ] Ajuste de `range_g` según amplitud real de vibraciones
- [ ] Thresholds del FaultClassifier (BEARING/IMBALANCE/LUBRICATION) con señales reales
- [ ] Calibración de baseline con la máquina en estado sano conocido
- [ ] Verificación de que SENSOR_ERROR no produce falsos positivos
- [ ] Comportamiento del scheduler 24/7 con reintentos reales
- [ ] Latencia de alertas en condición de anomalía real
