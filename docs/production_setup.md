# AuraPredict — Production Setup Guide

## 1. Sensor Selection

AuraPredict supports two sensor modes configured via YAML:

| `sensor.type` | Hardware required | Use case |
|---|---|---|
| `mock` | None — PC/Windows | Development, testing, CI |
| `adxl345` | Raspberry Pi + ADXL345 via I2C | Production on machine |

### Switching to ADXL345

In `config/machines/your_machine.yaml`, change:
```yaml
sensor:
  type: adxl345          # was: mock
  i2c_address: "0x53"   # ALT ADDRESS pin HIGH (default) or "0x1D" (LOW)
  i2c_bus: 1            # I2C bus 1 on Raspberry Pi (GPIO 2=SDA, GPIO 3=SCL)
  range_g: 2            # ±2g for low-vibration spindles; ±16g for heavy machinery
  sampling_rate_hz: 3200.0
  odr_hz: 3200.0
  samples_per_window: 3200
```

### ADXL345 Wiring (Raspberry Pi)

| ADXL345 pin | Raspberry Pi pin | GPIO |
|---|---|---|
| VCC | 3.3V (pin 1) | — |
| GND | GND (pin 6) | — |
| SDA | Pin 3 | GPIO 2 |
| SCL | Pin 5 | GPIO 3 |
| CS  | 3.3V (SPI disabled) | — |
| SDO/ALT | 3.3V → address 0x53 | — |

Enable I2C on Raspberry Pi:
```bash
sudo raspi-config → Interface Options → I2C → Enable
```

Verify device is detected:
```bash
sudo i2cdetect -y 1
# Should show 0x53 (or 0x1D)
```

## 2. Required Environment Variables

Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
chmod 600 .env   # Restrict permissions — contains secrets
nano .env
```

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ Always | Supabase PostgreSQL connection string |
| `SUPABASE_URL` | ✅ For sync | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ For sync | Storage API key |
| `SECRET_KEY` | ✅ For API | JWT signing secret |
| `EMAIL_ACTIVO` | Optional | `true` to send real email alerts |
| `EMAIL_ORIGEN` | If email | Gmail address |
| `EMAIL_CONTRASENA` | If email | Gmail App Password (not account password) |
| `EMAIL_DESTINO` | If email | Alert recipient email |
| `MAQUINA_CONFIG` | Optional | Default machine YAML path |
| `LOG_LEVEL` | Optional | DEBUG/INFO/WARNING/ERROR (default: INFO) |
| `LOG_FILE` | Optional | Path to write log file |
| `APP_ENV` | Optional | `development` or `production` |

### Gmail App Password Setup
1. Google Account → Security → 2-Step Verification (enable)
2. Security → App passwords → Create → "AuraPredict"
3. Use the generated 16-character password as `EMAIL_CONTRASENA`

## 3. Starting the Scheduler

### Manual (development/testing)
```bash
# With default config (MAQUINA_CONFIG env var or example_cnc.yaml)
python src/edge_scheduler.py

# With explicit config
python src/edge_scheduler.py --config config/machines/torno_cnc_1.yaml

# With ADXL345 and production logging
APP_ENV=production LOG_LEVEL=INFO \
python src/edge_scheduler.py --config config/machines/torno_cnc_1.yaml
```

### As a systemd service (Raspberry Pi production)
```bash
# 1. Edit paths in config/systemd/aurapredict-edge.service
# 2. Install
sudo cp config/systemd/aurapredict-edge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable aurapredict-edge
sudo systemctl start aurapredict-edge

# 3. Monitor
sudo systemctl status aurapredict-edge
sudo journalctl -u aurapredict-edge -f
```

## 4. Persistent Data Paths (Raspberry Pi)

For production, set these paths to the SD card (not `/tmp` which clears on reboot):

```yaml
buffer:
  base_dir: /home/pi/aura-predict/data/buffer

anomaly:
  model_base_dir: /home/pi/aura-predict/data/models
  raw_base_dir:   /home/pi/aura-predict/data/raw
```

Create directories before first run:
```bash
mkdir -p ~/aura-predict/data/buffer
mkdir -p ~/aura-predict/data/models
mkdir -p ~/aura-predict/data/raw
mkdir -p ~/aura-predict/logs
```

## 5. Failure Modes and Recovery

| Failure | System behaviour | Recovery |
|---|---|---|
| ADXL345 not connected | `SensorConfigurationError` → scheduler logs error and exits | Check wiring, restart service |
| I2C error during read | `RuntimeError` caught → cycle logged → next cycle attempted | Usually self-recovers |
| Supabase DB offline | Readings buffered in LocalBuffer on SD card | Auto-flushed on reconnect |
| Supabase Storage offline | RAW files kept locally, sync retried next cycle | Auto-synced on reconnect |
| SMTP failure | Alert logged in BD with `enviado=False` | Retried on next anomaly after cooldown |
| Scheduler crash | systemd restarts after 30s (Restart=always) | Automatic |
| SD card full | Buffer drops oldest readings; RAW sync stops | Clear old data; increase storage |

## 6. Raspberry Pi Requirements

- Raspberry Pi 3B+ or newer (tested: Pi 4 Model B)
- Raspberry OS Bullseye 64-bit or newer
- Python 3.10+
- smbus2: `pip install smbus2`
- I2C enabled via `raspi-config`
- Network connectivity for Supabase sync
- SD card: minimum 8 GB; recommended 32 GB Class 10
