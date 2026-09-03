"""
AuraPredict — SensorFactory  (Fase 5A)
========================================
Selects the appropriate sensor implementation based on EdgeConfig.

The factory is the ONLY place where sensor_type drives implementation choice.
No other module (EdgePipeline, EdgeScheduler) needs to import concrete sensors.

Supported types (from SensorConfig.sensor_type / YAML sensor.type):
  'mock'    → MockSensor   — synthetic signals, works on any machine, no hardware
  'adxl345' → ADXL345Sensor — MEMS I2C sensor, requires Raspberry Pi + hardware

Architecture:
    EdgeConfig (YAML)
         ↓
    create_sensor(config.sensor)
         ├── 'mock'    → MockSensor(config, MockSensorParams())
         └── 'adxl345' → ADXL345Sensor(config, bus=bus)
         ↓
    SensorInterface (identical from here onwards)
         ↓
    EdgePipeline → AcquisitionSession → AnomalyDetector → ...

The `bus` parameter is for dependency injection in tests (mock I2C bus).
In production it should be None — ADXL345Sensor creates its own SMBus.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from .base_sensor import SensorConfig, SensorInterface

logger = logging.getLogger(__name__)


def create_sensor(
    config: SensorConfig,
    bus:    Optional[Any] = None,
) -> SensorInterface:
    """
    Instantiate and return the sensor implementation for the given config.

    Args:
        config: SensorConfig loaded from EdgeConfig / YAML.
                config.sensor_type determines which sensor is created.
        bus:    Injectable I2C bus for tests (ADXL345 only).
                None in production — sensor creates its own bus.

    Returns:
        A SensorInterface implementation ready to call configure().

    Raises:
        ValueError: if sensor_type is not recognised.
    """
    sensor_type = config.sensor_type.lower().strip()
    logger.debug("Creating sensor: type=%s, id=%s", sensor_type, config.sensor_id)

    if sensor_type == "mock":
        from .mock_sensor import MockSensor, MockSensorParams
        logger.info("Using MockSensor (development mode — no hardware required)")
        return MockSensor(config, MockSensorParams())

    elif sensor_type == "adxl345":
        from .adxl345_sensor import ADXL345Sensor
        addr = config.i2c_address or "0x53"
        bus_n = config.extra.get("i2c_bus", 1)
        logger.info(
            "Using ADXL345Sensor: address=%s, I2C bus=%s, odr_hz=%s",
            addr, bus_n, config.odr_hz,
        )
        return ADXL345Sensor(config, bus=bus)

    else:
        raise ValueError(
            f"Unknown sensor type: '{sensor_type}'. "
            f"Supported: 'mock', 'adxl345'. "
            f"Check sensor.type in the machine YAML config."
        )
