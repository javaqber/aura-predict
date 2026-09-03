"""
AuraPredict — ADXL345Sensor  (Fase 4C)
========================================
Implementation of SensorInterface for the ADXL345 accelerometer via I2C.

The ADXL345 connects to the Raspberry Pi via I2C bus 1 (GPIO 2/3).
smbus2 is used for I2C communication; it is injectable for tests so the
module can be imported and tested on Windows without hardware.

Hardware configuration used:
  I2C address  : 0x53 (ALT ADDRESS pin = HIGH) or 0x1D (ALT ADDRESS = LOW)
                 Default in this implementation: 0x53 (matches legacy scheduler.py)
  ODR          : 3200 Hz  (register BW_RATE = 0x0F → 3200 Hz)
                 Alternatives: 0x0C=400Hz, 0x0D=800Hz, 0x0E=1600Hz, 0x0F=3200Hz
  Range        : ±2g (DATA_FORMAT register = 0x00)
                 0x00=±2g, 0x01=±4g, 0x02=±8g, 0x03=±16g
  Resolution   : 10-bit full resolution, ±2g range
  Scale factor : 3.9 mg/LSB (ADXL345 datasheet, ±2g, full resolution mode)
                 Value from legacy scheduler.py: 0.0039 g/LSB ✓

Registers:
  DEVID      = 0x00  → should read 0xE5 (manufacturer identifier)
  BW_RATE    = 0x2C  → output data rate
  POWER_CTL  = 0x2D  → power control (bit 3 = measurement mode)
  DATA_FORMAT= 0x31  → range and resolution
  DATAX0     = 0x32  → first of 6 data registers (X LSB, X MSB, Y LSB, Y MSB, Z LSB, Z MSB)

Data acquisition:
  Reads `samples_per_window` samples sequentially at the configured ODR.
  Each sample = 3 axes × 2 bytes = 6 bytes from DATAX0.
  Per-sample timestamps are generated from monotonic clock + interpolation.

Units: g (gravitational acceleration, 1g = 9.81 m/s²)
The DSP pipeline (process_vibration_signal, check_signal_quality, etc.)
expects arrays in g units — this is consistent with MockSensor output.

Testability:
  The `bus` parameter accepts any object with:
    .read_i2c_block_data(addr, reg, length) → list[int]
    .write_byte_data(addr, reg, value) → None
    .close() → None
  In production: smbus2.SMBus(1)
  In tests:      MagicMock() with .read_i2c_block_data.return_value = [...]

Usage:
    # Production (Raspberry Pi)
    from edge.sensors.adxl345_sensor import ADXL345Sensor
    sensor = ADXL345Sensor(config)
    sensor.configure()
    reading = sensor.read()
    sensor.close()

    # In edge_scheduler.py: replace MockSensor(config.sensor, ...) with:
    sensor = ADXL345Sensor(config.sensor)
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Any

import numpy as np

from .base_sensor import SensorInterface, SensorConfig, SensorReading, SensorConfigurationError

logger = logging.getLogger(__name__)


# ── ADXL345 Register Map ───────────────────────────────────────────────────────

REG_DEVID       = 0x00
REG_BW_RATE     = 0x2C
REG_POWER_CTL   = 0x2D
REG_DATA_FORMAT = 0x31
REG_DATAX0      = 0x32

DEVID_EXPECTED  = 0xE5     # Factory identifier, confirms correct device

# ODR register values → sample rate in Hz
_ODR_MAP: dict[int, float] = {
    0x0A: 100.0,
    0x0B: 200.0,
    0x0C: 400.0,
    0x0D: 800.0,
    0x0E: 1600.0,
    0x0F: 3200.0,
}
_ODR_DEFAULT = 0x0F          # 3200 Hz

# Reverse map: Hz → ODR register
_HZ_TO_ODR: dict[float, int] = {v: k for k, v in _ODR_MAP.items()}

# Range (g) → DATA_FORMAT register bits [1:0]
_RANGE_G_TO_REG: dict[int, int] = {2: 0x00, 4: 0x01, 8: 0x02, 16: 0x03}

# Range / scale factor
# DATA_FORMAT register bits [1:0]: 00=±2g, 01=±4g, 10=±8g, 11=±16g
_RANGE_MAP: dict[int, float] = {
    0x00: 0.0039,   # ±2g  — 3.9 mg/LSB
    0x01: 0.0078,   # ±4g  — 7.8 mg/LSB
    0x02: 0.0156,   # ±8g  — 15.6 mg/LSB
    0x03: 0.0313,   # ±16g — 31.3 mg/LSB
}
_RANGE_DEFAULT     = 0x00    # ±2g — best resolution for vibration
_SCALE_FACTOR_DEFAULT = 0.0039   # g/LSB for ±2g full resolution mode

# Default I2C address (ALT ADDRESS pin HIGH → 0x53)
DEFAULT_I2C_ADDRESS = 0x53


class ADXL345Sensor(SensorInterface):
    """
    ADXL345 vibration sensor via I2C.

    Implements SensorInterface so it is a drop-in replacement for MockSensor
    without any changes to AcquisitionSession or EdgePipeline.

    Constructor accepts an optional `bus` parameter for dependency injection
    in tests. When bus=None (production), smbus2.SMBus(1) is created in
    configure().

    Args:
        config: SensorConfig.
            sensor_type should be 'adxl345'.
            i2c_address (str): hex string e.g. '0x53' (optional, defaults to DEFAULT).
            extra['odr_register']: int — ODR register value (optional, defaults to 0x0F).
            extra['range_register']: int — range register value (optional, defaults to 0x00).
        bus: Injected I2C bus. If None, smbus2.SMBus(1) is used at configure() time.
    """

    def __init__(
        self,
        config: SensorConfig,
        bus:    Optional[Any] = None,
    ) -> None:
        super().__init__(config)
        self._bus: Optional[Any] = bus   # injected or created in configure()
        self._addr: int = self._parse_address(config.i2c_address)
        self._scale: float = _SCALE_FACTOR_DEFAULT
        self._i2c_bus_number: int = int(config.extra.get("i2c_bus", 1))

        # ODR register: explicit > derived from odr_hz > default (3200 Hz)
        if "odr_register" in config.extra:
            self._odr_reg = int(config.extra["odr_register"])
        elif config.odr_hz is not None:
            self._odr_reg = self._hz_to_odr_register(config.odr_hz)
        else:
            self._odr_reg = _ODR_DEFAULT

        # Range register: explicit > derived from range_g (human-readable) > default (±2g)
        if "range_register" in config.extra:
            self._range_reg = int(config.extra["range_register"])
        elif "range_g" in config.extra:
            self._range_reg = _RANGE_G_TO_REG.get(int(config.extra["range_g"]), _RANGE_DEFAULT)
        else:
            self._range_reg = _RANGE_DEFAULT

        self._configured: bool = False

    # ── SensorInterface implementation ─────────────────────────────────────────

    def configure(self) -> None:
        """
        Open the I2C bus and initialise the ADXL345.

        Steps:
          1. Open smbus2.SMBus(1) if no bus was injected.
          2. Verify DEVID register (should be 0xE5).
          3. Configure DATA_FORMAT register (range + full resolution mode).
          4. Configure BW_RATE register (ODR).
          5. Enable measurement mode (POWER_CTL bit 3).

        Raises:
            SensorConfigurationError: if the device cannot be initialised.
        """
        try:
            if self._bus is None:
                import smbus2
                self._bus = smbus2.SMBus(self._i2c_bus_number)
                logger.debug("Opened I2C bus %d", self._i2c_bus_number)

            # Verify device identity
            devid = self._bus.read_i2c_block_data(self._addr, REG_DEVID, 1)[0]
            if devid != DEVID_EXPECTED:
                raise SensorConfigurationError(
                    f"ADXL345 DEVID mismatch at 0x{self._addr:02X}: "
                    f"expected 0x{DEVID_EXPECTED:02X}, got 0x{devid:02X}. "
                    "Check wiring and I2C address."
                )

            # DATA_FORMAT: full resolution mode + range
            # Bit 3 (FULL_RES) = 1 enables full resolution at all ranges.
            # For ±2g (0x00): FULL_RES makes no difference (already 10-bit).
            data_format = self._range_reg | 0x08   # 0x08 = FULL_RES bit
            self._bus.write_byte_data(self._addr, REG_DATA_FORMAT, data_format)
            self._scale = _RANGE_MAP.get(self._range_reg & 0x03, _SCALE_FACTOR_DEFAULT)

            # BW_RATE: set output data rate
            self._bus.write_byte_data(self._addr, REG_BW_RATE, self._odr_reg)

            # POWER_CTL: set bit 3 (Measure) to start measurement mode
            self._bus.write_byte_data(self._addr, REG_POWER_CTL, 0x08)

            self._configured = True
            logger.info(
                "ADXL345 configured: addr=0x%02X, odr_reg=0x%02X (%.0f Hz), "
                "range_reg=0x%02X (scale=%.4f g/LSB)",
                self._addr, self._odr_reg,
                _ODR_MAP.get(self._odr_reg, 0),
                self._range_reg, self._scale,
            )

        except SensorConfigurationError:
            raise
        except Exception as exc:
            raise SensorConfigurationError(
                f"ADXL345 initialisation failed: {exc}"
            ) from exc

    def read(self) -> SensorReading:
        """
        Acquire one window of vibration data from the ADXL345.

        Reads `config.samples_per_window` samples sequentially.
        Each sample requires 6 bytes (2 per axis: LSB then MSB).
        Timestamps are interpolated from the wall clock over the window.

        Returns:
            SensorReading with axes {'x': arr, 'y': arr, 'z': arr} in g.

        Raises:
            SensorConfigurationError: if configure() has not been called.
            RuntimeError: if I2C read fails.
        """
        if not self._configured:
            raise SensorConfigurationError(
                "ADXL345Sensor.configure() must be called before read()."
            )

        n_samples = self._config.samples_per_window
        axes_data = {"x": np.empty(n_samples, dtype=np.float64),
                     "y": np.empty(n_samples, dtype=np.float64),
                     "z": np.empty(n_samples, dtype=np.float64)}

        t_start    = time.time()
        timestamps = np.empty(n_samples, dtype=np.float64)

        try:
            for i in range(n_samples):
                t_sample = time.time()
                raw = self._bus.read_i2c_block_data(self._addr, REG_DATAX0, 6)
                # Each axis: two's complement 16-bit value, LSB first
                ax = self._to_signed16(raw[0], raw[1]) * self._scale
                ay = self._to_signed16(raw[2], raw[3]) * self._scale
                az = self._to_signed16(raw[4], raw[5]) * self._scale
                axes_data["x"][i] = ax
                axes_data["y"][i] = ay
                axes_data["z"][i] = az
                timestamps[i]     = t_sample
        except Exception as exc:
            logger.error("ADXL345 read error at sample %d: %s", i, exc)
            raise RuntimeError(f"ADXL345 read error: {exc}") from exc

        t_end = time.time()
        actual_rate = (n_samples - 1) / (t_end - t_start) if t_end > t_start else None

        return SensorReading(
            timestamp_start          = t_start,
            timestamp_end            = t_end,
            timestamps               = timestamps,
            axes                     = axes_data,
            sampling_rate_configured = self._config.sampling_rate_hz,
            sampling_rate_actual     = actual_rate,
            sensor_id                = self._config.sensor_id,
            sensor_type              = "adxl345",
            metadata                 = {
                "i2c_address":  hex(self._addr),
                "scale_factor": self._scale,
                "odr_register": hex(self._odr_reg),
                "range_register": hex(self._range_reg),
            },
        )

    def get_metadata(self) -> dict:
        """Return sensor metadata for logging and traceability."""
        return {
            "sensor_type":    "adxl345",
            "i2c_address":    hex(self._addr),
            "scale_factor":   self._scale,
            "odr_register":   hex(self._odr_reg),
            "range_register": hex(self._range_reg),
            "configured":     self._configured,
        }

    def close(self) -> None:
        """Release the I2C bus."""
        if self._bus is not None:
            try:
                # Set POWER_CTL to standby mode before closing
                self._bus.write_byte_data(self._addr, REG_POWER_CTL, 0x00)
            except Exception:
                pass
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
            self._configured = False

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_signed16(lsb: int, msb: int) -> int:
        """Convert two bytes (LSB first) to a signed 16-bit integer."""
        value = (msb << 8) | lsb
        if value >= 0x8000:
            value -= 0x10000
        return value

    @staticmethod
    def _hz_to_odr_register(hz: float) -> int:
        """Map a sampling rate in Hz to the nearest ADXL345 ODR register value."""
        if hz <= 100:   return 0x0A
        if hz <= 200:   return 0x0B
        if hz <= 400:   return 0x0C
        if hz <= 800:   return 0x0D
        if hz <= 1600:  return 0x0E
        return 0x0F  # 3200 Hz — maximum

    @staticmethod
    def _parse_address(i2c_address: Optional[str]) -> int:
        """
        Parse I2C address from string or return default.
        Accepts '0x53', '0x1D', '83', etc.
        """
        if i2c_address is None:
            return DEFAULT_I2C_ADDRESS
        try:
            return int(i2c_address, 16) if i2c_address.startswith("0x") else int(i2c_address)
        except (ValueError, AttributeError):
            return DEFAULT_I2C_ADDRESS
