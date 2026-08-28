"""
AuraPredict — Sensor Interface
================================
Abstract base class for vibration sensors.
Decouples signal processing from hardware specifics.

Implementations:
  - MockSensor       → synthetic signals (testing / demo)
  - ADXL345Sensor    → MEMS I2C sensor (future)
  - IEPESensor       → industrial piezoelectric (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

@dataclass
class SensorConfig:
    """
    Configuration for a vibration sensor.

    sampling_rate_hz is what the software REQUESTS — the actual rate
    must be measured at runtime (see data_quality.detect_sampling_rate).
    Never assume configured_hz == actual_hz.
    """
    sensor_id:            str
    sensor_type:          str                        # 'adxl345' | 'iepe' | 'mock'
    sampling_rate_hz:     float                      # Target / configured rate (Hz)
    odr_hz:               Optional[float] = None     # Sensor ODR register value
    samples_per_window:   int = 4096
    axes:                 list[str] = field(default_factory=lambda: ['x', 'y', 'z'])
    i2c_address:          Optional[str] = None
    extra:                dict = field(default_factory=dict)


@dataclass
class SensorReading:
    """
    Raw data from one acquisition window.

    axes maps axis name to 1D array of raw samples (g or m/s²).
    NEVER modified by the sensor — raw data only.
    Signal processing and feature extraction happen downstream.
    """
    timestamp_start:           float                      # Unix time (start)
    timestamp_end:             float                      # Unix time (end)
    timestamps:                Optional[np.ndarray]       # Per-sample timestamps
    axes:                      dict[str, np.ndarray]      # {'x': arr, 'y': arr, 'z': arr}
    sampling_rate_configured:  float                      # From config
    sampling_rate_actual:      Optional[float]            # Measured (None if unknown)
    sensor_id:                 str
    sensor_type:               str
    metadata:                  dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        if not self.axes:
            return 0
        return len(next(iter(self.axes.values())))

    @property
    def available_axes(self) -> list[str]:
        return list(self.axes.keys())

    @property
    def duration_s(self) -> float:
        return self.timestamp_end - self.timestamp_start


# ─── EXCEPTIONS ────────────────────────────────────────────────────────────────

class SensorConfigurationError(Exception):
    """Raised when sensor initialization fails."""
    pass


class SensorReadError(Exception):
    """Raised when a data acquisition attempt fails."""
    pass


# ─── ABSTRACT INTERFACE ───────────────────────────────────────────────────────

class SensorInterface(ABC):
    """
    Abstract interface for vibration sensors.

    All concrete implementations must implement:
      - configure() → initialize hardware
      - read()      → acquire one window of raw samples
      - get_metadata() → hardware-specific info
      - close()     → release resources

    Usage:
        with MockSensor(config, params) as sensor:
            reading = sensor.read()
    """

    def __init__(self, config: SensorConfig) -> None:
        self._config = config
        self._is_configured = False

    @property
    def config(self) -> SensorConfig:
        return self._config

    @property
    def sensor_id(self) -> str:
        return self._config.sensor_id

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    @abstractmethod
    def configure(self) -> None:
        """
        Initialize and configure the sensor.

        For hardware sensors: open I2C/SPI bus, set ODR, set range.
        For mock sensors: set up signal generators.

        Raises:
            SensorConfigurationError: if initialization fails.
        """

    @abstractmethod
    def read(self) -> SensorReading:
        """
        Acquire one window of vibration data.

        Returns raw samples — no filtering, no feature extraction.
        sampling_rate_actual may differ from configured; callers must
        use data_quality.detect_sampling_rate to verify.

        Raises:
            SensorReadError: if acquisition fails.
        """

    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Return sensor metadata (firmware, calibration, temperature, etc.).
        Content is sensor-specific.
        """

    @abstractmethod
    def close(self) -> None:
        """Release hardware resources. Safe to call multiple times."""

    def __enter__(self) -> "SensorInterface":
        self.configure()
        return self

    def __exit__(self, *args) -> None:
        self.close()
