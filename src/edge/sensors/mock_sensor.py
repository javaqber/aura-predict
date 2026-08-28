"""
AuraPredict — Mock Sensor
==========================
Generates synthetic, reproducible vibration signals for pipeline validation.

NOT a physically accurate model of any real machine.
Purpose: produce controlled signals to test the full edge pipeline
before connecting real hardware.

All signals are deterministic given the same seed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .base_sensor import (
    SensorConfig,
    SensorInterface,
    SensorReading,
    SensorReadError,
)


# ─── SIGNAL MODES ─────────────────────────────────────────────────────────────

class SignalMode(str, Enum):
    """Synthetic signal modes for the mock sensor."""
    NORMAL              = "normal"
    IMBALANCE           = "imbalance"
    MISALIGNMENT        = "misalignment"
    LOOSENESS           = "looseness"
    BEARING_DEGRADATION = "bearing_degradation"
    SENSOR_FAILURE      = "sensor_failure"

    @property
    def description(self) -> str:
        return _MODE_DESCRIPTIONS[self]


_MODE_DESCRIPTIONS: dict[SignalMode, str] = {
    SignalMode.NORMAL: (
        "Clean sinusoid + low noise. Represents a healthy machine baseline."
    ),
    SignalMode.IMBALANCE: (
        "Dominant 1× RPM component with harmonic. "
        "High amplitude at fundamental frequency."
    ),
    SignalMode.MISALIGNMENT: (
        "Strong 2× and 3× harmonics alongside the fundamental. "
        "Phase coupling between axes."
    ),
    SignalMode.LOOSENESS: (
        "Sub-harmonics (0.5×, 1.5×, 2.5×) + random impacts. "
        "High peak-to-peak, moderate kurtosis."
    ),
    SignalMode.BEARING_DEGRADATION: (
        "Periodic impulsive pattern at bearing defect frequency. "
        "High kurtosis, elevated high-frequency energy."
    ),
    SignalMode.SENSOR_FAILURE: (
        "Flat signal (all zeros). Simulates a disconnected sensor. "
        "MUST NOT be classified as a machine anomaly."
    ),
}


# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

@dataclass
class MockSensorParams:
    """
    Parameters for the synthetic signal generator.

    axis_amplitude_factor modifies the base amplitude per axis,
    simulating real-world differences in sensor placement direction.
    """
    mode:                   SignalMode = SignalMode.NORMAL
    amplitude_g:            float = 0.07       # Base amplitude (g)
    dominant_freq_hz:       float = 50.0       # Fundamental frequency (Hz)
    noise_amplitude_g:      float = 0.005      # White noise level (g)
    seed:                   int = 42           # RNG seed for reproducibility

    # Per-axis amplitude factors (realistic: Z often dominant in vertical machines)
    axis_amplitude_factor: dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 0.6, "z": 0.3}
    )


# ─── MOCK SENSOR ──────────────────────────────────────────────────────────────

class MockSensor(SensorInterface):
    """
    Mock vibration sensor for testing the edge pipeline.

    Generates per-axis synthetic signals for each failure mode.
    All signals are deterministic given the same seed and reading count.

    Example:
        config = SensorConfig(
            sensor_id="mock_cnc_1",
            sensor_type="mock",
            sampling_rate_hz=3200.0,
            samples_per_window=3200,
            axes=['x', 'y', 'z'],
        )
        params = MockSensorParams(mode=SignalMode.BEARING_DEGRADATION)

        with MockSensor(config, params) as sensor:
            reading = sensor.read()
    """

    def __init__(
        self,
        config: SensorConfig,
        params: Optional[MockSensorParams] = None,
    ) -> None:
        super().__init__(config)
        self._params = params or MockSensorParams()
        self._reading_count = 0

    # ── SensorInterface implementation ────────────────────────────────────────

    def configure(self) -> None:
        """Initialize the mock sensor (no hardware involved)."""
        self._is_configured = True

    def read(self) -> SensorReading:
        """Generate one window of synthetic vibration data."""
        if not self._is_configured:
            raise SensorReadError("Call configure() before read()")

        n = self._config.samples_per_window
        fs = self._config.sampling_rate_hz
        t = np.linspace(0, n / fs, n, endpoint=False)

        t_start = time.time()
        axes_data: dict[str, np.ndarray] = {}

        for axis in self._config.axes:
            if axis not in ("x", "y", "z"):
                continue
            amp_factor = self._params.axis_amplitude_factor.get(axis, 1.0)
            axes_data[axis] = self._generate(t, axis, amp_factor)

        # Simulated monotonic per-sample timestamps
        duration = n / fs
        t_end = t_start + duration
        timestamps = np.linspace(t_start, t_end, n, endpoint=False)

        self._reading_count += 1

        return SensorReading(
            timestamp_start          = t_start,
            timestamp_end            = t_end,
            timestamps               = timestamps,
            axes                     = axes_data,
            sampling_rate_configured = fs,
            sampling_rate_actual     = fs,  # Mock always matches configured
            sensor_id                = self._config.sensor_id,
            sensor_type              = "mock",
            metadata = {
                "mode":          self._params.mode.value,
                "reading_count": self._reading_count,
                "seed":          self._params.seed,
                "description":   self._params.mode.description,
            },
        )

    def get_metadata(self) -> dict:
        return {
            "sensor_type":    "mock",
            "sensor_id":      self._config.sensor_id,
            "mode":           self._params.mode.value,
            "reading_count":  self._reading_count,
            "sampling_rate":  self._config.sampling_rate_hz,
            "description":    self._params.mode.description,
        }

    def close(self) -> None:
        self._is_configured = False

    # ── Signal generation ─────────────────────────────────────────────────────

    def set_mode(self, mode: SignalMode) -> None:
        """Change signal mode without re-configuring the sensor."""
        self._params.mode = mode

    @property
    def mode(self) -> SignalMode:
        return self._params.mode

    def _make_rng(self, axis: str) -> np.random.Generator:
        """Create a deterministic RNG per axis and reading."""
        axis_offset = {"x": 0, "y": 100, "z": 200}.get(axis, 0)
        return np.random.default_rng(
            self._params.seed + axis_offset + self._reading_count
        )

    def _generate(self, t: np.ndarray, axis: str, amp_factor: float) -> np.ndarray:
        """Generate signal for one axis based on the current mode."""
        rng = self._make_rng(axis)
        A   = self._params.amplitude_g * amp_factor
        f0  = self._params.dominant_freq_hz
        σ   = self._params.noise_amplitude_g
        noise = rng.normal(0.0, σ, size=len(t))

        mode = self._params.mode

        # ── SENSOR_FAILURE ── flat zeros, NOT a machine anomaly
        if mode == SignalMode.SENSOR_FAILURE:
            return np.zeros(len(t))

        # ── NORMAL ── clean sinusoid + low noise
        if mode == SignalMode.NORMAL:
            return A * np.sin(2 * np.pi * f0 * t) + noise

        # ── IMBALANCE ── dominant 1× + small 2×
        if mode == SignalMode.IMBALANCE:
            return (
                A * 2.0 * np.sin(2 * np.pi * f0 * t) +
                A * 0.2 * np.sin(2 * np.pi * 2 * f0 * t) +
                noise
            )

        # ── MISALIGNMENT ── strong 2× and 3×
        if mode == SignalMode.MISALIGNMENT:
            return (
                A       * np.sin(2 * np.pi * f0       * t) +
                A * 0.9 * np.sin(2 * np.pi * 2 * f0   * t) +
                A * 0.4 * np.sin(2 * np.pi * 3 * f0   * t) +
                noise
            )

        # ── LOOSENESS ── sub-harmonics + random impacts
        if mode == SignalMode.LOOSENESS:
            n_impacts = 15
            impact_idx = rng.integers(0, len(t), n_impacts)
            impacts = np.zeros(len(t))
            impacts[impact_idx] = A * 4.0 * rng.uniform(0.5, 1.0, n_impacts)
            return (
                A       * np.sin(2 * np.pi * f0 * t) +
                A * 0.5 * np.sin(2 * np.pi * 0.5 * f0 * t) +
                A * 0.4 * np.sin(2 * np.pi * 1.5 * f0 * t) +
                impacts + noise
            )

        # ── BEARING_DEGRADATION ── periodic impulses at bearing freq
        if mode == SignalMode.BEARING_DEGRADATION:
            # Approximate bearing pass frequency outer race ~3.5× fundamental
            bpf = f0 * 3.5
            period_samples = max(1, int(len(t) / max(bpf * (t[-1] - t[0]), 1)))
            impulses = np.zeros(len(t))
            for i in range(0, len(t), period_samples):
                decay_len = min(period_samples // 2, len(t) - i)
                if decay_len <= 0:
                    continue
                decay = np.exp(-np.arange(decay_len) * 300.0 / len(t))
                scale = A * 2.5 * (1.0 + i / len(t))  # progressive degradation
                impulses[i: i + decay_len] += scale * decay
            return (
                A * 0.4 * np.sin(2 * np.pi * f0 * t) +
                impulses +
                noise
            )

        # ── Fallback (should not reach here) ──
        return A * np.sin(2 * np.pi * f0 * t) + noise
