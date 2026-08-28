"""
AuraPredict — Data Quality Module
===================================
Validates signal quality and detects real sampling rate.

KEY RULE: a SENSOR_ERROR must NEVER be classified as a machine anomaly.
If DataQualityResult.is_sensor_error is True, the reading must NOT
be passed to anomaly detection.

Detects:
  - Real vs configured sampling rate (from timestamps)
  - Dropped / lost samples
  - Timestamp anomalies (non-monotonic, large gaps)
  - Flat signal (sensor disconnected)
  - Saturated / clipped signal (ADC overloaded)
  - NaN / Inf values
  - Amplitude outside physical range
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─── RESULT STRUCTURES ─────────────────────────────────────────────────────────

@dataclass
class SamplingRateResult:
    """
    Result of sampling rate analysis.

    actual_hz is measured from timestamps — NOT assumed from config.
    Never use configured_hz as a substitute for actual_hz.
    """
    configured_hz:        float
    actual_hz:            Optional[float]      # Measured from timestamps
    odr_hz:               Optional[float]      # Sensor ODR register value
    sample_count:         int
    expected_samples:     Optional[int]        # Based on configured rate + duration
    sample_loss_fraction: Optional[float]      # 0.0 = no loss, 1.0 = complete loss
    mismatch_detected:    bool
    mismatch_fraction:    Optional[float]      # |actual - configured| / configured
    warning:              Optional[str]

    @property
    def effective_hz(self) -> float:
        """
        Sampling rate to use for analysis.
        Returns actual_hz if available, otherwise configured_hz.
        """
        return self.actual_hz if self.actual_hz is not None else self.configured_hz

    @property
    def nyquist_hz(self) -> float:
        """Maximum reliably detectable frequency (Nyquist of effective rate)."""
        return self.effective_hz / 2.0


@dataclass
class DataQualityResult:
    """
    Structured result of signal quality checks.

    NEVER pass this reading to anomaly detection if is_sensor_error is True.
    A sensor failure must not be reported as a machine failure.
    """
    is_valid:              bool           # Ready for signal processing
    is_sensor_error:       bool           # Sensor problem — not a machine fault
    quality_score:         float          # 0.0 – 1.0

    sampling_rate:         Optional[SamplingRateResult] = None

    has_nan_inf:           bool = False
    is_flat:               bool = False   # Sensor disconnected / dead
    is_saturated:          bool = False   # ADC clipping
    is_out_of_range:       bool = False   # Physically impossible values
    has_timestamp_anomaly: bool = False

    warnings: list[str] = field(default_factory=list)
    errors:   list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Human-readable status string."""
        if self.is_sensor_error:
            return "SENSOR_ERROR"
        if not self.is_valid:
            return "INVALID"
        if self.quality_score >= 0.9:
            return "OK"
        if self.quality_score >= 0.7:
            return "DEGRADED"
        return "POOR"


# ─── SAMPLING RATE DETECTION ───────────────────────────────────────────────────

def detect_sampling_rate(
    timestamps:            Optional[np.ndarray],
    n_samples:             int,
    configured_hz:         float,
    duration_s:            Optional[float] = None,
    odr_hz:                Optional[float] = None,
    mismatch_threshold:    float = 0.05,
) -> SamplingRateResult:
    """
    Detect actual sampling rate from timestamps or sample count.

    Args:
        timestamps:           Per-sample Unix timestamps. Can be None.
        n_samples:            Number of samples in the window.
        configured_hz:        Target rate from config YAML.
        duration_s:           Expected window duration (seconds).
                              Used as fallback if no timestamps provided.
        odr_hz:               ODR from sensor register (informational only).
        mismatch_threshold:   Fraction beyond which we flag a mismatch.
                              Default 5%.

    Returns:
        SamplingRateResult.
        actual_hz is None if neither timestamps nor duration_s are provided.
    """
    actual_hz:            Optional[float] = None
    sample_loss_fraction: Optional[float] = None
    expected_samples:     Optional[int]   = None
    mismatch_fraction:    Optional[float] = None
    warning:              Optional[str]   = None

    # ── Derive from timestamps (most accurate) ────────────────────────────────
    if timestamps is not None and len(timestamps) >= 2:
        ts         = np.asarray(timestamps, dtype=np.float64)
        total_time = float(ts[-1] - ts[0])

        if total_time > 0:
            actual_hz = (len(ts) - 1) / total_time
            expected_samples = max(1, int(configured_hz * total_time))
            sample_loss_fraction = max(
                0.0,
                (expected_samples - len(ts)) / expected_samples
            )
            mismatch_fraction = abs(actual_hz - configured_hz) / configured_hz

    # ── Fallback: estimate from sample count + duration ───────────────────────
    elif duration_s is not None and duration_s > 0:
        actual_hz         = n_samples / duration_s
        expected_samples  = max(1, int(configured_hz * duration_s))
        mismatch_fraction = abs(actual_hz - configured_hz) / configured_hz

    # ── Check mismatch ────────────────────────────────────────────────────────
    mismatch_detected = (
        mismatch_fraction is not None and
        mismatch_fraction > mismatch_threshold
    )

    if mismatch_detected and actual_hz is not None:
        warning = (
            f"Sampling rate mismatch: configured={configured_hz:.0f} Hz, "
            f"actual={actual_hz:.1f} Hz "
            f"({mismatch_fraction * 100:.1f}% difference). "
            f"FFT results above {actual_hz / 2:.0f} Hz may be unreliable."
        )

    return SamplingRateResult(
        configured_hz        = configured_hz,
        actual_hz            = actual_hz,
        odr_hz               = odr_hz,
        sample_count         = n_samples,
        expected_samples     = expected_samples,
        sample_loss_fraction = sample_loss_fraction,
        mismatch_detected    = mismatch_detected,
        mismatch_fraction    = mismatch_fraction,
        warning              = warning,
    )


# ─── SIGNAL QUALITY CHECK ─────────────────────────────────────────────────────

def check_signal_quality(
    signal:                   np.ndarray,
    configured_hz:            float,
    timestamps:               Optional[np.ndarray] = None,
    duration_s:               Optional[float]       = None,
    odr_hz:                   Optional[float]       = None,
    expected_range_g:         tuple[float, float]   = (-16.0, 16.0),
    saturation_fraction_limit: float                = 0.05,
    flat_std_threshold:       float                 = 1e-6,
    mismatch_threshold:       float                 = 0.05,
) -> DataQualityResult:
    """
    Full data quality check for a single-axis signal.

    NEVER classifies sensor failures (flat signal, NaN, etc.) as
    machine anomalies. Always check is_sensor_error before processing.

    Args:
        signal:                    1D array of acceleration samples (g or m/s²).
        configured_hz:             Sampling rate from config YAML.
        timestamps:                Per-sample Unix timestamps. Optional.
        duration_s:                Expected window duration (seconds). Optional.
        odr_hz:                    Sensor ODR from config. Informational.
        expected_range_g:          Physical amplitude range. Exceeding = sensor error.
        saturation_fraction_limit: Fraction of samples at peak → clipped.
        flat_std_threshold:        Below this std → flat signal (sensor dead).
        mismatch_threshold:        Sampling rate mismatch tolerance fraction.

    Returns:
        DataQualityResult with is_sensor_error flag and quality_score 0–1.
    """
    warnings: list[str] = []
    errors:   list[str] = []
    quality_score  = 1.0
    is_sensor_error = False

    # 1. Minimum length ────────────────────────────────────────────────────────
    if signal.ndim != 1 or len(signal) < 32:
        return DataQualityResult(
            is_valid        = False,
            is_sensor_error = True,
            quality_score   = 0.0,
            is_flat         = True,
            errors          = [f"Señal demasiado corta: {len(signal)} muestras (mínimo 32)"],
        )

    # 2. NaN / Inf ─────────────────────────────────────────────────────────────
    has_nan_inf = not bool(np.all(np.isfinite(signal)))
    if has_nan_inf:
        return DataQualityResult(
            is_valid        = False,
            is_sensor_error = True,
            quality_score   = 0.0,
            has_nan_inf     = True,
            errors          = ["Señal contiene NaN o Inf — posible error de hardware. "
                               "No clasificar como anomalía de máquina."],
        )

    # 3. Flat signal (sensor disconnected) ────────────────────────────────────
    std = float(np.std(signal))
    is_flat = std < flat_std_threshold
    if is_flat:
        return DataQualityResult(
            is_valid        = False,
            is_sensor_error = True,
            quality_score   = 0.0,
            is_flat         = True,
            errors          = [
                f"Señal plana (std={std:.2e}) — posible sensor desconectado. "
                "Reportar como SENSOR_ERROR, no como anomalía de máquina."
            ],
        )

    # 4. Physical range check ──────────────────────────────────────────────────
    sig_max = float(np.max(signal))
    sig_min = float(np.min(signal))
    is_out_of_range = (sig_max > expected_range_g[1] or sig_min < expected_range_g[0])
    if is_out_of_range:
        warnings.append(
            f"Amplitud fuera del rango esperado "
            f"({expected_range_g[0]:.1f} a {expected_range_g[1]:.1f} g). "
            f"Señal: [{sig_min:.3f}, {sig_max:.3f}] g. "
            "Verificar calibración o rango del sensor."
        )
        quality_score -= 0.30

    # 5. Saturated / clipped signal ────────────────────────────────────────────
    max_abs = float(np.max(np.abs(signal)))
    is_saturated = False
    if max_abs > 0:
        n_at_peak    = int(np.sum(np.abs(signal) >= max_abs * 0.9995))
        sat_fraction = n_at_peak / len(signal)
        is_saturated = sat_fraction > saturation_fraction_limit
        if is_saturated:
            warnings.append(
                f"Señal saturada — {sat_fraction * 100:.1f}% de muestras "
                f"en el límite ({max_abs:.4f} g). ADC posiblemente sobrecargado."
            )
            quality_score -= 0.40

    # 6. Sampling rate detection ───────────────────────────────────────────────
    sr = detect_sampling_rate(
        timestamps         = timestamps,
        n_samples          = len(signal),
        configured_hz      = configured_hz,
        duration_s         = duration_s,
        odr_hz             = odr_hz,
        mismatch_threshold = mismatch_threshold,
    )
    if sr.warning:
        warnings.append(sr.warning)
        quality_score -= 0.15

    if sr.sample_loss_fraction is not None and sr.sample_loss_fraction > 0.10:
        warnings.append(
            f"Pérdida de muestras: {sr.sample_loss_fraction * 100:.1f}% de muestras "
            f"perdidas (esperadas={sr.expected_samples}, recibidas={len(signal)})."
        )
        quality_score -= 0.15

    # 7. Timestamp anomalies ───────────────────────────────────────────────────
    has_timestamp_anomaly = False
    if timestamps is not None and len(timestamps) >= 2:
        ts    = np.asarray(timestamps, dtype=np.float64)
        diffs = np.diff(ts)

        if bool(np.any(diffs <= 0)):
            has_timestamp_anomaly = True
            warnings.append("Timestamps no monótonicos detectados (posible error de reloj).")
            quality_score -= 0.20

        expected_dt = 1.0 / configured_hz
        n_large_gaps = int(np.sum(diffs > expected_dt * 5.0))
        if n_large_gaps > 0:
            has_timestamp_anomaly = True
            warnings.append(
                f"{n_large_gaps} gaps grandes en timestamps "
                f"(>5× período esperado de {expected_dt * 1000:.2f} ms)."
            )
            quality_score -= 0.10

    # ── Final score and validity ───────────────────────────────────────────────
    quality_score = round(max(0.0, min(1.0, quality_score)), 3)
    is_valid      = quality_score > 0.30 and not is_sensor_error

    return DataQualityResult(
        is_valid              = is_valid,
        is_sensor_error       = is_sensor_error,
        quality_score         = quality_score,
        sampling_rate         = sr,
        has_nan_inf           = has_nan_inf,
        is_flat               = is_flat,
        is_saturated          = is_saturated,
        is_out_of_range       = is_out_of_range,
        has_timestamp_anomaly = has_timestamp_anomaly,
        warnings              = warnings,
        errors                = errors,
    )
