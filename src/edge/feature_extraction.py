"""
AuraPredict — Feature Extraction
==================================
Extracts structured features from processed vibration signals.

Key design decisions:
  - Axes X/Y/Z are ALWAYS kept separate — never silently combined.
  - OperatingContext carries rpm_nominal vs rpm_real explicitly.
  - rpm_nominal is NEVER used silently as rpm_real.
  - OrderAnalysisPrep prepares for future order-based analysis
    but does NOT implement it yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .signal_processing import SignalConfig, process_vibration_signal


# ─── FEATURE DATACLASSES ──────────────────────────────────────────────────────

@dataclass
class TimeFeatures:
    """Time-domain features for a single vibration axis."""
    rms:          float
    peak:         float
    peak_to_peak: float
    std:          float
    kurtosis:     float
    skewness:     float
    crest_factor: float
    axis:         str    # 'x', 'y', 'z', or 'total'


@dataclass
class FreqFeatures:
    """Frequency-domain features for a single vibration axis."""
    dominant_freq:      float              # Hz
    dominant_amplitude: float              # g or m/s²
    spectral_energy:    float
    band_energies:      dict[str, float]   # {band_name: energy}
    axis:               str


@dataclass
class VibrationFeatures:
    """
    Combined time + frequency features for one axis.
    Named VibrationFeatures per spec (analogous to AxisFeatures internally).
    """
    time: TimeFeatures
    freq: FreqFeatures
    axis: str

    # Convenience accessors matching signal_processing.py legacy API
    @property
    def RMS(self) -> float:          return self.time.rms
    @property
    def Peak_to_Peak(self) -> float: return self.time.peak_to_peak
    @property
    def Kurtosis(self) -> float:     return self.time.kurtosis
    @property
    def Skewness(self) -> float:     return self.time.skewness


# ─── OPERATING CONTEXT ────────────────────────────────────────────────────────

@dataclass
class OperatingContext:
    """
    Operating conditions at the time of a vibration reading.

    IMPORTANT: rpm_nominal is the nameplate/config value.
    rpm_real comes from an actual measurement source (encoder, OPC-UA, etc.).
    rpm_nominal is NEVER used silently as rpm_real.
    If rpm_real is None, order analysis is NOT available.
    """
    rpm_nominal: Optional[float] = None  # From machine config YAML
    rpm_real:    Optional[float] = None  # From encoder / OPC-UA / Modbus / etc.
    rpm_source:  Optional[str]   = None  # 'encoder'|'opc_ua'|'modbus'|None
    temperatura: Optional[float] = None  # °C
    carga:       Optional[float] = None  # % or kW — machine-specific units

    @property
    def rpm_real_available(self) -> bool:
        """True only when rpm_real is from a real measurement source."""
        return self.rpm_real is not None and self.rpm_real > 0

    def rpm_for_analysis(self) -> tuple[Optional[float], str]:
        """
        Returns (rpm, source_description) for use in analysis.
        NEVER substitutes rpm_nominal for rpm_real.
        Callers must check is not None before using rpm.
        """
        if self.rpm_real is not None:
            return self.rpm_real, self.rpm_source or "real_unknown_source"
        return None, "not_available"


# ─── ORDER ANALYSIS PREPARATION ───────────────────────────────────────────────

@dataclass
class OrderAnalysisPrep:
    """
    Interface preparation for future order-based vibration analysis.

    NOT YET IMPLEMENTED — provides the structure for calculating
    1X, 2X, 3X order energies when rpm_real becomes available.

    Usage:
        prep = OrderAnalysisPrep(context=ctx)
        if prep.is_available:
            f_1x = prep.order_frequency(1)
            # Future: extract energy at f_1x from FFT
        else:
            # rpm_real not available — skip order analysis
            pass
    """
    context: OperatingContext
    orders:  list[int] = field(default_factory=lambda: [1, 2, 3])

    @property
    def is_available(self) -> bool:
        """True only when rpm_real is known — NEVER falls back to rpm_nominal."""
        return self.context.rpm_real_available

    def order_frequency(self, order: int) -> Optional[float]:
        """
        Frequency (Hz) for the given order based on rpm_real.
        Returns None if rpm_real is not available.
        NEVER uses rpm_nominal as substitute.
        """
        if not self.is_available:
            return None
        rpm, _ = self.context.rpm_for_analysis()
        return order * rpm / 60.0

    def order_frequencies(self) -> dict[str, Optional[float]]:
        """
        Map of order label → frequency_hz for configured orders.
        Values are None when rpm_real is not available.
        """
        return {f"{o}X": self.order_frequency(o) for o in self.orders}


# ─── MULTI-AXIS READING ───────────────────────────────────────────────────────

@dataclass
class MultiAxisReading:
    """
    Complete feature set for a multi-axis vibration measurement.

    Axes are always kept separate. A 'total' (magnitude) axis
    can optionally be computed for legacy compatibility.

    From data_quality.py:
      - sampling_rate_actual is the MEASURED rate (from timestamps).
      - It may differ from sampling_rate_configured.
    """
    timestamp:                 str
    sensor_id:                 str
    sampling_rate_configured:  float
    sampling_rate_actual:      Optional[float]

    x:     Optional[VibrationFeatures] = None
    y:     Optional[VibrationFeatures] = None
    z:     Optional[VibrationFeatures] = None
    total: Optional[VibrationFeatures] = None  # sqrt(x²+y²+z²) — explicit only

    context:    OperatingContext = field(default_factory=OperatingContext)
    order_prep: Optional[OrderAnalysisPrep] = None

    def get_axis(self, axis: str) -> Optional[VibrationFeatures]:
        return {"x": self.x, "y": self.y, "z": self.z, "total": self.total}.get(axis)

    def available_axes(self) -> list[str]:
        return [a for a in ("x", "y", "z", "total") if getattr(self, a) is not None]

    def to_api_dict(self, axis: str = "x") -> dict:
        """
        Backward-compatible dict for POST /predict/bearing.
        axis: which axis to use for the legacy 4-feature dict.
        """
        vf = self.get_axis(axis)
        if vf is None:
            raise ValueError(f"Axis '{axis}' not available in this reading")
        return {
            "maquina":      self.sensor_id,
            "RMS":          round(vf.time.rms, 4),
            "Peak_to_Peak": round(vf.time.peak_to_peak, 4),
            "Kurtosis":     round(vf.time.kurtosis, 4),
            "Skewness":     round(vf.time.skewness, 4),
        }


# ─── EXTRACTION FUNCTIONS ─────────────────────────────────────────────────────

def extract_vibration_features(
    signal: np.ndarray,
    axis:   str,
    fs:     float,
    config: Optional[SignalConfig] = None,
) -> VibrationFeatures:
    """
    Extract time + frequency features for a single axis signal.

    Calls signal_processing.process_vibration_signal internally.
    validate=False because data quality was checked by data_quality.py.

    Args:
        signal: 1D array of acceleration samples (g or m/s²).
        axis:   Axis label: 'x', 'y', 'z', or 'total'.
        fs:     Sampling rate (Hz) — use actual, not configured.
        config: Processing configuration. Uses defaults if None.

    Returns:
        VibrationFeatures with TimeFeatures and FreqFeatures.
    """
    vf = process_vibration_signal(signal, fs, config=config, validate=False)

    time_f = TimeFeatures(
        rms          = vf.time.rms,
        peak         = vf.time.peak,
        peak_to_peak = vf.time.peak_to_peak,
        std          = vf.time.std,
        kurtosis     = vf.time.kurtosis,
        skewness     = vf.time.skewness,
        crest_factor = vf.time.crest_factor,
        axis         = axis,
    )

    freq_f = FreqFeatures(
        dominant_freq      = vf.frequency.dominant_freq,
        dominant_amplitude = vf.frequency.dominant_amplitude,
        spectral_energy    = vf.frequency.spectral_energy,
        band_energies      = vf.frequency.band_energies,
        axis               = axis,
    )

    return VibrationFeatures(time=time_f, freq=freq_f, axis=axis)


def extract_multiaxis_features(
    signals:                   dict[str, np.ndarray],
    sensor_id:                 str,
    sampling_rate_configured:  float,
    sampling_rate_actual:      Optional[float] = None,
    timestamp:                 str = "",
    context:                   Optional[OperatingContext] = None,
    config:                    Optional[SignalConfig] = None,
    include_total:             bool = False,
) -> MultiAxisReading:
    """
    Extract VibrationFeatures for each axis independently.

    Axes are NEVER combined silently. If include_total=True, the
    total magnitude axis sqrt(x²+y²+z²) is computed EXPLICITLY and
    clearly labelled as 'total' — not substituted for individual axes.

    Args:
        signals:                  Dict mapping axis label to signal array.
                                  e.g. {'x': arr_x, 'y': arr_y, 'z': arr_z}
        sensor_id:                Sensor identifier string.
        sampling_rate_configured: Target rate from config (Hz).
        sampling_rate_actual:     Measured rate from data_quality (Hz). May differ.
        timestamp:                ISO timestamp string.
        context:                  Operating conditions (RPM, temperature, etc.).
        config:                   Signal processing config.
        include_total:            If True, compute explicit total magnitude axis.

    Returns:
        MultiAxisReading with per-axis VibrationFeatures.
    """
    # Use actual rate if available — never silently fall back to configured
    fs  = sampling_rate_actual if sampling_rate_actual is not None else sampling_rate_configured
    ctx = context or OperatingContext()

    axis_features: dict[str, Optional[VibrationFeatures]] = {
        "x": None, "y": None, "z": None, "total": None
    }

    for axis_name, sig in signals.items():
        if axis_name in ("x", "y", "z"):
            axis_features[axis_name] = extract_vibration_features(
                sig, axis_name, fs, config
            )

    # Explicit total magnitude (only when requested AND all axes present)
    if include_total and all(a in signals for a in ("x", "y", "z")):
        total_sig = np.sqrt(
            signals["x"] ** 2 +
            signals["y"] ** 2 +
            signals["z"] ** 2
        )
        axis_features["total"] = extract_vibration_features(
            total_sig, "total", fs, config
        )

    order_prep = OrderAnalysisPrep(context=ctx)

    return MultiAxisReading(
        timestamp                = timestamp,
        sensor_id                = sensor_id,
        sampling_rate_configured = sampling_rate_configured,
        sampling_rate_actual     = sampling_rate_actual,
        x                        = axis_features["x"],
        y                        = axis_features["y"],
        z                        = axis_features["z"],
        total                    = axis_features["total"],
        context                  = ctx,
        order_prep               = order_prep,
    )
