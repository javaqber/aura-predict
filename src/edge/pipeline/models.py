"""
AuraPredict — Fase 2B pipeline data models
==========================================
Lightweight structures that orchestrate existing Fase 1 modules.

Design decisions:
  - RawSignal wraps SensorReading without copying arrays; adds machine context.
  - FeatureSet wraps MultiAxisReading + quality results + maps to BD payload.
  - AnomalyTrigger is a placeholder interface; real detection arrives in Fase 2C.
  - window_id is an internal UUID4 — NOT persisted to the database.
    Neither lecturas_cnc_v2 nor raw_event_windows have a window_id column.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..sensors.base_sensor import SensorReading
from ..feature_extraction import MultiAxisReading
from ..data_quality import DataQualityResult


# ─── RAW SIGNAL ───────────────────────────────────────────────────────────────

@dataclass
class RawSignal:
    """
    Wraps a SensorReading with machine context and a unique window identifier.

    Does NOT copy signal arrays — holds a reference to the original reading.

    window_id is generated once per acquisition window and propagated through
    the pipeline for internal correlation (logging, buffer filenames).
    It is NEVER stored in the database — no DB column accepts it.

    Separation of identifiers:
      machine_id : str  — logical name from EdgeConfig YAML
      maquina_id : int  — integer PK in 'maquinas' table (resolved at startup)
      empresa_id : int  — integer PK in 'empresas' table
    """
    window_id:    str            # UUID4 — internal Edge identifier
    machine_id:   str            # Logical name (e.g. 'Torno_CNC_1')
    maquina_id:   int            # Integer PK in maquinas table
    empresa_id:   int            # Integer PK in empresas table
    reading:      SensorReading  # Original sensor data (arrays not copied)
    acquired_at:  datetime       # UTC datetime of acquisition

    @staticmethod
    def from_reading(
        reading:    SensorReading,
        machine_id: str,
        maquina_id: int,
        empresa_id: int,
    ) -> "RawSignal":
        """Create a RawSignal, generating a new UUID4 as window_id."""
        return RawSignal(
            window_id   = str(uuid.uuid4()),
            machine_id  = machine_id,
            maquina_id  = maquina_id,
            empresa_id  = empresa_id,
            reading     = reading,
            acquired_at = datetime.now(timezone.utc),
        )


# ─── FEATURE SET ──────────────────────────────────────────────────────────────

@dataclass
class FeatureSet:
    """
    Complete output of one acquisition cycle.

    Contains:
      - window_id        : internal UUID (NOT in DB)
      - maquina_id/empresa_id : BD integer IDs
      - multiaxis        : MultiAxisReading from feature_extraction.py
      - quality_per_axis : DataQualityResult per axis from data_quality.py
      - primary_axis     : axis used for frequency-domain features (DB columns)

    Phase 2B values persisted as NULL in lecturas_cnc_v2:
      - anomaly_score  → Fase 2C (Isolation Forest)
      - health_score   → Fase 2C
    """
    window_id:        str
    maquina_id:       int
    empresa_id:       int
    acquired_at:      datetime
    multiaxis:        MultiAxisReading
    quality_per_axis: dict[str, DataQualityResult]
    primary_axis:     str = "x"

    def to_lectura_cnc_payload(self) -> dict:
        """
        Build the kwargs dict for database_v2.repositories.registrar_lectura_cnc().

        Phase 2B:
          resultado    = 'OK - Sin validar'  (anomaly engine not yet active)
          nivel_riesgo = 'Pendiente'          (Fase 2C will compute this)
          anomaly_score / health_score = None (Fase 2C fills these)
          model_version_id = None             (no ML model registered yet)
        """
        ma  = self.multiaxis
        ctx = ma.context
        pq  = self.quality_per_axis.get(self.primary_axis)

        # Sampling rate from primary axis quality result
        sr_actual = None
        sr_loss   = None
        if pq and pq.sampling_rate:
            sr_actual = pq.sampling_rate.actual_hz
            sr_loss   = pq.sampling_rate.sample_loss_fraction

        # Frequency-domain features from primary axis
        primary_vf = ma.get_axis(self.primary_axis)
        dom_freq = dom_amp = spec_energy = None
        band_low = band_mid = band_high  = None
        if primary_vf:
            dom_freq    = primary_vf.freq.dominant_freq
            dom_amp     = primary_vf.freq.dominant_amplitude
            spec_energy = primary_vf.freq.spectral_energy
            be          = primary_vf.freq.band_energies
            band_low    = be.get("low")
            band_mid    = be.get("mid")
            band_high   = be.get("high")

        def _t(axis: str, attr: str) -> Optional[float]:
            """Return a time-domain feature for an axis, or None if unavailable."""
            vf = ma.get_axis(axis)
            return getattr(vf.time, attr, None) if vf else None

        # Overall quality score: minimum across valid (non-sensor-error) axes
        valid_scores = [
            q.quality_score
            for q in self.quality_per_axis.values()
            if not q.is_sensor_error
        ]
        overall_quality = min(valid_scores) if valid_scores else None
        quality_status  = pq.status if pq else "UNKNOWN"

        return dict(
            maquina_id               = self.maquina_id,
            empresa_id               = self.empresa_id,
            resultado                = "OK - Sin validar",  # Fase 2C updates
            nivel_riesgo             = "Pendiente",          # Fase 2C updates
            sampling_rate_configured = ma.sampling_rate_configured,
            sampling_rate_actual     = sr_actual,
            sample_loss_fraction     = sr_loss,
            # Operating context
            rpm_nominal              = ctx.rpm_nominal,
            rpm_real                 = ctx.rpm_real,
            rpm_source               = ctx.rpm_source,
            temperatura_c            = ctx.temperatura,
            carga_pct                = ctx.carga,
            # Time domain — per axis (None if axis failed quality check)
            rms_x                    = _t("x", "rms"),
            rms_y                    = _t("y", "rms"),
            rms_z                    = _t("z", "rms"),
            peak_x                   = _t("x", "peak"),
            peak_y                   = _t("y", "peak"),
            peak_z                   = _t("z", "peak"),
            peak_to_peak_x           = _t("x", "peak_to_peak"),
            peak_to_peak_y           = _t("y", "peak_to_peak"),
            peak_to_peak_z           = _t("z", "peak_to_peak"),
            kurtosis_x               = _t("x", "kurtosis"),
            kurtosis_y               = _t("y", "kurtosis"),
            kurtosis_z               = _t("z", "kurtosis"),
            skewness_x               = _t("x", "skewness"),
            skewness_y               = _t("y", "skewness"),
            skewness_z               = _t("z", "skewness"),
            crest_factor_x           = _t("x", "crest_factor"),
            crest_factor_y           = _t("y", "crest_factor"),
            crest_factor_z           = _t("z", "crest_factor"),
            # Frequency domain — primary axis only
            dominant_freq_hz         = dom_freq,
            dominant_amplitude       = dom_amp,
            spectral_energy          = spec_energy,
            band_low_energy          = band_low,
            band_mid_energy          = band_mid,
            band_high_energy         = band_high,
            # Order analysis — None until rpm_real available (Fase 2C)
            order_1x_energy          = None,
            order_2x_energy          = None,
            order_3x_energy          = None,
            # Signal quality
            signal_quality_score     = overall_quality,
            data_quality_status      = quality_status,
            # Anomaly / Health — None in Fase 2B; computed by Fase 2C
            anomaly_score            = None,
            health_score             = None,
            diagnostico              = "",
            model_version_id         = None,
        )


# ─── ANOMALY TRIGGER ──────────────────────────────────────────────────────────

class AnomalyTrigger(ABC):
    """
    Interface for deciding when to capture a RAW event window.

    Fase 2B: PlaceholderAnomalyTrigger always returns False.
             No RAW capture occurs.
    Fase 2C: IsolationForestTrigger will implement real anomaly detection.

    Keeping this as an interface ensures the pipeline can accept any
    future implementation without modification.
    """

    @abstractmethod
    def should_capture(self, feature_set: FeatureSet) -> bool:
        """
        Return True if a RAW signal window should be saved and uploaded.
        Called once per acquisition cycle, AFTER feature extraction.
        """
        ...


class PlaceholderAnomalyTrigger(AnomalyTrigger):
    """
    Fase 2B placeholder — never triggers RAW capture.

    Exists to:
      1. Satisfy the AnomalyTrigger interface.
      2. Allow the full pipeline to run without anomaly detection.
      3. Be testable as a no-op.

    Replace with IsolationForestTrigger in Fase 2C.
    """

    def should_capture(self, feature_set: FeatureSet) -> bool:
        return False
