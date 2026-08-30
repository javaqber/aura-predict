"""
AuraPredict — AcquisitionSession
==================================
Orchestrates one acquisition cycle:

  SensorReading → RawSignal → quality checks → MultiAxisReading → FeatureSet

Does NOT implement DSP or feature math — delegates entirely to Fase 1 modules:
  data_quality.check_signal_quality()      → DataQualityResult per axis
  feature_extraction.extract_multiaxis_features() → MultiAxisReading

No Preprocessor class. No ProcessedSignal class. The existing functions
already provide the correct separation of concerns.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from ..sensors.base_sensor import SensorInterface, SensorReading
from ..data_quality import check_signal_quality, DataQualityResult
from ..feature_extraction import extract_multiaxis_features, OperatingContext
from .models import RawSignal, FeatureSet
from ..config.edge_config import EdgeConfig


class AcquisitionSession:
    """
    Processes one SensorReading into a FeatureSet.

    Wires together the existing Fase 1 functions with the machine context
    provided by EdgeConfig. Returns None if all axes fail sensor checks.

    This class is the bridge between the sensor layer and the analytics layer.
    It does not compute features itself — it delegates to existing modules.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self._config = config

    def acquire(self, reading: SensorReading) -> Optional[FeatureSet]:
        """
        Process one SensorReading through quality check and feature extraction.

        Steps:
          1. Wrap reading in RawSignal (adds window_id UUID, machine IDs)
          2. Per-axis: check_signal_quality() → DataQualityResult
             - A SENSOR_ERROR never reaches feature extraction
          3. extract_multiaxis_features() on valid axes → MultiAxisReading
          4. Build FeatureSet

        Args:
            reading: Raw sensor data from SensorInterface.read()

        Returns:
            FeatureSet if at least one axis passes quality checks.
            None if every axis is a SENSOR_ERROR (disconnected/dead sensor).

        Raises:
            RuntimeError: if maquina_id has not been resolved yet.
        """
        cfg = self._config

        if cfg.machine.maquina_id is None:
            raise RuntimeError(
                f"maquina_id not resolved for '{cfg.machine.machine_id}'. "
                "Call EdgePipeline.startup() before acquiring data."
            )

        # 1. Wrap: RawSignal adds window_id UUID and machine context
        raw = RawSignal.from_reading(
            reading    = reading,
            machine_id = cfg.machine.machine_id,
            maquina_id = cfg.machine.maquina_id,
            empresa_id = cfg.machine.empresa_id,
        )

        # 2. Quality check per axis (data_quality.py — not modified)
        quality_per_axis: dict[str, DataQualityResult] = {}
        valid_signals:    dict[str, np.ndarray]          = {}

        for axis in reading.available_axes:
            signal = reading.axes[axis]
            qr = check_signal_quality(
                signal        = signal,
                configured_hz = reading.sampling_rate_configured,
                timestamps    = reading.timestamps,
                odr_hz        = cfg.sensor.odr_hz,
            )
            quality_per_axis[axis] = qr

            if qr.is_valid and not qr.is_sensor_error:
                valid_signals[axis] = signal

        # 3. If ALL axes are sensor errors → return None (do not process)
        if not valid_signals:
            return None

        # 4. Determine actual sampling rate from primary axis quality result
        #    Never assumes configured == actual (core Fase 2B requirement)
        primary_qr = quality_per_axis.get(cfg.acquisition.primary_axis)
        sr_actual: Optional[float] = None
        if primary_qr and primary_qr.sampling_rate:
            sr_actual = primary_qr.sampling_rate.actual_hz

        # 5. Operating context — rpm_real is None in Fase 2B (no real RPM source yet)
        context = OperatingContext(
            rpm_nominal = cfg.machine.rpm_nominal,
            rpm_real    = None,   # Real RPM source arrives in a future phase
            rpm_source  = None,
            temperatura = None,   # Populated from sensor data in future phases
            carga       = None,
        )

        # 6. Feature extraction (feature_extraction.py — not modified)
        #    Uses sr_actual (measured) if available, falls back to configured
        multiaxis = extract_multiaxis_features(
            signals                  = valid_signals,
            sensor_id                = reading.sensor_id,
            sampling_rate_configured = reading.sampling_rate_configured,
            sampling_rate_actual     = sr_actual,
            timestamp                = raw.acquired_at.isoformat(),
            context                  = context,
            config                   = cfg.signal,
            include_total            = cfg.acquisition.include_total_axis,
        )

        # 7. Build and return FeatureSet
        return FeatureSet(
            window_id        = raw.window_id,
            maquina_id       = raw.maquina_id,
            empresa_id       = raw.empresa_id,
            acquired_at      = raw.acquired_at,
            multiaxis        = multiaxis,
            quality_per_axis = quality_per_axis,
            primary_axis     = cfg.acquisition.primary_axis,
        )
