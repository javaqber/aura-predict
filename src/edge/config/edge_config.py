"""
AuraPredict — Edge Configuration
==================================
Loads per-machine YAML and constructs existing Fase 1 configuration classes.

Reuses WITHOUT duplication:
  SensorConfig  from sensors/base_sensor.py  (unchanged)
  SignalConfig   from signal_processing.py   (unchanged)
  BandDefinition from signal_processing.py   (unchanged)

Adds (new in Fase 2B):
  MachineConfig    — machine identity and BD integer IDs
  AcquisitionConfig — timing, primary axis selection
  BufferConfig     — LocalBuffer offline settings
  EdgeConfig       — top-level container, loads from YAML
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml

# ── Reuse existing Fase 1 classes — not duplicated ────────────────────────────
from ..sensors.base_sensor import SensorConfig
from ..signal_processing import SignalConfig, BandDefinition


# ─── MACHINE CONFIG ────────────────────────────────────────────────────────────

@dataclass
class MachineConfig:
    """
    Machine identity and database references.

    Strict separation of identifiers:
      machine_id  : str          — logical name from YAML (human-readable)
      maquina_id  : Optional[int] — integer PK in 'maquinas' table.
                                   Set from YAML to allow offline startup,
                                   or resolved at runtime via DB lookup.
      empresa_id  : int          — integer PK in 'empresas' table.

    machine_id and maquina_id are NEVER used interchangeably. The pipeline
    resolves maquina_id from DB at startup if not present in YAML.
    """
    machine_id:  str
    empresa_id:  int
    tipo:        str            = "torno_cnc"
    rpm_nominal: Optional[float] = None
    maquina_id:  Optional[int]  = None  # Optional YAML override; resolved at startup


# ─── ACQUISITION CONFIG ────────────────────────────────────────────────────────

@dataclass
class AcquisitionConfig:
    """
    Timing and axis selection for one acquisition cycle.

    primary_axis  : axis used for frequency-domain features in the DB payload.
                    (dominant_freq_hz, spectral_energy, band_*_energy map to
                     this axis in lecturas_cnc_v2)
    """
    primary_axis:        str   = "x"
    include_total_axis:  bool  = False   # compute sqrt(x²+y²+z²) axis
    interval_normal_s:   float = 120.0   # seconds between readings (normal mode)
    interval_anomaly_s:  float = 30.0    # seconds between readings (anomaly mode)


# ─── BUFFER CONFIG ─────────────────────────────────────────────────────────────

@dataclass
class BufferConfig:
    """LocalBuffer offline storage configuration."""
    base_dir:    str = "/tmp/aurapredict/buffer"
    max_entries: int = 500


# ─── EDGE CONFIG ───────────────────────────────────────────────────────────────

@dataclass
class EdgeConfig:
    """
    Top-level configuration for one Edge device / machine pair.

    Uses existing SensorConfig and SignalConfig from Fase 1 — not duplicated.
    Loaded from a per-machine YAML file via EdgeConfig.from_yaml().
    """
    machine:     MachineConfig
    sensor:      SensorConfig       # from sensors/base_sensor.py
    signal:      SignalConfig       # from signal_processing.py
    acquisition: AcquisitionConfig
    buffer:      BufferConfig

    @classmethod
    def from_yaml(cls, path: str) -> "EdgeConfig":
        """
        Load configuration from a YAML file and build all sub-configs.

        Raises:
            FileNotFoundError: if the YAML file does not exist.
            KeyError:          if a required field ('machine.id', 'machine.empresa_id')
                               is missing.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # ── MachineConfig ──────────────────────────────────────────────────────
        m = data.get("machine", {})
        machine = MachineConfig(
            machine_id  = m["id"],
            empresa_id  = int(m["empresa_id"]),
            tipo        = m.get("tipo", "torno_cnc"),
            rpm_nominal = m.get("rpm_nominal"),
            maquina_id  = int(m["maquina_id"]) if m.get("maquina_id") is not None else None,
        )

        # ── SensorConfig (existing Fase 1 class — not modified) ───────────────
        s = data.get("sensor", {})
        sensor = SensorConfig(
            sensor_id          = m.get("id", "edge_sensor"),
            sensor_type        = s.get("type", "mock"),
            sampling_rate_hz   = float(s.get("sampling_rate_hz", 3200.0)),
            odr_hz             = float(s["odr_hz"]) if s.get("odr_hz") is not None else None,
            samples_per_window = int(s.get("samples_per_window", 3200)),
            axes               = list(s.get("axes", ["x", "y", "z"])),
            i2c_address        = s.get("i2c_address"),
        )

        # ── SignalConfig (existing Fase 1 class — not modified) ───────────────
        dsp = data.get("preprocessing", {})
        bands_raw = dsp.get("bands", [
            {"name": "low",  "low_hz": 10.0,  "high_hz": 100.0},
            {"name": "mid",  "low_hz": 100.0, "high_hz": 500.0},
            {"name": "high", "low_hz": 500.0, "high_hz": 1600.0},
        ])
        bands = [
            BandDefinition(b["name"], float(b["low_hz"]), float(b["high_hz"]))
            for b in bands_raw
        ]
        signal = SignalConfig(
            fs               = float(s.get("sampling_rate_hz", 3200.0)),
            window_type      = dsp.get("window_type", "hann"),
            filter_order     = int(dsp.get("filter_order", 4)),
            bandpass_low_hz  = float(dsp.get("bandpass_low_hz", 10.0)),
            bandpass_high_hz = float(dsp.get("bandpass_high_hz", 1000.0)),
            detrend_type     = dsp.get("detrend", "linear"),
            bands            = bands,
        )

        # ── AcquisitionConfig ─────────────────────────────────────────────────
        acq = data.get("acquisition", {})
        acquisition = AcquisitionConfig(
            primary_axis       = acq.get("primary_axis", "x"),
            include_total_axis = bool(acq.get("include_total_axis", False)),
            interval_normal_s  = float(acq.get("interval_normal_s", 120.0)),
            interval_anomaly_s = float(acq.get("interval_anomaly_s", 30.0)),
        )

        # ── BufferConfig ──────────────────────────────────────────────────────
        buf = data.get("buffer", {})
        buffer = BufferConfig(
            base_dir    = buf.get("base_dir", "/tmp/aurapredict/buffer"),
            max_entries = int(buf.get("max_entries", 500)),
        )

        return cls(
            machine     = machine,
            sensor      = sensor,
            signal      = signal,
            acquisition = acquisition,
            buffer      = buffer,
        )
