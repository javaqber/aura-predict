"""
AuraPredict — RawEventCapture (Fase 2C)
=========================================
Saves the raw vibration signal when an anomaly is detected.

Decision A: files are stored on the local filesystem of the Edge device
(Raspberry Pi SD card), not in Supabase Storage. Supabase Storage upload
is reserved for Fase 2D.

Flow:
  1. Create the raw event directory for this machine.
  2. Save all captured axes as a compressed .npz file (NumPy archive).
  3. Compute SHA-256 checksum of the file.
  4. Call registrar_evento_raw() to record the metadata in the database.
  5. Return the raw_event_id, or None if the capture failed.

The .npz format allows loading individual axes by name:
  data = np.load(path)
  x_signal = data['x']   # original vibration array for axis X

If the database is unavailable, the .npz file is still saved locally.
The metadata registration is retried by the pipeline on the next online cycle.
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

# sys.path for database_v2 lazy imports (inside methods)
_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if TYPE_CHECKING:
    from ..sensors.base_sensor import SensorReading
    from ..pipeline.models import FeatureSet


class RawEventCapture:
    """
    Captures and persists raw vibration signals on anomaly events.

    One instance per Edge device.
    Thread-safety: not required (single-threaded acquisition loop).
    """

    def __init__(self, raw_base_dir: str) -> None:
        self._base = Path(raw_base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def capture(
        self,
        reading:     "SensorReading",
        feature_set: "FeatureSet",
        lectura_id:  Optional[int],
    ) -> Optional[int]:
        """
        Save the raw signal and register the event in the database.

        Args:
            reading:     The original SensorReading from the sensor.
                         Its .axes dict contains the raw numpy arrays.
            feature_set: The FeatureSet produced from this reading.
                         Used for metadata (window_id, maquina_id, acquired_at).
            lectura_id:  The BD row id from registrar_lectura_cnc(),
                         or None if the reading was buffered offline.

        Returns:
            raw_event_id (int) on success, None on failure.
            The .npz file is saved even if the BD registration fails.
        """
        window_id  = feature_set.window_id
        maquina_id = feature_set.maquina_id
        empresa_id = feature_set.empresa_id
        acquired_at = feature_set.acquired_at
        ar = feature_set.anomaly_result

        # ── 1. Save .npz file ──────────────────────────────────────────────────
        try:
            npz_path = self._save_npz(reading, maquina_id, window_id)
        except Exception as exc:
            print(f"[RawCapture] Failed to save .npz: {exc}")
            return None

        # ── 2. Compute checksum ────────────────────────────────────────────────
        try:
            checksum = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        except Exception:
            checksum = None

        # ── 3. Register metadata in BD ─────────────────────────────────────────
        sampling_hz = (
            reading.sampling_rate_actual
            if reading.sampling_rate_actual is not None
            else reading.sampling_rate_configured
        )

        try:
            from database_v2.repositories import registrar_evento_raw
            raw_event_id = registrar_evento_raw(
                maquina_id               = maquina_id,
                empresa_id               = empresa_id,
                event_timestamp          = acquired_at,
                pre_event_s              = 0.0,
                post_event_s             = float(reading.duration_s),
                sampling_rate_hz         = float(sampling_hz),
                total_samples            = reading.n_samples,
                axes_captured            = list(reading.axes.keys()),
                storage_type             = "local",
                file_path                = str(npz_path),
                file_size_bytes          = int(npz_path.stat().st_size),
                file_checksum            = checksum,
                anomaly_score            = ar.anomaly_score if ar else None,
                health_score_at_event    = ar.health_score  if ar else None,
                triggered_by_lectura_id  = lectura_id,
            )
            return raw_event_id
        except Exception as exc:
            # DB offline: file is saved, metadata will be registered later
            print(f"[RawCapture] BD offline, .npz saved at {npz_path}: {exc}")
            return None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _machine_dir(self, maquina_id: int) -> Path:
        d = self._base / str(maquina_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_npz(
        self,
        reading:    "SensorReading",
        maquina_id: int,
        window_id:  str,
    ) -> Path:
        """
        Save all sensor axes to a compressed NumPy archive (.npz).

        File path: {raw_base_dir}/{maquina_id}/{window_id}.npz

        The .npz archive contains one array per axis key (e.g. 'x', 'y', 'z')
        plus a 'timestamps' array if available.

        Load example:
            data = np.load(path)
            x = data['x']   # shape (n_samples,)
        """
        machine_dir = self._machine_dir(maquina_id)

        # Build the array dict
        arrays: dict[str, np.ndarray] = {}
        for axis_name, arr in reading.axes.items():
            arrays[axis_name] = np.asarray(arr, dtype=np.float32)
        if reading.timestamps is not None:
            arrays["timestamps"] = np.asarray(reading.timestamps, dtype=np.float64)

        # np.savez_compressed adds '.npz' only if the path doesn't already end in it
        npz_path = machine_dir / f"{window_id}.npz"
        np.savez_compressed(str(npz_path), **arrays)

        return npz_path
