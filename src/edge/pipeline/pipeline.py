"""
AuraPredict — EdgePipeline
============================
Top-level orchestrator for one Edge device.

Wires together:
  SensorInterface → AcquisitionSession → FeatureSet → BD or LocalBuffer

Three operating scenarios (Fase 2B):
  A. Online:  FeatureSet → registrar_lectura_cnc() → BD
  B. Offline: FeatureSet → LocalBuffer → flush on reconnect
  C. Anomaly: interface prepared, always False in Fase 2B (real detection: Fase 2C)

Dependency injection:
  persist_fn             — injectable for tests; real DB import used if None.
  resolve_machine_id_fn  — injectable for tests; real DB lookup used if None.

This keeps pipeline.py testable without requiring a Supabase connection.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional

from ..sensors.base_sensor import SensorInterface
from ..config.edge_config import EdgeConfig
from ..buffer.local_buffer import LocalBuffer
from .models import FeatureSet, AnomalyTrigger, PlaceholderAnomalyTrigger
from .acquisition import AcquisitionSession

# Type aliases
PersistFn         = Callable[..., Optional[int]]
ResolveMachineIdFn = Callable[[str], Optional[int]]


class EdgePipeline:
    """
    Runs acquisition cycles on one Edge device / machine.

    Production usage:
        config   = EdgeConfig.from_yaml('config/machines/torno_cnc_1.yaml')
        sensor   = MockSensor(config.sensor, MockSensorParams())
        pipeline = EdgePipeline(config, sensor)
        pipeline.startup()             # resolves maquina_id, configures sensor
        feature_set = pipeline.run_once()

    Test usage (no Supabase required):
        pipeline = EdgePipeline(
            config, sensor,
            persist_fn=lambda **kw: 42,          # mock registrar_lectura_cnc
            resolve_machine_id_fn=lambda name: 5, # mock obtener_maquina_id_por_nombre
        )
        pipeline.startup()
        feature_set = pipeline.run_once()
    """

    def __init__(
        self,
        config:                EdgeConfig,
        sensor:                SensorInterface,
        anomaly_trigger:       Optional[AnomalyTrigger]  = None,
        persist_fn:            Optional[PersistFn]        = None,
        resolve_machine_id_fn: Optional[ResolveMachineIdFn] = None,
    ) -> None:
        self._config   = config
        self._sensor   = sensor
        self._trigger  = anomaly_trigger or PlaceholderAnomalyTrigger()
        self._session  = AcquisitionSession(config)
        self._buffer   = LocalBuffer(
            base_dir    = config.buffer.base_dir,
            max_entries = config.buffer.max_entries,
        )
        # Injected callables (None = lazy-import from database_v2 at first use)
        self._persist_fn            = persist_fn
        self._resolve_machine_id_fn = resolve_machine_id_fn

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def startup(self) -> None:
        """
        Initialize the pipeline:
          1. Resolve maquina_id from DB if not set in YAML config.
          2. Configure (initialize) the sensor.

        Raises:
            RuntimeError: if maquina_id cannot be resolved and is not in YAML.
        """
        if self._config.machine.maquina_id is None:
            self._resolve_maquina_id()
        self._sensor.configure()

    def shutdown(self) -> None:
        """Release sensor hardware resources. Safe to call multiple times."""
        self._sensor.close()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_once(self) -> Optional[FeatureSet]:
        """
        Execute one complete acquisition cycle.

        Returns:
            FeatureSet on success, None if sensor error on all axes.
            Never raises on Supabase connectivity issues — uses LocalBuffer.
        """
        # 1. Read from sensor
        try:
            reading = self._sensor.read()
        except Exception as exc:
            print(f"[EdgePipeline] Sensor read error: {exc}")
            return None

        # 2. Process through AcquisitionSession → FeatureSet
        feature_set = self._session.acquire(reading)
        if feature_set is None:
            print("[EdgePipeline] All axes returned SENSOR_ERROR — skipping cycle")
            return None

        # 3. Build DB payload and persist (BD or buffer)
        payload    = feature_set.to_lectura_cnc_payload()
        lectura_id = self._try_persist(payload, feature_set.window_id)
        is_online  = lectura_id is not None

        # 4. If online, flush any pending offline entries
        if is_online and not self._buffer.is_empty():
            self._flush_buffer()

        # 5. Anomaly trigger — always False in Fase 2B
        if self._trigger.should_capture(feature_set):
            self._capture_raw_event(feature_set, lectura_id)

        return feature_set

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def buffer(self) -> LocalBuffer:
        """The LocalBuffer used for offline storage."""
        return self._buffer

    @property
    def config(self) -> EdgeConfig:
        """The EdgeConfig loaded at construction time."""
        return self._config

    # ── Internal ───────────────────────────────────────────────────────────────

    def _try_persist(self, payload: dict, window_id: str) -> Optional[int]:
        """
        Try to insert payload into Supabase.

        On any exception (network, DB, import error) falls back to LocalBuffer.
        The payload is stored with window_id in the buffer filename for traceability.

        Returns:
            int lectura_id on success, None on failure (buffered).
        """
        try:
            fn = self._persist_fn or self._import_persist_fn()
            lectura_id = fn(**payload)
            if lectura_id is None:
                # persist_fn returned None → treat as failure, buffer locally
                self._buffer.push(payload, window_id=window_id)
            return lectura_id
        except Exception as exc:
            print(f"[EdgePipeline] Supabase unavailable — buffering locally: "
                  f"{type(exc).__name__}: {exc}")
            self._buffer.push(payload, window_id=window_id)
            return None

    def _flush_buffer(self) -> None:
        """
        Attempt to flush buffered offline payloads to Supabase.

        Stops at first failure; remaining entries stay in buffer for next cycle.
        """
        try:
            fn = self._persist_fn or self._import_persist_fn()

            def send_fn(payload: dict) -> Optional[int]:
                return fn(**payload)

            n = self._buffer.flush(send_fn=send_fn)
            if n > 0:
                print(f"[EdgePipeline] Flushed {n} offline reading(s) to Supabase")
        except Exception as exc:
            print(f"[EdgePipeline] Buffer flush error: {exc}")

    def _resolve_maquina_id(self) -> None:
        """
        Resolve maquina_id from the DB using the machine's logical name.

        Uses injected resolve_machine_id_fn if provided, otherwise lazy-imports
        from database_v2.repositories.

        Raises:
            RuntimeError: if the machine name is not found in 'maquinas' table.
        """
        machine_id = self._config.machine.machine_id
        try:
            fn = self._resolve_machine_id_fn or self._import_resolve_fn()
            maquina_id = fn(machine_id)
        except Exception as exc:
            raise RuntimeError(
                f"Could not resolve maquina_id for '{machine_id}': {exc}. "
                "Set maquina_id directly in the YAML config for offline startup."
            ) from exc

        if maquina_id is None:
            raise RuntimeError(
                f"Machine '{machine_id}' not found in 'maquinas' table. "
                "Register the machine first, or set maquina_id in the YAML config."
            )
        self._config.machine.maquina_id = maquina_id

    def _capture_raw_event(
        self,
        feature_set: FeatureSet,
        lectura_id:  Optional[int],
    ) -> None:
        """
        Fase 2B: AnomalyTrigger always returns False → this is never called.

        Interface prepared for Fase 2C, which will:
          1. Save raw signal arrays to a local .npy file.
          2. Call registrar_evento_raw(
                 maquina_id=..., empresa_id=..., event_timestamp=...,
                 pre_event_s=..., post_event_s=..., sampling_rate_hz=...,
                 total_samples=..., axes_captured=[...],
                 storage_type='local', file_path=...,
                 triggered_by_lectura_id=lectura_id
             )
        """
        pass  # Fase 2C implementation

    @staticmethod
    def _import_persist_fn() -> PersistFn:
        """Lazy-import registrar_lectura_cnc from database_v2 (production use)."""
        _src = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from database_v2.repositories import registrar_lectura_cnc
        return registrar_lectura_cnc

    @staticmethod
    def _import_resolve_fn() -> ResolveMachineIdFn:
        """Lazy-import obtener_maquina_id_por_nombre from database_v2 (production use)."""
        _src = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from database_v2.repositories import obtener_maquina_id_por_nombre
        return obtener_maquina_id_por_nombre
