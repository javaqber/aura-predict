"""
AuraPredict — EdgePipeline
============================
Top-level orchestrator for one Edge device.

Fase 2B wiring:
  SensorInterface → AcquisitionSession → FeatureSet → BD or LocalBuffer

Fase 2C additions (all behind _baseline_manager — None = Fase 2B behaviour):
  → AnomalyDetector    sets feature_set.anomaly_result before persistence
  → HealthScoreCalculator registers a health_scores row per cycle
  → IsolationForestTrigger / RawEventCapture saves .npz on anomaly
  → Alert log entry when nivel_riesgo is Alto or CRÍTICO

Dependency injection (all default to None → lazy production import):
  persist_fn             mock for registrar_lectura_cnc
  resolve_machine_id_fn  mock for obtener_maquina_id_por_nombre
  baseline_manager       mock or real MachineBaselineManager
  raw_capture            mock or real RawEventCapture

When baseline_manager=None and the pipeline auto-creates one (startup()),
anomaly detection runs silently. Existing Fase 2B tests are unaffected
because the ZScore cold-start detector returns anomaly_score=0.0 and
health_score=None, which do not change any assertion they make.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional, TYPE_CHECKING

from ..sensors.base_sensor import SensorInterface, SensorReading
from ..config.edge_config import EdgeConfig
from ..buffer.local_buffer import LocalBuffer
from .models import FeatureSet, AnomalyTrigger, PlaceholderAnomalyTrigger
from .acquisition import AcquisitionSession

if TYPE_CHECKING:
    from ..anomaly.baseline_manager import MachineBaselineManager
    from ..anomaly.raw_capture import RawEventCapture
    from ..anomaly.anomaly_detector import AnomalyResult

# Type aliases
PersistFn          = Callable[..., Optional[int]]
ResolveMachineIdFn = Callable[[str], Optional[int]]


class EdgePipeline:
    """
    Runs acquisition cycles on one Edge device / machine.

    Production (Fase 2C):
        config   = EdgeConfig.from_yaml('config/machines/torno_cnc_1.yaml')
        sensor   = MockSensor(config.sensor, MockSensorParams())
        pipeline = EdgePipeline(config, sensor)
        pipeline.startup()          # resolves maquina_id, inits baseline manager
        feature_set = pipeline.run_once()

    Test (no Supabase, no filesystem side-effects):
        pipeline = EdgePipeline(
            config, sensor,
            persist_fn=lambda **kw: 42,
            resolve_machine_id_fn=lambda name: 5,
            baseline_manager=<mock_or_real>,
        )
        pipeline.startup()
        feature_set = pipeline.run_once()
    """

    def __init__(
        self,
        config:                EdgeConfig,
        sensor:                SensorInterface,
        anomaly_trigger:       Optional[AnomalyTrigger]        = None,
        persist_fn:            Optional[PersistFn]             = None,
        resolve_machine_id_fn: Optional[ResolveMachineIdFn]   = None,
        # ── Fase 2C injectable components ─────────────────────────────────────
        baseline_manager:      Optional["MachineBaselineManager"] = None,
        raw_capture:           Optional["RawEventCapture"]        = None,
    ) -> None:
        self._config   = config
        self._sensor   = sensor
        self._trigger  = anomaly_trigger or PlaceholderAnomalyTrigger()
        self._session  = AcquisitionSession(config)
        self._buffer   = LocalBuffer(
            base_dir    = config.buffer.base_dir,
            max_entries = config.buffer.max_entries,
        )
        self._persist_fn            = persist_fn
        self._resolve_machine_id_fn = resolve_machine_id_fn
        # Fase 2C — None until startup() initialises them (or injected by tests)
        self._baseline_manager: Optional["MachineBaselineManager"] = baseline_manager
        self._raw_capture:      Optional["RawEventCapture"]        = raw_capture

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def startup(self) -> None:
        """
        Initialise the pipeline:
          1. Resolve maquina_id from DB if not set in YAML.
          2. Create and load MachineBaselineManager (Fase 2C).
          3. Create RawEventCapture (Fase 2C).
          4. Configure the sensor.
        """
        if self._config.machine.maquina_id is None:
            self._resolve_maquina_id()

        # Fase 2C: auto-create baseline_manager if not injected
        if self._baseline_manager is None:
            self._baseline_manager = self._create_baseline_manager()
            self._baseline_manager.load_from_db()   # silent fail when offline

        # Fase 2C: auto-create raw_capture if not injected
        if self._raw_capture is None:
            from ..anomaly.raw_capture import RawEventCapture
            self._raw_capture = RawEventCapture(self._config.anomaly.raw_base_dir)

        self._sensor.configure()

    def shutdown(self) -> None:
        """Release sensor hardware resources. Safe to call multiple times."""
        self._sensor.close()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_once(self) -> Optional[FeatureSet]:
        """
        Execute one complete acquisition cycle.

        Returns FeatureSet on success, None if all axes are SENSOR_ERROR.
        Never raises on Supabase connectivity issues — uses LocalBuffer.

        Cycle order (Fase 2C):
          1. sensor.read()
          2. AcquisitionSession.acquire() → FeatureSet
          3. _run_anomaly_detection()     → sets feature_set.anomaly_result
          4. to_lectura_cnc_payload()     → includes real anomaly/health values
          5. _try_persist()              → BD or LocalBuffer
          6. _register_health_score()    → health_scores table (if online)
          7. _flush_buffer()             → drain offline backlog (if online)
          8. _maybe_send_alert()         → alert_log (if nivel_riesgo high)
          9. trigger check + _capture_raw_event()
        """
        # 1. Read sensor
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

        # 3. Anomaly detection (Fase 2C — no-op if baseline_manager is None)
        self._run_anomaly_detection(feature_set)

        # 4. Build DB payload (uses anomaly_result if set, Fase 2B fallback if not)
        payload = feature_set.to_lectura_cnc_payload()

        # 5. Persist: BD first, LocalBuffer on failure
        lectura_id = self._try_persist(payload, feature_set.window_id)
        is_online  = lectura_id is not None

        # 6. Register health score (only when online and score is available)
        ar = feature_set.anomaly_result
        if is_online and ar is not None and ar.health_score is not None:
            self._register_health_score(ar, lectura_id)

        # 7. Flush buffered offline readings if we are back online
        if is_online and not self._buffer.is_empty():
            self._flush_buffer()

        # 8. Alert if nivel_riesgo is high (Fase 2C)
        if ar is not None:
            self._maybe_send_alert(ar)

        # 9. RAW capture if trigger fires (Fase 2C — IsolationForestTrigger
        #    checks anomaly_result internally; PlaceholderAnomalyTrigger always False)
        if self._trigger.should_capture(feature_set):
            self._capture_raw_event(feature_set, reading, lectura_id)

        return feature_set

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def buffer(self) -> LocalBuffer:
        return self._buffer

    @property
    def config(self) -> EdgeConfig:
        return self._config

    # ── Fase 2C: anomaly detection ────────────────────────────────────────────

    def _run_anomaly_detection(self, feature_set: FeatureSet) -> None:
        """
        Run the active anomaly detector and set feature_set.anomaly_result.

        Also feeds the result back to the baseline manager so 'normal'
        readings accumulate toward the Isolation Forest training threshold.
        Does nothing if baseline_manager is None (Fase 2B behaviour).
        """
        if self._baseline_manager is None:
            return

        vector = self._baseline_manager.extract_feature_vector(feature_set)
        if vector is None:
            return   # primary axis unavailable — skip this cycle

        # Signal quality from the primary axis quality check
        pq = feature_set.quality_per_axis.get(feature_set.primary_axis)
        signal_quality = (
            pq.quality_score
            if pq is not None and not pq.is_sensor_error
            else 0.3
        )

        detector = self._baseline_manager.get_active_detector()
        ar = detector.analyze(
            feature_vector = vector,
            baseline_stats = self._baseline_manager.baseline_stats,
            signal_quality = signal_quality,
        )
        feature_set.anomaly_result = ar

        # Feed back to baseline — only accumulate genuinely normal readings
        is_normal = ar.resultado in ("OK - Sano", "OK - Aprendiendo")
        self._baseline_manager.record_reading(vector, is_normal=is_normal)

    # ── Fase 2C: health score ─────────────────────────────────────────────────

    def _register_health_score(
        self,
        ar:         "AnomalyResult",
        lectura_id: int,
    ) -> None:
        """Insert a health_scores row with trend and slope. Silent on failure."""
        try:
            _src = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "../../.."))
            if _src not in sys.path:
                sys.path.insert(0, _src)
            from database_v2.repositories import (
                registrar_health_score,
                obtener_historial_health,
            )
            from ..anomaly.health_score import HealthScoreCalculator

            maquina_id = self._config.machine.maquina_id
            empresa_id = self._config.machine.empresa_id

            recent         = obtener_historial_health(maquina_id, dias=7)
            trend, slope   = HealthScoreCalculator().compute(recent, ar.health_score)

            registrar_health_score(
                maquina_id = maquina_id,
                empresa_id = empresa_id,
                score      = ar.health_score,
                trend      = trend,
                slope      = slope,
                lectura_id = lectura_id,
            )
        except Exception as exc:
            print(f"[EdgePipeline] Health score registration skipped: {exc}")

    # ── Fase 2C: alerts ───────────────────────────────────────────────────────

    def _maybe_send_alert(self, ar: "AnomalyResult") -> None:
        """
        Register an alert entry in alert_log when nivel_riesgo is high.

        Checks cooldown via puede_enviar_alerta() before inserting.
        In MVP: alert is logged with enviado=False (no SMTP in Fase 2C).
        Silent on failure (DB offline, etc.).
        """
        if ar.is_cold_start:
            return
        if ar.nivel_riesgo not in ("Alto", "CRÍTICO"):
            return

        maquina_id = self._config.machine.maquina_id
        empresa_id = self._config.machine.empresa_id
        cooldown   = self._config.anomaly.alert_cooldown_hours
        emails     = self._config.anomaly.alert_emails
        tipo       = "CRITICAL" if ar.nivel_riesgo == "CRÍTICO" else "WARNING"

        try:
            _src = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "../../.."))
            if _src not in sys.path:
                sys.path.insert(0, _src)
            from database_v2.repositories import puede_enviar_alerta, registrar_alerta

            if not puede_enviar_alerta(maquina_id, cooldown_hours=cooldown):
                return  # still in cooldown

            destinatario = emails[0] if emails else ""
            asunto = (
                f"AuraPredict {tipo}: {self._config.machine.machine_id} "
                f"— health={ar.health_score}"
            )
            registrar_alerta(
                maquina_id   = maquina_id,
                empresa_id   = empresa_id,
                tipo_alerta  = tipo,
                destinatario = destinatario,
                asunto       = asunto,
                enviado      = False,   # MVP: log only, no SMTP
                error_msg    = "Email sending not configured in Fase 2C",
            )
        except Exception as exc:
            print(f"[EdgePipeline] Alert registration skipped: {exc}")

    # ── Fase 2C: RAW capture ──────────────────────────────────────────────────

    def _capture_raw_event(
        self,
        feature_set: FeatureSet,
        reading:     SensorReading,
        lectura_id:  Optional[int],
    ) -> None:
        """
        Save the raw signal as a .npz file and register it in raw_event_windows.

        Decision B (approved): SensorReading is passed explicitly from run_once()
        rather than stored in FeatureSet. The reading is available in the pipeline
        scope at the time the trigger fires.
        """
        if self._raw_capture is None:
            return
        try:
            self._raw_capture.capture(reading, feature_set, lectura_id)
        except Exception as exc:
            print(f"[EdgePipeline] RAW capture failed: {exc}")

    # ── Persist / flush / resolve ──────────────────────────────────────────────

    def _try_persist(self, payload: dict, window_id: str) -> Optional[int]:
        """Try BD; fall back to LocalBuffer on any failure."""
        try:
            fn = self._persist_fn or self._import_persist_fn()
            lectura_id = fn(**payload)
            if lectura_id is None:
                self._buffer.push(payload, window_id=window_id)
            return lectura_id
        except Exception as exc:
            print(f"[EdgePipeline] Supabase unavailable — buffering: "
                  f"{type(exc).__name__}: {exc}")
            self._buffer.push(payload, window_id=window_id)
            return None

    def _flush_buffer(self) -> None:
        """Drain buffered offline payloads. Stops at first failure."""
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

    # ── Fase 2C: factory ──────────────────────────────────────────────────────

    def _create_baseline_manager(self) -> "MachineBaselineManager":
        """Instantiate MachineBaselineManager from AnomalyConfig."""
        from ..anomaly.baseline_manager import MachineBaselineManager
        from ..anomaly.model_store import ModelStore
        ac = self._config.anomaly
        return MachineBaselineManager(
            maquina_id           = self._config.machine.maquina_id,
            empresa_id           = self._config.machine.empresa_id,
            model_store          = ModelStore(ac.model_base_dir),
            primary_axis         = self._config.acquisition.primary_axis,
            baseline_min_samples = ac.baseline_min_samples,
            update_every_n       = ac.update_every_n,
            baseline_window_n    = ac.baseline_window_n,
            z_threshold          = ac.z_score_threshold,
            if_n_estimators      = ac.if_n_estimators,
            if_contamination     = ac.if_contamination,
        )

    # ── Static lazy imports ───────────────────────────────────────────────────

    @staticmethod
    def _import_persist_fn() -> PersistFn:
        _src = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from database_v2.repositories import registrar_lectura_cnc
        return registrar_lectura_cnc

    @staticmethod
    def _import_resolve_fn() -> ResolveMachineIdFn:
        _src = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from database_v2.repositories import obtener_maquina_id_por_nombre
        return obtener_maquina_id_por_nombre
