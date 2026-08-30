"""
Tests de la Fase 2B — Pipeline Edge completo

Cubre:
  - RawSignal: estructura, UUID, propagación
  - EdgeConfig: carga YAML, reutilización de clases Fase 1
  - FeatureSet: mapeo completo a payload BD
  - AnomalyTrigger: interfaz placeholder
  - LocalBuffer: FIFO, atomicidad, max_entries, flush, recuperación
  - AcquisitionSession: pipeline completo por escenario de sensor
  - EdgePipeline: online, offline, reconexión, errores de sensor

Todos los tests son unitarios — no requieren Supabase.
La dependencia a BD se inyecta vía persist_fn y resolve_machine_id_fn.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, call

import numpy as np
import pytest

# ── Fase 2B imports ────────────────────────────────────────────────────────────
from src.edge.pipeline.models import (
    RawSignal, FeatureSet, AnomalyTrigger, PlaceholderAnomalyTrigger,
)
from src.edge.config.edge_config import (
    EdgeConfig, MachineConfig, AcquisitionConfig, BufferConfig,
)
from src.edge.buffer.local_buffer import LocalBuffer
from src.edge.pipeline.acquisition import AcquisitionSession
from src.edge.pipeline.pipeline import EdgePipeline

# ── Fase 1 imports (existing, not modified) ────────────────────────────────────
from src.edge.sensors.base_sensor import SensorConfig, SensorReading
from src.edge.sensors.mock_sensor import MockSensor, MockSensorParams, SignalMode
from src.edge.signal_processing import SignalConfig, BandDefinition


# ─── FIXTURES AND HELPERS ──────────────────────────────────────────────────────

FS = 3200.0
N  = 3200   # 1 second of data


def make_sensor_config(sensor_id: str = "test_sensor") -> SensorConfig:
    return SensorConfig(
        sensor_id          = sensor_id,
        sensor_type        = "mock",
        sampling_rate_hz   = FS,
        odr_hz             = FS,
        samples_per_window = N,
        axes               = ["x", "y", "z"],
    )


def make_signal_config() -> SignalConfig:
    return SignalConfig(
        fs               = FS,
        bandpass_low_hz  = 10.0,
        bandpass_high_hz = 1000.0,
    )


def make_edge_config(tmp_path: Path, maquina_id: int = 42) -> EdgeConfig:
    """Build an EdgeConfig in memory — no YAML needed for unit tests."""
    return EdgeConfig(
        machine     = MachineConfig(
            machine_id  = "TestCNC_1",
            empresa_id  = 1,
            maquina_id  = maquina_id,
            rpm_nominal = 3000.0,
        ),
        sensor      = make_sensor_config(),
        signal      = make_signal_config(),
        acquisition = AcquisitionConfig(primary_axis="x", include_total_axis=False),
        buffer      = BufferConfig(base_dir=str(tmp_path), max_entries=10),
    )


def make_sensor_reading(
    mode: SignalMode = SignalMode.NORMAL,
    seed: int        = 42,
) -> SensorReading:
    """Generate a SensorReading using MockSensor."""
    config = make_sensor_config()
    params = MockSensorParams(mode=mode, seed=seed)
    sensor = MockSensor(config, params)
    sensor.configure()
    return sensor.read()


def make_feature_set(tmp_path: Path) -> FeatureSet:
    """Run acquisition pipeline and return a FeatureSet."""
    config  = make_edge_config(tmp_path)
    session = AcquisitionSession(config)
    reading = make_sensor_reading()
    fs      = session.acquire(reading)
    assert fs is not None
    return fs


def dummy_payload() -> dict:
    """Minimal payload matching registrar_lectura_cnc signature."""
    return {
        "maquina_id": 42, "empresa_id": 1,
        "resultado": "OK - Sin validar", "nivel_riesgo": "Pendiente",
        "sampling_rate_configured": FS,
    }


# ─── TESTS: RawSignal ──────────────────────────────────────────────────────────

class TestRawSignal:

    def test_window_id_is_valid_uuid4(self):
        reading = make_sensor_reading()
        raw     = RawSignal.from_reading(reading, "TestCNC", 1, 1)
        parsed  = uuid.UUID(raw.window_id)
        assert parsed.version == 4

    def test_window_id_unique_per_call(self):
        reading = make_sensor_reading()
        ids = {RawSignal.from_reading(reading, "T", 1, 1).window_id for _ in range(20)}
        assert len(ids) == 20, "Every window_id must be unique"

    def test_machine_ids_preserved(self):
        reading = make_sensor_reading()
        raw     = RawSignal.from_reading(reading, "Torno_CNC_1", 5, 2)
        assert raw.machine_id  == "Torno_CNC_1"
        assert raw.maquina_id  == 5
        assert raw.empresa_id  == 2

    def test_acquired_at_is_utc_aware(self):
        reading = make_sensor_reading()
        raw     = RawSignal.from_reading(reading, "T", 1, 1)
        assert raw.acquired_at.tzinfo is not None
        assert raw.acquired_at.tzinfo == timezone.utc

    def test_reading_not_copied(self):
        """RawSignal holds a reference — does not deep-copy signal arrays."""
        reading = make_sensor_reading()
        raw     = RawSignal.from_reading(reading, "T", 1, 1)
        assert raw.reading is reading

    def test_machine_id_and_maquina_id_are_separate(self):
        """machine_id (str) and maquina_id (int) must never be confused."""
        reading = make_sensor_reading()
        raw     = RawSignal.from_reading(reading, "Torno_CNC_1", 99, 3)
        assert isinstance(raw.machine_id, str)
        assert isinstance(raw.maquina_id, int)
        assert raw.machine_id != str(raw.maquina_id)


# ─── TESTS: EdgeConfig ─────────────────────────────────────────────────────────

class TestEdgeConfig:

    def test_load_from_yaml(self, tmp_path):
        """EdgeConfig.from_yaml() must load the example YAML without error."""
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "machines", "example_cnc.yaml"
        )
        cfg = EdgeConfig.from_yaml(yaml_path)
        assert cfg.machine.machine_id   == "Torno_CNC_1"
        assert cfg.machine.empresa_id   == 1
        assert cfg.sensor.sampling_rate_hz == 3200.0
        assert cfg.acquisition.primary_axis == "x"

    def test_signal_config_is_fase1_class(self, tmp_path):
        """EdgeConfig.signal must be an instance of signal_processing.SignalConfig."""
        cfg = make_edge_config(tmp_path)
        assert isinstance(cfg.signal, SignalConfig)

    def test_sensor_config_is_fase1_class(self, tmp_path):
        """EdgeConfig.sensor must be an instance of sensors.base_sensor.SensorConfig."""
        cfg = make_edge_config(tmp_path)
        assert isinstance(cfg.sensor, SensorConfig)

    def test_maquina_id_optional(self, tmp_path):
        """maquina_id can be None in config (resolved at startup)."""
        cfg = EdgeConfig(
            machine     = MachineConfig(machine_id="T", empresa_id=1, maquina_id=None),
            sensor      = make_sensor_config(),
            signal      = make_signal_config(),
            acquisition = AcquisitionConfig(),
            buffer      = BufferConfig(base_dir=str(tmp_path)),
        )
        assert cfg.machine.maquina_id is None

    def test_band_names_preserved_from_yaml(self, tmp_path):
        """Band names defined in YAML must appear in SignalConfig.bands."""
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "machines", "example_cnc.yaml"
        )
        cfg        = EdgeConfig.from_yaml(yaml_path)
        band_names = [b.name for b in cfg.signal.bands]
        assert "low"  in band_names
        assert "mid"  in band_names
        assert "high" in band_names

    def test_missing_machine_id_raises(self, tmp_path):
        """Loading a YAML without machine.id must raise KeyError."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("machine:\n  empresa_id: 1\n")
        with pytest.raises(KeyError):
            EdgeConfig.from_yaml(str(bad_yaml))


# ─── TESTS: FeatureSet ─────────────────────────────────────────────────────────

class TestFeatureSet:

    def test_payload_has_required_keys(self, tmp_path):
        fs   = make_feature_set(tmp_path)
        pl   = fs.to_lectura_cnc_payload()
        for key in ("maquina_id", "empresa_id", "resultado", "nivel_riesgo",
                    "sampling_rate_configured", "rms_x", "kurtosis_x",
                    "dominant_freq_hz", "signal_quality_score"):
            assert key in pl, f"Missing key: {key}"

    def test_rms_x_matches_multiaxis(self, tmp_path):
        fs  = make_feature_set(tmp_path)
        pl  = fs.to_lectura_cnc_payload()
        vf  = fs.multiaxis.get_axis("x")
        assert vf is not None
        assert abs(pl["rms_x"] - vf.time.rms) < 1e-10

    def test_anomaly_score_is_none_in_phase_2b(self, tmp_path):
        pl = make_feature_set(tmp_path).to_lectura_cnc_payload()
        assert pl["anomaly_score"] is None

    def test_health_score_is_none_in_phase_2b(self, tmp_path):
        pl = make_feature_set(tmp_path).to_lectura_cnc_payload()
        assert pl["health_score"] is None

    def test_resultado_is_pending_in_phase_2b(self, tmp_path):
        pl = make_feature_set(tmp_path).to_lectura_cnc_payload()
        assert pl["resultado"] == "OK - Sin validar"
        assert pl["nivel_riesgo"] == "Pendiente"

    def test_window_id_not_in_payload(self, tmp_path):
        """window_id must NOT appear in the BD payload — it has no DB column."""
        fs = make_feature_set(tmp_path)
        pl = fs.to_lectura_cnc_payload()
        assert "window_id" not in pl

    def test_sampling_rate_in_payload(self, tmp_path):
        fs = make_feature_set(tmp_path)
        pl = fs.to_lectura_cnc_payload()
        assert pl["sampling_rate_configured"] == FS

    def test_quality_score_positive(self, tmp_path):
        pl = make_feature_set(tmp_path).to_lectura_cnc_payload()
        assert pl["signal_quality_score"] is not None
        assert pl["signal_quality_score"] > 0

    def test_model_version_id_is_none(self, tmp_path):
        pl = make_feature_set(tmp_path).to_lectura_cnc_payload()
        assert pl["model_version_id"] is None

    def test_band_energies_present(self, tmp_path):
        pl = make_feature_set(tmp_path).to_lectura_cnc_payload()
        # At least the primary-axis bands should be present
        assert pl["band_low_energy"] is not None or pl["band_mid_energy"] is not None


# ─── TESTS: AnomalyTrigger ─────────────────────────────────────────────────────

class TestAnomalyTrigger:

    def test_placeholder_always_false(self, tmp_path):
        trigger = PlaceholderAnomalyTrigger()
        fs      = make_feature_set(tmp_path)
        for _ in range(5):
            assert trigger.should_capture(fs) is False

    def test_is_abstract_interface(self):
        """AnomalyTrigger cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AnomalyTrigger()

    def test_interface_has_should_capture(self):
        assert hasattr(AnomalyTrigger, "should_capture")

    def test_custom_trigger_can_return_true(self, tmp_path):
        """A custom trigger that always returns True must be accepted."""
        class AlwaysTrigger(AnomalyTrigger):
            def should_capture(self, fs):
                return True

        trigger = AlwaysTrigger()
        fs      = make_feature_set(tmp_path)
        assert trigger.should_capture(fs) is True


# ─── TESTS: LocalBuffer ────────────────────────────────────────────────────────

class TestLocalBuffer:

    def test_push_creates_file(self, tmp_path):
        buf = LocalBuffer(str(tmp_path), max_entries=10)
        buf.push(dummy_payload(), window_id="w1")
        assert buf.pending_count() == 1

    def test_file_content_is_valid_json(self, tmp_path):
        buf     = LocalBuffer(str(tmp_path), max_entries=10)
        payload = dummy_payload()
        buf.push(payload, window_id="w2")
        files   = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        stored = json.loads(files[0].read_text())
        assert stored["maquina_id"] == 42

    def test_window_id_stored_in_json(self, tmp_path):
        buf = LocalBuffer(str(tmp_path), max_entries=10)
        buf.push(dummy_payload(), window_id="my-uuid-123")
        stored = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
        assert stored["_window_id"] == "my-uuid-123"

    def test_window_id_in_filename(self, tmp_path):
        buf = LocalBuffer(str(tmp_path), max_entries=10)
        buf.push(dummy_payload(), window_id="test-uuid-456")
        filenames = [f.name for f in tmp_path.glob("*.json")]
        assert any("test-uuid-456" in name for name in filenames)

    def test_fifo_order_maintained(self, tmp_path):
        buf     = LocalBuffer(str(tmp_path), max_entries=10)
        results = []
        for i in range(3):
            time.sleep(0.001)   # ensure distinct timestamps
            buf.push({"value": i, **dummy_payload()}, window_id=f"w{i}")
        
        def send_fn(payload):
            results.append(payload["value"])
            return 99
        
        buf.flush(send_fn=send_fn)
        assert results == [0, 1, 2], "Buffer must flush in FIFO order"

    def test_flush_deletes_files_on_success(self, tmp_path):
        buf = LocalBuffer(str(tmp_path), max_entries=10)
        for i in range(3):
            buf.push(dummy_payload(), window_id=f"w{i}")
        n = buf.flush(send_fn=lambda p: 42)
        assert n == 3
        assert buf.is_empty()

    def test_flush_stops_on_failure_none(self, tmp_path):
        buf = LocalBuffer(str(tmp_path), max_entries=10)
        for i in range(3):
            time.sleep(0.001)
            buf.push(dummy_payload(), window_id=f"w{i}")

        call_n = [0]
        def send_fn(payload):
            call_n[0] += 1
            return None if call_n[0] >= 2 else 42

        n = buf.flush(send_fn=send_fn)
        assert n == 1                    # 1 success before failure
        assert buf.pending_count() == 2  # 2 remaining

    def test_flush_stops_on_exception(self, tmp_path):
        buf = LocalBuffer(str(tmp_path), max_entries=10)
        for i in range(3):
            time.sleep(0.001)
            buf.push(dummy_payload(), window_id=f"w{i}")

        call_n = [0]
        def send_fn(payload):
            call_n[0] += 1
            if call_n[0] >= 2:
                raise ConnectionError("Supabase down")
            return 42

        n = buf.flush(send_fn=send_fn)
        assert n == 1
        assert buf.pending_count() == 2

    def test_no_double_send(self, tmp_path):
        """A file is deleted ONLY after send_fn confirms. Re-flush must not re-send."""
        buf = LocalBuffer(str(tmp_path), max_entries=10)
        buf.push(dummy_payload(), window_id="w1")

        sent = []
        def send_fn(payload):
            sent.append(1)
            return 42

        buf.flush(send_fn=send_fn)
        buf.flush(send_fn=send_fn)   # second flush — buffer is empty
        assert len(sent) == 1, "Must not send the same entry twice"

    def test_max_entries_drops_oldest(self, tmp_path):
        buf = LocalBuffer(str(tmp_path), max_entries=3)
        for i in range(5):
            time.sleep(0.001)
            buf.push({"value": i, **dummy_payload()}, window_id=f"w{i}")
        assert buf.pending_count() == 3
        # The 3 remaining must be the newest (values 2, 3, 4)
        remaining = []
        buf.flush(send_fn=lambda p: (remaining.append(p.get("value")), 42)[1])
        assert remaining == [2, 3, 4], f"Expected newest 3, got {remaining}"

    def test_persistence_after_reinit(self, tmp_path):
        """Files survive LocalBuffer reconstruction (Pi restart simulation)."""
        buf1 = LocalBuffer(str(tmp_path), max_entries=10)
        buf1.push(dummy_payload(), window_id="survivor")

        buf2 = LocalBuffer(str(tmp_path), max_entries=10)  # new instance
        assert buf2.pending_count() == 1
        assert any("survivor" in name for name in buf2.list_entries())

    def test_empty_buffer_flush_is_noop(self, tmp_path):
        buf = LocalBuffer(str(tmp_path), max_entries=10)
        n   = buf.flush(send_fn=lambda p: 42)
        assert n == 0

    def test_atomic_write_no_tmp_left(self, tmp_path):
        """After push(), no .tmp files must remain."""
        buf = LocalBuffer(str(tmp_path), max_entries=10)
        buf.push(dummy_payload(), window_id="w1")
        assert list(tmp_path.glob("*.tmp")) == []


# ─── TESTS: AcquisitionSession ─────────────────────────────────────────────────

class TestAcquisitionSession:

    def test_normal_signal_produces_featureset(self, tmp_path):
        cfg     = make_edge_config(tmp_path)
        session = AcquisitionSession(cfg)
        reading = make_sensor_reading(SignalMode.NORMAL)
        fs      = session.acquire(reading)
        assert fs is not None
        assert isinstance(fs, FeatureSet)

    def test_featureset_has_window_id(self, tmp_path):
        fs = make_feature_set(tmp_path)
        assert isinstance(fs.window_id, str)
        assert len(fs.window_id) == 36  # UUID4 format

    def test_sensor_error_all_axes_returns_none(self, tmp_path):
        """SENSOR_FAILURE on all axes must return None, not crash."""
        cfg     = make_edge_config(tmp_path)
        session = AcquisitionSession(cfg)
        reading = make_sensor_reading(SignalMode.SENSOR_FAILURE)
        result  = session.acquire(reading)
        assert result is None

    def test_quality_check_per_axis_independent(self, tmp_path):
        cfg     = make_edge_config(tmp_path)
        session = AcquisitionSession(cfg)
        reading = make_sensor_reading(SignalMode.NORMAL)
        fs      = session.acquire(reading)
        assert fs is not None
        for axis in ["x", "y", "z"]:
            assert axis in fs.quality_per_axis

    def test_sampling_rate_actual_in_featureset(self, tmp_path):
        fs = make_feature_set(tmp_path)
        # MockSensor sets actual == configured
        assert fs.multiaxis.sampling_rate_actual is not None or \
               fs.multiaxis.sampling_rate_configured == FS

    def test_operating_context_rpm_nominal_not_as_real(self, tmp_path):
        """rpm_real must remain None even when rpm_nominal is configured."""
        fs = make_feature_set(tmp_path)
        assert fs.multiaxis.context.rpm_nominal == 3000.0
        assert fs.multiaxis.context.rpm_real    is None

    def test_maquina_id_propagated(self, tmp_path):
        fs = make_feature_set(tmp_path)
        assert fs.maquina_id == 42

    def test_empresa_id_propagated(self, tmp_path):
        fs = make_feature_set(tmp_path)
        assert fs.empresa_id == 1

    def test_raises_without_maquina_id(self, tmp_path):
        """acquire() must raise RuntimeError if maquina_id is None."""
        cfg               = make_edge_config(tmp_path, maquina_id=None)
        cfg.machine.maquina_id = None
        session           = AcquisitionSession(cfg)
        reading           = make_sensor_reading()
        with pytest.raises(RuntimeError, match="maquina_id"):
            session.acquire(reading)

    # ── All MockSensor modes ────────────────────────────────────────────────────

    @pytest.mark.parametrize("mode", [
        SignalMode.NORMAL,
        SignalMode.IMBALANCE,
        SignalMode.MISALIGNMENT,
        SignalMode.LOOSENESS,
        SignalMode.BEARING_DEGRADATION,
    ])
    def test_all_signal_modes_produce_featureset(self, tmp_path, mode):
        cfg     = make_edge_config(tmp_path)
        session = AcquisitionSession(cfg)
        reading = make_sensor_reading(mode)
        fs      = session.acquire(reading)
        assert fs is not None, f"Mode {mode} should produce a FeatureSet"
        assert fs.multiaxis.x is not None

    def test_sensor_failure_mode_returns_none(self, tmp_path):
        cfg     = make_edge_config(tmp_path)
        session = AcquisitionSession(cfg)
        reading = make_sensor_reading(SignalMode.SENSOR_FAILURE)
        assert session.acquire(reading) is None


# ─── TESTS: EdgePipeline ───────────────────────────────────────────────────────

class TestEdgePipeline:

    def _make_pipeline(
        self, tmp_path: Path,
        persist_fn=None,
        fail_after: int = -1,     # -1 = never fail
        sensor_mode: SignalMode = SignalMode.NORMAL,
    ) -> tuple[EdgePipeline, MockSensor]:
        config = make_edge_config(tmp_path)
        sensor_cfg = make_sensor_config()
        params     = MockSensorParams(mode=sensor_mode, seed=42)
        sensor     = MockSensor(sensor_cfg, params)

        call_n = [0]
        if persist_fn is None:
            def persist_fn(**kw):
                call_n[0] += 1
                if fail_after >= 0 and call_n[0] > fail_after:
                    return None
                return call_n[0]

        pipeline = EdgePipeline(
            config               = config,
            sensor               = sensor,
            persist_fn           = persist_fn,
            resolve_machine_id_fn = lambda name: 42,
        )
        return pipeline, sensor

    def test_startup_configures_sensor(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        pipeline.startup()
        assert pipeline.config.machine.maquina_id == 42

    def test_online_run_once_returns_featureset(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        pipeline.startup()
        fs = pipeline.run_once()
        assert fs is not None
        assert isinstance(fs, FeatureSet)

    def test_online_buffer_stays_empty(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path)
        pipeline.startup()
        pipeline.run_once()
        assert pipeline.buffer.is_empty()

    def test_offline_goes_to_buffer(self, tmp_path):
        """When persist_fn returns None, payload goes to LocalBuffer."""
        pipeline, _ = self._make_pipeline(tmp_path, persist_fn=lambda **kw: None)
        pipeline.startup()
        pipeline.run_once()
        assert pipeline.buffer.pending_count() == 1

    def test_supabase_error_does_not_crash(self, tmp_path):
        """Any exception from persist_fn must be caught; buffer used instead."""
        def bad_persist(**kw):
            raise ConnectionError("Supabase is down")

        pipeline, _ = self._make_pipeline(tmp_path, persist_fn=bad_persist)
        pipeline.startup()
        fs = pipeline.run_once()
        assert fs is not None                    # pipeline still returns FeatureSet
        assert pipeline.buffer.pending_count() == 1

    def test_reconnect_flushes_buffer(self, tmp_path):
        """After offline period, first successful persist flushes the buffer."""
        n_fail = [0]
        def flaky_persist(**kw):
            n_fail[0] += 1
            if n_fail[0] <= 2:   # fail first 2 calls
                return None
            return n_fail[0]

        pipeline, _ = self._make_pipeline(tmp_path, persist_fn=flaky_persist)
        pipeline.startup()

        # Cycle 1 & 2: offline → buffer fills
        pipeline.run_once()
        pipeline.run_once()
        assert pipeline.buffer.pending_count() == 2

        # Cycle 3: back online → current reading + flush 2 buffered
        pipeline.run_once()
        assert pipeline.buffer.is_empty()

    def test_sensor_error_returns_none(self, tmp_path):
        pipeline, _ = self._make_pipeline(tmp_path, sensor_mode=SignalMode.SENSOR_FAILURE)
        pipeline.startup()
        fs = pipeline.run_once()
        assert fs is None

    def test_sensor_error_does_not_fill_buffer(self, tmp_path):
        """SENSOR_ERROR must not push to buffer — no reading to persist."""
        pipeline, _ = self._make_pipeline(
            tmp_path,
            persist_fn=lambda **kw: None,
            sensor_mode=SignalMode.SENSOR_FAILURE,
        )
        pipeline.startup()
        pipeline.run_once()
        # SENSOR_ERROR → no FeatureSet → buffer not pushed
        assert pipeline.buffer.is_empty()

    def test_window_id_in_buffer_filename(self, tmp_path):
        """Buffered entries must be identifiable by window_id in the filename."""
        pipeline, _ = self._make_pipeline(tmp_path, persist_fn=lambda **kw: None)
        pipeline.startup()
        pipeline.run_once()
        entries = pipeline.buffer.list_entries()
        assert len(entries) == 1
        # Filename must contain a UUID-like string
        assert len(entries[0]) > 36  # timestamp prefix + uuid

    def test_startup_resolves_maquina_id_via_injected_fn(self, tmp_path):
        """resolve_machine_id_fn is used when maquina_id is None in config."""
        config                 = make_edge_config(tmp_path)
        config.machine.maquina_id = None  # force resolution

        pipeline = EdgePipeline(
            config                = config,
            sensor                = MockSensor(make_sensor_config(), MockSensorParams()),
            persist_fn            = lambda **kw: 42,
            resolve_machine_id_fn = lambda name: 99,
        )
        pipeline.startup()
        assert pipeline.config.machine.maquina_id == 99

    def test_startup_raises_if_machine_not_found(self, tmp_path):
        config                 = make_edge_config(tmp_path)
        config.machine.maquina_id = None

        pipeline = EdgePipeline(
            config                = config,
            sensor                = MockSensor(make_sensor_config(), MockSensorParams()),
            persist_fn            = lambda **kw: 42,
            resolve_machine_id_fn = lambda name: None,  # not found in DB
        )
        with pytest.raises(RuntimeError, match="not found"):
            pipeline.startup()

    def test_placeholder_trigger_never_captures_raw(self, tmp_path):
        """In Fase 2B, AnomalyTrigger always returns False — no RAW capture."""
        captured = []

        class MonitorTrigger(AnomalyTrigger):
            def should_capture(self, fs):
                captured.append(True)
                return False  # same as placeholder

        config  = make_edge_config(tmp_path)
        sensor  = MockSensor(make_sensor_config(), MockSensorParams())
        pipeline = EdgePipeline(
            config               = config,
            sensor               = sensor,
            anomaly_trigger      = MonitorTrigger(),
            persist_fn           = lambda **kw: 42,
            resolve_machine_id_fn = lambda n: 42,
        )
        pipeline.startup()
        pipeline.run_once()
        # Trigger was consulted but returned False
        assert len(captured) == 1
        assert pipeline.buffer.is_empty()


# ─── REGRESSION ────────────────────────────────────────────────────────────────

class TestFase1Regression:
    """Verify that Fase 1 modules are unmodified and still pass their contracts."""

    def test_signal_processing_still_importable(self):
        from src.edge.signal_processing import (
            process_vibration_signal, SignalConfig, VibrationFeatures
        )
        assert callable(process_vibration_signal)

    def test_feature_extraction_still_importable(self):
        from src.edge.feature_extraction import (
            extract_multiaxis_features, MultiAxisReading, OperatingContext
        )
        assert callable(extract_multiaxis_features)

    def test_data_quality_still_importable(self):
        from src.edge.data_quality import check_signal_quality, DataQualityResult
        assert callable(check_signal_quality)

    def test_mock_sensor_still_works(self):
        reading = make_sensor_reading(SignalMode.NORMAL)
        assert reading.n_samples == N
        assert "x" in reading.axes

    def test_pipeline_does_not_reimport_dsp(self, tmp_path):
        """Verify no DSP function is duplicated in pipeline modules."""
        import src.edge.pipeline.models as m
        import src.edge.pipeline.acquisition as a
        # DSP functions must not be defined in these modules
        for name in ("compute_rms", "compute_fft", "bandpass_filter", "detrend_signal"):
            assert not hasattr(m, name), f"pipeline.models must not define {name}"
            assert not hasattr(a, name), f"pipeline.acquisition must not define {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
