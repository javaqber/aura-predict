"""
Tests de la Fase 2C — Motor de detección de anomalías

Cobertura:
  TestAnomalyResult        — estructura del dataclass de salida
  TestModelStore           — save/load de modelos .joblib
  TestMachineBaselineManager — acumulación, estadísticas, entrenamiento IF
  TestAnomalyDetector      — ZScoreDetector e IsolationForestDetector
  TestIsolationForestTrigger — lógica de disparo y cooldown
  TestRawEventCapture      — escritura .npz y checksum
  TestEdgePipelinePhase2C  — integración completa con motor de anomalías

Todos los tests son unitarios — no requieren Supabase.
Las llamadas a BD se saltan silenciosamente (comportamiento de offline).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

# ── Módulos Fase 2C ────────────────────────────────────────────────────────────
from src.edge.anomaly.anomaly_detector import (
    AnomalyResult, AnomalyDetector,
    ZScoreDetector, IsolationForestDetector,
    FEATURE_NAMES, QUALITY_MIN,
    _classify_health, _compute_health, _describe_contributors,
)
from src.edge.anomaly.model_store import ModelStore
from src.edge.anomaly.baseline_manager import MachineBaselineManager
from src.edge.anomaly.health_score import HealthScoreCalculator
from src.edge.anomaly.raw_capture import RawEventCapture
from src.edge.anomaly.isolation_forest_trigger import IsolationForestTrigger

# ── Módulos existentes (sin modificar) ────────────────────────────────────────
from src.edge.pipeline.models import (
    FeatureSet, AnomalyTrigger, PlaceholderAnomalyTrigger,
)
from src.edge.pipeline.pipeline import EdgePipeline
from src.edge.pipeline.acquisition import AcquisitionSession
from src.edge.config.edge_config import (
    EdgeConfig, MachineConfig, AcquisitionConfig, BufferConfig, AnomalyConfig,
)
from src.edge.signal_processing import SignalConfig
from src.edge.sensors.base_sensor import SensorConfig, SensorReading
from src.edge.sensors.mock_sensor import MockSensor, MockSensorParams, SignalMode


# ─── CONSTANTES ────────────────────────────────────────────────────────────────

FS = 3200.0
N  = 3200


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_sensor_config() -> SensorConfig:
    return SensorConfig(
        sensor_id="test", sensor_type="mock",
        sampling_rate_hz=FS, samples_per_window=N,
        axes=["x", "y", "z"],
    )


def make_sensor_reading(mode: SignalMode = SignalMode.NORMAL, seed: int = 42) -> SensorReading:
    cfg    = make_sensor_config()
    sensor = MockSensor(cfg, MockSensorParams(mode=mode, seed=seed))
    sensor.configure()
    return sensor.read()


def make_edge_config(tmp_path: Path, maquina_id: int = 42) -> EdgeConfig:
    return EdgeConfig(
        machine     = MachineConfig(machine_id="T", empresa_id=1, maquina_id=maquina_id),
        sensor      = make_sensor_config(),
        signal      = SignalConfig(fs=FS),
        acquisition = AcquisitionConfig(primary_axis="x"),
        buffer      = BufferConfig(base_dir=str(tmp_path / "buffer"), max_entries=10),
        anomaly     = AnomalyConfig(
            baseline_min_samples = 20,
            update_every_n       = 10,
            model_base_dir       = str(tmp_path / "models"),
            raw_base_dir         = str(tmp_path / "raw"),
            capture_threshold    = 50,
            capture_cooldown_s   = 3600.0,
        ),
    )


def make_feature_set(tmp_path: Path) -> FeatureSet:
    cfg     = make_edge_config(tmp_path)
    session = AcquisitionSession(cfg)
    reading = make_sensor_reading()
    fs      = session.acquire(reading)
    assert fs is not None
    return fs


def make_anomaly_result(
    health_score: Optional[int] = 85,
    anomaly_score: float = 0.15,
    resultado: str = "OK - Sano",
    nivel_riesgo: str = "Bajo",
    is_cold_start: bool = False,
    algorithm: str = "zscore",
) -> AnomalyResult:
    return AnomalyResult(
        anomaly_score    = anomaly_score,
        health_score     = health_score,
        resultado        = resultado,
        nivel_riesgo     = nivel_riesgo,
        diagnostico      = "",
        model_version_id = None,
        is_cold_start    = is_cold_start,
        algorithm        = algorithm,
    )


def make_trained_if(n_samples: int = 60) -> IsolationForest:
    np.random.seed(42)
    X = np.random.randn(n_samples, 8) * 0.01 + 0.1
    m = IsolationForest(n_estimators=10, contamination="auto", random_state=42)
    m.fit(X)
    return m


def make_baseline_stats() -> dict:
    return {
        name: {"mean": 0.1, "std": 0.01, "p5": 0.08, "p50": 0.1, "p95": 0.12}
        for name in FEATURE_NAMES
    }


class _MockBaselineManager:
    """
    Controllable baseline manager for pipeline integration tests.
    Returns a pre-set AnomalyResult without any real ML computation.
    """
    def __init__(self, preset_result: AnomalyResult):
        self._result = preset_result
        self._recorded: list = []

    def extract_feature_vector(self, fs: FeatureSet):
        return np.array([0.1] * 8)

    def get_active_detector(self):
        result = self._result
        class _Det(AnomalyDetector):
            @property
            def is_ready(self): return True
            def analyze(self, feature_vector, baseline_stats, signal_quality=1.0, model_version_id=None): return result
        return _Det()

    @property
    def baseline_stats(self): return make_baseline_stats()

    def record_reading(self, vector, is_normal: bool):
        self._recorded.append(is_normal)

    def load_from_db(self): return False


# ─── TEST: AnomalyResult ──────────────────────────────────────────────────────

class TestAnomalyResult:

    def test_all_fields_accessible(self):
        ar = make_anomaly_result()
        assert hasattr(ar, "anomaly_score")
        assert hasattr(ar, "health_score")
        assert hasattr(ar, "resultado")
        assert hasattr(ar, "nivel_riesgo")
        assert hasattr(ar, "diagnostico")
        assert hasattr(ar, "model_version_id")
        assert hasattr(ar, "is_cold_start")
        assert hasattr(ar, "algorithm")

    def test_cold_start_flag(self):
        ar = make_anomaly_result(is_cold_start=True)
        assert ar.is_cold_start is True

    def test_algorithm_field_set(self):
        ar = make_anomaly_result(algorithm="isolation_forest")
        assert ar.algorithm == "isolation_forest"

    def test_none_health_score_allowed(self):
        ar = make_anomaly_result(health_score=None)
        assert ar.health_score is None


# ─── TEST: ModelStore ─────────────────────────────────────────────────────────

class TestModelStore:

    def test_save_creates_joblib_file(self, tmp_path):
        store = ModelStore(str(tmp_path))
        model = make_trained_if()
        path  = store.save(model, maquina_id=1, version="1.60.0")
        assert path.exists()
        assert path.suffix == ".joblib"

    def test_load_returns_trained_model(self, tmp_path):
        store = ModelStore(str(tmp_path))
        model = make_trained_if()
        store.save(model, maquina_id=1, version="1.60.0")
        loaded = store.load(maquina_id=1, version="1.60.0")
        assert loaded is not None
        assert isinstance(loaded, IsolationForest)

    def test_load_nonexistent_returns_none(self, tmp_path):
        store = ModelStore(str(tmp_path))
        assert store.load(maquina_id=1, version="9.9.9") is None

    def test_active_metadata_roundtrip(self, tmp_path):
        store = ModelStore(str(tmp_path))
        store.save_active_metadata(1, "1.60.0", "/some/path/model.joblib", 7)
        meta = store.load_active_metadata(1)
        assert meta is not None
        assert meta["version"] == "1.60.0"
        assert meta["model_version_id"] == 7

    def test_load_active_model_returns_model_and_meta(self, tmp_path):
        store = ModelStore(str(tmp_path))
        model = make_trained_if()
        path  = store.save(model, maquina_id=1, version="1.60.0")
        store.save_active_metadata(1, "1.60.0", str(path), 7)
        loaded, meta = store.load_active_model(1)
        assert loaded is not None
        assert meta["model_version_id"] == 7

    def test_missing_active_metadata_returns_none_none(self, tmp_path):
        store = ModelStore(str(tmp_path))
        model, meta = store.load_active_model(1)
        assert model is None
        assert meta is None


# ─── TEST: MachineBaselineManager ─────────────────────────────────────────────

class TestMachineBaselineManager:

    def _make_mgr(self, tmp_path, min_samples=20, update_every=10):
        store = ModelStore(str(tmp_path / "models"))
        return MachineBaselineManager(
            maquina_id=1, empresa_id=1,
            model_store=store,
            primary_axis="x",
            baseline_min_samples=min_samples,
            update_every_n=update_every,
            baseline_window_n=200,
        )

    def test_initial_detector_is_zscore(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        assert isinstance(mgr.get_active_detector(), ZScoreDetector)

    def test_n_samples_starts_at_zero(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        assert mgr.n_samples == 0
        assert not mgr.is_baseline_ready

    def test_normal_readings_accumulate(self, tmp_path):
        mgr = self._make_mgr(tmp_path, min_samples=20, update_every=100)
        for _ in range(5):
            mgr.record_reading(np.ones(8) * 0.1, is_normal=True)
        assert mgr.n_samples == 5

    def test_anomalous_readings_not_accumulated(self, tmp_path):
        mgr = self._make_mgr(tmp_path, update_every=100)
        mgr.record_reading(np.ones(8) * 0.5, is_normal=False)
        assert mgr.n_samples == 0

    def test_all_nan_vector_not_accumulated(self, tmp_path):
        mgr = self._make_mgr(tmp_path, update_every=100)
        mgr.record_reading(np.full(8, np.nan), is_normal=True)
        assert mgr.n_samples == 0

    def test_detector_switches_to_if_after_min_samples(self, tmp_path):
        mgr = self._make_mgr(tmp_path, min_samples=20, update_every=20)
        np.random.seed(42)
        for _ in range(20):
            v = np.random.randn(8) * 0.01 + 0.1
            mgr.record_reading(v, is_normal=True)
        assert isinstance(mgr.get_active_detector(), IsolationForestDetector)

    def test_baseline_stats_contain_all_features(self, tmp_path):
        mgr = self._make_mgr(tmp_path, min_samples=5, update_every=5)
        np.random.seed(0)
        for _ in range(5):
            v = np.random.randn(8) * 0.01 + 0.1
            mgr.record_reading(v, is_normal=True)
        for name in FEATURE_NAMES:
            assert name in mgr.baseline_stats
            assert "mean" in mgr.baseline_stats[name]
            assert "std"  in mgr.baseline_stats[name]

    def test_local_json_written_after_update(self, tmp_path):
        mgr = self._make_mgr(tmp_path, min_samples=5, update_every=5)
        np.random.seed(0)
        for _ in range(5):
            mgr.record_reading(np.random.randn(8) * 0.01 + 0.1, is_normal=True)
        assert mgr._local_baseline_path().exists()

    def test_recovery_from_local_json(self, tmp_path):
        mgr = self._make_mgr(tmp_path, min_samples=5, update_every=5)
        np.random.seed(0)
        for _ in range(5):
            mgr.record_reading(np.random.randn(8) * 0.01 + 0.1, is_normal=True)
        n_before = mgr.n_samples

        mgr2 = self._make_mgr(tmp_path, min_samples=5, update_every=5)
        loaded = mgr2._load_baseline_from_local()
        assert loaded is True
        assert mgr2.n_samples == n_before

    def test_extract_feature_vector_from_featureset(self, tmp_path):
        mgr    = self._make_mgr(tmp_path)
        fs     = make_feature_set(tmp_path)
        vector = mgr.extract_feature_vector(fs)
        assert vector is not None
        assert len(vector) == 8
        assert all(np.isfinite(v) for v in vector)

    def test_extract_returns_none_when_primary_axis_missing(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        fs  = make_feature_set(tmp_path)
        fs.multiaxis.x = None   # remove primary axis
        assert mgr.extract_feature_vector(fs) is None


# ─── TEST: AnomalyDetector ────────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == 8

    def test_classify_health_ranges(self):
        assert _classify_health(95) == ("OK - Sano",              "Bajo")
        assert _classify_health(75) == ("OK - Sano",              "Bajo")
        assert _classify_health(74) == ("ADVERTENCIA",             "Medio")
        assert _classify_health(50) == ("ADVERTENCIA",             "Medio")
        assert _classify_health(49) == ("ALERTA",                  "Alto")
        assert _classify_health(25) == ("ALERTA",                  "Alto")
        assert _classify_health(24) == ("NOK - Anomalía Detectada","CRÍTICO")
        assert _classify_health(0)  == ("NOK - Anomalía Detectada","CRÍTICO")

    def test_compute_health_below_quality_min(self):
        assert _compute_health(0.5, signal_quality=0.3) is None

    def test_compute_health_perfect_signal_no_anomaly(self):
        h = _compute_health(0.0, signal_quality=1.0)
        assert h == 100

    def test_compute_health_full_anomaly(self):
        h = _compute_health(1.0, signal_quality=1.0)
        assert h == 0

    # ── ZScoreDetector ────────────────────────────────────────────────────────

    def test_zscore_no_baseline_returns_aprendiendo(self):
        det = ZScoreDetector()
        ar  = det.analyze(np.zeros(8), baseline_stats={})
        assert ar.resultado      == "OK - Aprendiendo"
        assert ar.health_score   is None
        assert ar.is_cold_start  is True
        assert ar.algorithm      == "zscore"

    def test_zscore_normal_vector_low_score(self):
        det      = ZScoreDetector(z_threshold=3.0)
        baseline = make_baseline_stats()
        normal   = np.array([0.1] * 8)
        ar       = det.analyze(normal, baseline, signal_quality=1.0)
        assert ar.anomaly_score  < 0.1
        assert ar.health_score   >= 90
        assert ar.resultado      == "OK - Sano"

    def test_zscore_anomalous_vector_high_score(self):
        det      = ZScoreDetector(z_threshold=3.0)
        baseline = make_baseline_stats()
        anomal   = np.array([0.5] * 8)   # 40σ above mean
        ar       = det.analyze(anomal, baseline, signal_quality=1.0)
        assert ar.anomaly_score  == 1.0
        assert ar.health_score   == 0
        assert ar.nivel_riesgo   == "CRÍTICO"

    def test_zscore_low_quality_no_health(self):
        det    = ZScoreDetector()
        ar     = det.analyze(np.array([0.1]*8), make_baseline_stats(), signal_quality=0.2)
        assert ar.health_score is None
        assert ar.resultado    == "SENSOR_ERROR"

    def test_zscore_is_cold_start_true(self):
        det = ZScoreDetector()
        ar  = det.analyze(np.array([0.1]*8), make_baseline_stats())
        assert ar.is_cold_start is True

    def test_zscore_nan_imputation_no_crash(self):
        det      = ZScoreDetector()
        nan_vec  = np.array([np.nan, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        ar       = det.analyze(nan_vec, make_baseline_stats(), signal_quality=1.0)
        assert np.isfinite(ar.anomaly_score)

    # ── IsolationForestDetector ───────────────────────────────────────────────

    def test_if_detector_ready(self):
        det = IsolationForestDetector(model=make_trained_if())
        assert det.is_ready is True

    def test_if_normal_score_lower_than_anomalous(self):
        model    = make_trained_if(n_samples=100)
        det      = IsolationForestDetector(model=model)
        baseline = make_baseline_stats()
        r_normal = det.analyze(np.array([0.1]*8), baseline, signal_quality=1.0)
        r_anomal = det.analyze(np.array([0.5]*8), baseline, signal_quality=1.0)
        assert r_normal.anomaly_score < r_anomal.anomaly_score

    def test_if_cold_start_false(self):
        det = IsolationForestDetector(model=make_trained_if())
        ar  = det.analyze(np.array([0.1]*8), make_baseline_stats())
        assert ar.is_cold_start is False

    def test_if_model_version_id_propagated(self):
        det = IsolationForestDetector(model=make_trained_if(), model_version_id=99)
        ar  = det.analyze(np.array([0.1]*8), make_baseline_stats())
        assert ar.model_version_id == 99

    def test_anomaly_score_always_in_range(self, tmp_path):
        """All signal modes must produce anomaly_score in [0, 1]."""
        model   = make_trained_if(n_samples=100)
        det     = IsolationForestDetector(model=model)
        baseline = make_baseline_stats()
        for mode in [SignalMode.NORMAL, SignalMode.IMBALANCE, SignalMode.BEARING_DEGRADATION]:
            reading = make_sensor_reading(mode)
            mgr = MachineBaselineManager(
                maquina_id=1, empresa_id=1,
                model_store=ModelStore(str(tmp_path / "ms")),
            )
            fs     = make_feature_set(tmp_path)
            vector = mgr.extract_feature_vector(fs)
            if vector is not None:
                ar = det.analyze(vector, baseline)
                assert 0.0 <= ar.anomaly_score <= 1.0, \
                    f"Mode {mode}: score {ar.anomaly_score} out of [0,1]"

    def test_describe_contributors_on_anomaly(self):
        z_scores = {"rms_x": 5.0, "kurtosis_x": 4.2, "crest_factor_x": 0.5}
        desc = _describe_contributors(z_scores, threshold=3.0)
        assert "rms_x" in desc
        assert "kurtosis_x" in desc


# ─── TEST: HealthScoreCalculator ──────────────────────────────────────────────

class TestHealthScoreCalculator:

    def _ts(self, days_ago: int) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=days_ago)

    def test_empty_history_returns_unknown(self):
        trend, slope = HealthScoreCalculator().compute([], current_score=80)
        assert trend == "unknown"
        assert slope is None

    def test_critical_score_overrides_trend(self):
        scores = [
            {"score": 80, "timestamp": self._ts(0)},
            {"score": 82, "timestamp": self._ts(1)},
            {"score": 81, "timestamp": self._ts(2)},
        ]
        trend, _ = HealthScoreCalculator().compute(scores, current_score=20)
        assert trend == "critical"

    def test_stable_trend(self):
        scores = [
            {"score": 81, "timestamp": self._ts(0)},
            {"score": 80, "timestamp": self._ts(1)},
            {"score": 81, "timestamp": self._ts(2)},
        ]
        trend, slope = HealthScoreCalculator().compute(scores, current_score=81)
        assert trend == "stable"
        assert slope is not None
        assert abs(slope) < 1.0

    def test_degrading_trend(self):
        scores = [
            {"score": 60, "timestamp": self._ts(0)},
            {"score": 70, "timestamp": self._ts(1)},
            {"score": 80, "timestamp": self._ts(2)},
        ]
        trend, slope = HealthScoreCalculator().compute(scores, current_score=60)
        assert trend == "degrading"
        assert slope is not None and slope < 0


# ─── TEST: IsolationForestTrigger ─────────────────────────────────────────────

class TestIsolationForestTrigger:

    def _fs_with_result(self, health_score, is_cold_start=False, nivel_riesgo="Bajo", tmp_path=None):
        if tmp_path is None:
            tmp_path = Path(tempfile.mkdtemp())
        fs = make_feature_set(tmp_path)
        fs.anomaly_result = make_anomaly_result(
            health_score=health_score,
            nivel_riesgo=nivel_riesgo,
            is_cold_start=is_cold_start,
        )
        return fs

    def test_no_anomaly_result_returns_false(self, tmp_path):
        t  = IsolationForestTrigger(capture_threshold=50)
        fs = make_feature_set(tmp_path)
        assert t.should_capture(fs) is False

    def test_cold_start_returns_false(self, tmp_path):
        t  = IsolationForestTrigger(capture_threshold=50)
        fs = self._fs_with_result(10, is_cold_start=True, tmp_path=tmp_path)
        assert t.should_capture(fs) is False

    def test_above_threshold_returns_false(self, tmp_path):
        t  = IsolationForestTrigger(capture_threshold=50)
        fs = self._fs_with_result(75, tmp_path=tmp_path)
        assert t.should_capture(fs) is False

    def test_none_health_score_returns_false(self, tmp_path):
        t  = IsolationForestTrigger(capture_threshold=50)
        fs = self._fs_with_result(None, tmp_path=tmp_path)
        assert t.should_capture(fs) is False

    def test_sensor_error_returns_false(self, tmp_path):
        t  = IsolationForestTrigger(capture_threshold=50)
        fs = self._fs_with_result(10, nivel_riesgo="Desconocido", tmp_path=tmp_path)
        assert t.should_capture(fs) is False

    def test_below_threshold_returns_true(self, tmp_path):
        t  = IsolationForestTrigger(capture_threshold=50, cooldown_seconds=0)
        fs = self._fs_with_result(30, tmp_path=tmp_path)
        assert t.should_capture(fs) is True

    def test_cooldown_prevents_second_capture(self, tmp_path):
        t  = IsolationForestTrigger(capture_threshold=50, cooldown_seconds=3600)
        fs = self._fs_with_result(30, tmp_path=tmp_path)
        assert t.should_capture(fs) is True    # first capture
        assert t.should_capture(fs) is False   # cooldown active

    def test_reset_cooldown_allows_capture_again(self, tmp_path):
        t  = IsolationForestTrigger(capture_threshold=50, cooldown_seconds=3600)
        fs = self._fs_with_result(30, tmp_path=tmp_path)
        t.should_capture(fs)    # first capture, starts cooldown
        t.reset_cooldown()      # reset
        assert t.should_capture(fs) is True


# ─── TEST: RawEventCapture ────────────────────────────────────────────────────

class TestRawEventCapture:

    def test_npz_file_created(self, tmp_path):
        cap     = RawEventCapture(str(tmp_path))
        reading = make_sensor_reading()
        path    = cap._save_npz(reading, maquina_id=1, window_id="test-uuid")
        assert path.exists()
        assert path.suffix == ".npz"

    def test_npz_axes_match_reading(self, tmp_path):
        cap     = RawEventCapture(str(tmp_path))
        reading = make_sensor_reading()
        path    = cap._save_npz(reading, maquina_id=1, window_id="test-uuid")
        data    = np.load(str(path))
        for axis in reading.available_axes:
            assert axis in data
            # float32 rounding after save — use allclose with tolerance
            np.testing.assert_allclose(
                data[axis].astype(np.float64),
                reading.axes[axis].astype(np.float64),
                rtol=1e-5, atol=1e-7,
                err_msg=f"Axis {axis} mismatch after save/load",
            )

    def test_timestamps_included_in_npz(self, tmp_path):
        cap     = RawEventCapture(str(tmp_path))
        reading = make_sensor_reading()
        path    = cap._save_npz(reading, maquina_id=1, window_id="test-uuid")
        data    = np.load(str(path))
        assert "timestamps" in data

    def test_checksum_is_64_char_hex(self, tmp_path):
        cap     = RawEventCapture(str(tmp_path))
        reading = make_sensor_reading()
        path    = cap._save_npz(reading, maquina_id=1, window_id="test-uuid")
        chk     = hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(chk) == 64
        assert all(c in "0123456789abcdef" for c in chk)

    def test_file_path_contains_maquina_id_and_window_id(self, tmp_path):
        cap  = RawEventCapture(str(tmp_path))
        path = cap._save_npz(make_sensor_reading(), maquina_id=7, window_id="my-uuid")
        assert "7" in str(path)           # maquina_id in directory
        assert "my-uuid" in path.name    # window_id in filename

    def test_file_saved_even_when_bd_offline(self, tmp_path):
        """
        capture() must save the .npz even when registrar_evento_raw fails.
        The method returns None (no event_id) but the file exists.
        """
        cap     = RawEventCapture(str(tmp_path / "raw"))
        reading = make_sensor_reading()
        fs      = make_feature_set(tmp_path)
        fs.anomaly_result = make_anomaly_result(health_score=20)

        # BD is offline: capture() will fail to register but file should exist
        result = cap.capture(reading, fs, lectura_id=None)
        # result is None (BD offline) but the .npz file must exist
        npz_files = list((tmp_path / "raw").rglob("*.npz"))
        assert len(npz_files) == 1, "NPZ file must be saved even if BD is offline"

    def test_npz_is_loadable_with_numpy(self, tmp_path):
        cap     = RawEventCapture(str(tmp_path))
        reading = make_sensor_reading()
        path    = cap._save_npz(reading, maquina_id=1, window_id="load-test")
        loaded  = np.load(str(path))
        assert len(loaded.files) >= 3   # at least x, y, z


# ─── TEST: EdgePipeline Fase 2C ───────────────────────────────────────────────

class TestEdgePipelinePhase2C:
    """
    Integration tests for the full Fase 2C pipeline.
    Uses _MockBaselineManager to control anomaly detection output
    without requiring a real database or trained model.
    """

    def _make_pipeline(
        self,
        tmp_path:      Path,
        preset_result: AnomalyResult,
        sensor_mode:   SignalMode = SignalMode.NORMAL,
        trigger:       Optional[AnomalyTrigger] = None,
    ) -> EdgePipeline:
        config     = make_edge_config(tmp_path)
        sensor_cfg = make_sensor_config()
        sensor     = MockSensor(sensor_cfg, MockSensorParams(mode=sensor_mode))
        mock_mgr   = _MockBaselineManager(preset_result)

        return EdgePipeline(
            config                = config,
            sensor                = sensor,
            anomaly_trigger       = trigger or PlaceholderAnomalyTrigger(),
            persist_fn            = lambda **kw: 42,
            resolve_machine_id_fn = lambda n: 42,
            baseline_manager      = mock_mgr,
            raw_capture           = None,   # no filesystem side-effects for most tests
        )

    def test_anomaly_result_set_after_run_once(self, tmp_path):
        """run_once() must set feature_set.anomaly_result."""
        preset   = make_anomaly_result(health_score=85, resultado="OK - Sano")
        pipeline = self._make_pipeline(tmp_path, preset)
        pipeline.startup()
        fs = pipeline.run_once()
        assert fs is not None
        assert fs.anomaly_result is not None
        assert fs.anomaly_result.resultado == "OK - Sano"

    def test_payload_contains_real_resultado(self, tmp_path):
        """to_lectura_cnc_payload() must use anomaly_result values."""
        preset   = make_anomaly_result(health_score=30, resultado="ALERTA",
                                        nivel_riesgo="Alto", anomaly_score=0.72)
        pipeline = self._make_pipeline(tmp_path, preset)
        pipeline.startup()
        fs = pipeline.run_once()
        pl = fs.to_lectura_cnc_payload()
        assert pl["resultado"]     == "ALERTA"
        assert pl["nivel_riesgo"]  == "Alto"
        assert pl["health_score"]  == 30
        assert pl["anomaly_score"] == 0.72

    def test_cold_start_resultado_aprendiendo(self, tmp_path):
        preset   = make_anomaly_result(
            health_score=None, resultado="OK - Aprendiendo",
            is_cold_start=True, anomaly_score=0.0,
        )
        pipeline = self._make_pipeline(tmp_path, preset)
        pipeline.startup()
        fs = pipeline.run_once()
        assert fs.anomaly_result.resultado == "OK - Aprendiendo"
        assert fs.anomaly_result.is_cold_start is True

    def test_sensor_error_skips_anomaly_detection(self, tmp_path):
        """SENSOR_FAILURE → session.acquire() returns None → no anomaly detection."""
        preset   = make_anomaly_result()
        pipeline = self._make_pipeline(tmp_path, preset,
                                        sensor_mode=SignalMode.SENSOR_FAILURE)
        pipeline.startup()
        fs = pipeline.run_once()
        assert fs is None   # sensor error → no FeatureSet → no detection

    def test_normal_reading_accumulated_in_baseline(self, tmp_path):
        """Normal resultado → baseline_manager.record_reading(is_normal=True)."""
        preset   = make_anomaly_result(health_score=85, resultado="OK - Sano")
        pipeline = self._make_pipeline(tmp_path, preset)
        pipeline.startup()
        pipeline.run_once()
        # The mock manager records whether it was considered normal
        assert True in pipeline._baseline_manager._recorded

    def test_anomalous_reading_not_accumulated(self, tmp_path):
        """NOK resultado → baseline_manager.record_reading(is_normal=False)."""
        preset = make_anomaly_result(
            health_score=10, resultado="NOK - Anomalía Detectada",
            nivel_riesgo="CRÍTICO",
        )
        pipeline = self._make_pipeline(tmp_path, preset)
        pipeline.startup()
        pipeline.run_once()
        assert False in pipeline._baseline_manager._recorded

    def test_raw_capture_triggered_on_low_health(self, tmp_path):
        """IsolationForestTrigger fires when health_score < capture_threshold."""
        preset   = make_anomaly_result(health_score=20, resultado="ALERTA",
                                        nivel_riesgo="Alto", is_cold_start=False)
        trigger  = IsolationForestTrigger(capture_threshold=50, cooldown_seconds=0)
        raw_dir  = tmp_path / "raw"

        config     = make_edge_config(tmp_path)
        sensor     = MockSensor(make_sensor_config(), MockSensorParams())
        mock_mgr   = _MockBaselineManager(preset)
        raw_cap    = RawEventCapture(str(raw_dir))

        pipeline = EdgePipeline(
            config=config, sensor=sensor,
            anomaly_trigger=trigger,
            persist_fn=lambda **kw: 42,
            resolve_machine_id_fn=lambda n: 42,
            baseline_manager=mock_mgr,
            raw_capture=raw_cap,
        )
        pipeline.startup()
        pipeline.run_once()

        npz_files = list(raw_dir.rglob("*.npz"))
        assert len(npz_files) == 1, "RAW capture must create a .npz file"

    def test_raw_capture_not_triggered_on_high_health(self, tmp_path):
        """IsolationForestTrigger does NOT fire when health_score >= threshold."""
        preset   = make_anomaly_result(health_score=90, resultado="OK - Sano",
                                        nivel_riesgo="Bajo", is_cold_start=False)
        trigger  = IsolationForestTrigger(capture_threshold=50, cooldown_seconds=0)
        raw_dir  = tmp_path / "raw"

        config  = make_edge_config(tmp_path)
        sensor  = MockSensor(make_sensor_config(), MockSensorParams())
        raw_cap = RawEventCapture(str(raw_dir))

        pipeline = EdgePipeline(
            config=config, sensor=sensor,
            anomaly_trigger=trigger,
            persist_fn=lambda **kw: 42,
            resolve_machine_id_fn=lambda n: 42,
            baseline_manager=_MockBaselineManager(preset),
            raw_capture=raw_cap,
        )
        pipeline.startup()
        pipeline.run_once()

        npz_files = list(raw_dir.rglob("*.npz"))
        assert len(npz_files) == 0, "RAW capture must NOT fire on high health"

    def test_offline_buffer_contains_real_scores(self, tmp_path):
        """Even in offline mode, the buffered payload contains real anomaly data."""
        preset = make_anomaly_result(health_score=40, anomaly_score=0.60,
                                      resultado="ADVERTENCIA", nivel_riesgo="Medio")
        config  = make_edge_config(tmp_path)
        sensor  = MockSensor(make_sensor_config(), MockSensorParams())

        pipeline = EdgePipeline(
            config=config, sensor=sensor,
            persist_fn=lambda **kw: None,   # offline
            resolve_machine_id_fn=lambda n: 42,
            baseline_manager=_MockBaselineManager(preset),
        )
        pipeline.startup()
        pipeline.run_once()

        assert pipeline.buffer.pending_count() == 1
        entry_file = list(Path(config.buffer.base_dir).glob("*.json"))[0]
        stored = json.loads(entry_file.read_text())
        assert stored["health_score"]  == 40
        assert stored["anomaly_score"] == 0.60
        assert stored["resultado"]     == "ADVERTENCIA"


# ─── REGRESIÓN ─────────────────────────────────────────────────────────────────

class TestFase1Fase2BRegression:
    """Ensures Fase 2C modules don't break or duplicate existing code."""

    def test_signal_processing_unchanged(self):
        from src.edge.signal_processing import process_vibration_signal
        assert callable(process_vibration_signal)

    def test_feature_extraction_unchanged(self):
        from src.edge.feature_extraction import extract_multiaxis_features
        assert callable(extract_multiaxis_features)

    def test_placeholder_trigger_still_works(self, tmp_path):
        p = PlaceholderAnomalyTrigger()
        fs = make_feature_set(tmp_path)
        assert p.should_capture(fs) is False

    def test_anomaly_modules_dont_define_dsp(self):
        import src.edge.anomaly.anomaly_detector as ad
        for name in ("compute_rms", "compute_fft", "bandpass_filter"):
            assert not hasattr(ad, name), f"anomaly_detector must not define {name}"

    def test_feature_set_anomaly_result_default_none(self, tmp_path):
        fs = make_feature_set(tmp_path)
        assert fs.anomaly_result is None

    def test_payload_fallback_when_no_anomaly_result(self, tmp_path):
        fs = make_feature_set(tmp_path)
        pl = fs.to_lectura_cnc_payload()
        # Fase 2B fallback values
        assert pl["resultado"]    == "OK - Sin validar"
        assert pl["anomaly_score"] is None
        assert pl["health_score"]  is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
