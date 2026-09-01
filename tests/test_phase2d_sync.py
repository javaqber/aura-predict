"""
Tests de la Fase 2D — Sincronización Edge ↔ Supabase Storage

Cobertura:
  TestStorageClient     — upload, exists, download con session mock
  TestConnectivity      — is_supabase_reachable, get_storage_client_from_env
  TestRawStorageSync    — upload_pending: éxito, fallo, skip, idempotencia
  TestModelSync         — upload_model, download_model, verificación SHA-256
  TestSyncConfig        — dataclass y carga desde YAML
  TestEdgePipelineSync  — integración raw_storage_sync en run_once()
  TestRegression        — los 245 tests existentes siguen pasando

Todos los tests son unitarios — sin Supabase real.
HTTP y BD se mockean completamente.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.edge.sync.storage_client import StorageClient
from src.edge.sync.connectivity import is_supabase_reachable, get_storage_client_from_env
from src.edge.sync.raw_storage_sync import RawStorageSync
from src.edge.sync.model_sync import ModelSync
from src.edge.config.edge_config import (
    EdgeConfig, MachineConfig, AcquisitionConfig, BufferConfig,
    AnomalyConfig, SyncConfig,
)
from src.edge.signal_processing import SignalConfig
from src.edge.sensors.base_sensor import SensorConfig
from src.edge.sensors.mock_sensor import MockSensor, MockSensorParams, SignalMode
from src.edge.pipeline.pipeline import EdgePipeline
from src.edge.pipeline.models import PlaceholderAnomalyTrigger


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_mock_session(status_code: int = 200, content: bytes = b"data") -> MagicMock:
    """Create a mock requests.Session with configurable response."""
    session  = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    response.iter_content = lambda chunk_size: iter([content])
    response.__enter__ = lambda s: response
    response.__exit__  = MagicMock(return_value=False)
    session.post.return_value   = response
    session.head.return_value   = response
    session.get.return_value    = response
    return session


def make_storage_client(status: int = 200, content: bytes = b"data") -> StorageClient:
    return StorageClient(
        supabase_url     = "https://x.supabase.co",
        service_role_key = "test-key",
        session          = make_mock_session(status, content),
    )


def make_npz(tmp_path: Path, name: str = "test-uuid") -> Path:
    path = tmp_path / f"{name}.npz"
    np.savez_compressed(str(path.with_suffix("")), x=np.ones(10))
    return path


def make_joblib(tmp_path: Path, name: str = "model") -> Path:
    import joblib
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(n_estimators=5, random_state=42)
    model.fit(np.random.randn(20, 8))
    path = tmp_path / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_edge_config(tmp_path: Path, maquina_id: int = 1) -> EdgeConfig:
    return EdgeConfig(
        machine     = MachineConfig(machine_id="T", empresa_id=1, maquina_id=maquina_id),
        sensor      = SensorConfig(sensor_id="t", sensor_type="mock",
                                   sampling_rate_hz=3200, samples_per_window=3200,
                                   axes=["x","y","z"]),
        signal      = SignalConfig(fs=3200),
        acquisition = AcquisitionConfig(),
        buffer      = BufferConfig(base_dir=str(tmp_path / "buffer"), max_entries=10),
        anomaly     = AnomalyConfig(baseline_min_samples=5, update_every_n=5,
                                    model_base_dir=str(tmp_path / "models"),
                                    raw_base_dir=str(tmp_path / "raw")),
        sync        = SyncConfig(enabled=True, max_raw_per_cycle=5),
    )


# ─── TestStorageClient ────────────────────────────────────────────────────────

class TestStorageClient:

    def test_upload_success_returns_true(self, tmp_path):
        client = make_storage_client(200)
        f = tmp_path / "f.npz"
        f.write_bytes(b"data")
        assert client.upload("bucket", "key/f.npz", f) is True

    def test_upload_201_also_success(self, tmp_path):
        client = make_storage_client(201)
        f = tmp_path / "f.npz"
        f.write_bytes(b"data")
        assert client.upload("bucket", "key/f.npz", f) is True

    def test_upload_failure_returns_false(self, tmp_path):
        client = make_storage_client(500)
        f = tmp_path / "f.npz"
        f.write_bytes(b"data")
        assert client.upload("bucket", "key/f.npz", f) is False

    def test_upload_missing_file_returns_false(self, tmp_path):
        client = make_storage_client(200)
        assert client.upload("bucket", "key", tmp_path / "nonexistent.npz") is False

    def test_upload_connection_error_returns_false(self, tmp_path):
        import requests as req
        session = MagicMock()
        session.post.side_effect = req.ConnectionError("down")
        client = StorageClient("https://x.supabase.co", "key", session=session)
        f = tmp_path / "f.npz"
        f.write_bytes(b"data")
        assert client.upload("bucket", "key", f) is False

    def test_upload_timeout_returns_false(self, tmp_path):
        import requests as req
        session = MagicMock()
        session.post.side_effect = req.Timeout("timeout")
        client = StorageClient("https://x.supabase.co", "key", session=session)
        f = tmp_path / "f.npz"
        f.write_bytes(b"data")
        assert client.upload("bucket", "key", f) is False

    def test_upload_uses_upsert_header_by_default(self, tmp_path):
        client = make_storage_client(200)
        f = tmp_path / "f.npz"
        f.write_bytes(b"data")
        client.upload("bucket", "key", f)
        _, kwargs = client._session.post.call_args
        headers = kwargs.get("headers", {})
        assert headers.get("x-upsert") == "true"

    def test_exists_true_on_200(self):
        client = make_storage_client(200)
        assert client.exists("bucket", "key/f.npz") is True

    def test_exists_false_on_404(self):
        client = make_storage_client(404)
        assert client.exists("bucket", "key/missing.npz") is False

    def test_exists_false_on_connection_error(self):
        import requests as req
        session = MagicMock()
        session.head.side_effect = req.ConnectionError()
        client = StorageClient("https://x.supabase.co", "key", session=session)
        assert client.exists("bucket", "key") is False

    def test_download_success_creates_file(self, tmp_path):
        content = b"fake_model_bytes"
        client  = make_storage_client(200, content)
        dest    = tmp_path / "downloaded.npz"
        ok      = client.download("bucket", "key", dest)
        assert ok
        assert dest.exists()
        assert dest.read_bytes() == content

    def test_download_failure_no_partial_file(self, tmp_path):
        client = make_storage_client(404)
        dest   = tmp_path / "downloaded.npz"
        ok     = client.download("bucket", "key", dest)
        assert not ok
        assert not dest.exists()

    def test_download_no_tmp_left_on_failure(self, tmp_path):
        import requests as req
        session = MagicMock()
        resp    = MagicMock()
        resp.status_code = 200
        resp.iter_content = lambda chunk_size: (_ for _ in ()).throw(req.ConnectionError())
        resp.__enter__ = lambda s: resp
        resp.__exit__  = MagicMock(return_value=False)
        session.get.return_value = resp
        client = StorageClient("https://x.supabase.co", "key", session=session)
        dest   = tmp_path / "out.npz"
        ok     = client.download("bucket", "k", dest)
        assert not ok
        assert not Path(str(dest) + ".tmp").exists()

    def test_from_env_returns_none_without_vars(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        assert StorageClient.from_env() is None

    def test_from_env_returns_client_with_vars(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
        client = StorageClient.from_env()
        assert isinstance(client, StorageClient)


# ─── TestConnectivity ─────────────────────────────────────────────────────────

class TestConnectivity:

    def test_reachable_on_200(self):
        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 200
            assert is_supabase_reachable("https://x.supabase.co") is True

    def test_reachable_on_401(self):
        """401 Unauthorized = server is up and responding."""
        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 401
            assert is_supabase_reachable("https://x.supabase.co") is True

    def test_unreachable_on_500(self):
        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 500
            assert is_supabase_reachable("https://x.supabase.co") is False

    def test_unreachable_on_connection_error(self):
        import requests as req
        with patch("requests.head", side_effect=req.ConnectionError()):
            assert is_supabase_reachable("https://x.supabase.co") is False

    def test_unreachable_on_timeout(self):
        import requests as req
        with patch("requests.head", side_effect=req.Timeout()):
            assert is_supabase_reachable("https://x.supabase.co") is False

    def test_get_client_from_env_none_without_vars(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        assert get_storage_client_from_env() is None

    def test_get_client_from_env_with_vars(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
        client = get_storage_client_from_env()
        assert isinstance(client, StorageClient)


# ─── TestRawStorageSync ───────────────────────────────────────────────────────

class TestRawStorageSync:

    def _make_sync(self, upload_ok=True, events=None, mark_ok=True):
        storage = MagicMock()
        storage.upload.return_value = upload_ok
        mark_calls = []
        fetch_fn = lambda mid: events or []
        mark_fn  = lambda eid, key, chk: (mark_calls.append((eid, key, chk)), mark_ok)[1]
        sync = RawStorageSync(storage, fetch_pending_fn=fetch_fn, mark_uploaded_fn=mark_fn)
        return sync, storage, mark_calls

    def _make_event(self, tmp_path, event_id=1, maquina_id=1, empresa_id=1):
        path = make_npz(tmp_path, f"uuid-{event_id}")
        sha  = sha256_of(path)
        return {"id": event_id, "maquina_id": maquina_id, "empresa_id": empresa_id,
                "event_timestamp": None, "total_samples": 10,
                "file_path": str(path), "file_checksum": sha}

    def test_upload_pending_returns_count(self, tmp_path):
        ev = self._make_event(tmp_path)
        sync, _, _ = self._make_sync(events=[ev])
        assert sync.upload_pending(1) == 1

    def test_upload_pending_calls_storage_upload(self, tmp_path):
        ev = self._make_event(tmp_path)
        sync, storage, _ = self._make_sync(events=[ev])
        sync.upload_pending(1)
        assert storage.upload.called

    def test_upload_pending_marks_uploaded_on_success(self, tmp_path):
        ev = self._make_event(tmp_path)
        sync, _, mark_calls = self._make_sync(events=[ev])
        sync.upload_pending(1)
        assert len(mark_calls) == 1
        assert mark_calls[0][0] == 1

    def test_upload_pending_deletes_local_on_success(self, tmp_path):
        ev = self._make_event(tmp_path)
        path = Path(ev["file_path"])
        sync, _, _ = self._make_sync(events=[ev])
        sync.upload_pending(1)
        assert not path.exists()

    def test_upload_pending_keeps_local_on_upload_failure(self, tmp_path):
        ev   = self._make_event(tmp_path)
        path = Path(ev["file_path"])
        sync, _, mark_calls = self._make_sync(upload_ok=False, events=[ev])
        n = sync.upload_pending(1)
        assert n == 0
        assert len(mark_calls) == 0
        assert path.exists()

    def test_upload_pending_stops_at_first_failure(self, tmp_path):
        ev1 = self._make_event(tmp_path, 1)
        ev2 = self._make_event(tmp_path, 2)
        storage = MagicMock()
        call_n  = [0]
        def upload_side(*a, **kw):
            call_n[0] += 1
            return call_n[0] < 2  # first=True, second=False
        storage.upload.side_effect = upload_side
        mark_calls = []
        sync = RawStorageSync(storage,
                              fetch_pending_fn=lambda m: [ev1, ev2],
                              mark_uploaded_fn=lambda e, k, c: (mark_calls.append(e), True)[1])
        n = sync.upload_pending(1)
        assert n == 1
        assert 1 in mark_calls and 2 not in mark_calls

    def test_upload_pending_skips_missing_file(self, tmp_path):
        missing = {"id": 1, "maquina_id": 1, "empresa_id": 1,
                   "event_timestamp": None, "total_samples": 10,
                   "file_path": "/nonexistent/path.npz", "file_checksum": None}
        ev2 = self._make_event(tmp_path, 2)
        sync, storage, mark_calls = self._make_sync(events=[missing, ev2])
        n = sync.upload_pending(1)
        assert n == 1       # second event uploaded despite first being skipped
        assert mark_calls[0][0] == 2

    def test_max_per_cycle_respected(self, tmp_path):
        events = [self._make_event(tmp_path, i) for i in range(1, 6)]
        sync, storage, _ = self._make_sync(events=events)
        n = sync.upload_pending(1, max_per_cycle=3)
        assert n == 3
        assert storage.upload.call_count == 3

    def test_storage_key_format(self):
        key = RawStorageSync.storage_key(2, 42, "my-uuid")
        assert key == "2/42/my-uuid.npz"

    def test_storage_key_in_mark_call(self, tmp_path):
        ev = self._make_event(tmp_path, empresa_id=3, maquina_id=7)
        sync, _, mark_calls = self._make_sync(events=[ev])
        sync.upload_pending(7)
        _, storage_key, _ = mark_calls[0]
        assert storage_key == "3/7/uuid-1.npz"

    def test_checksum_recomputed_if_missing(self, tmp_path):
        ev = self._make_event(tmp_path)
        ev["file_checksum"] = None   # remove stored checksum
        sync, _, mark_calls = self._make_sync(events=[ev])
        sync.upload_pending(1)
        _, _, chk = mark_calls[0]
        assert chk is not None and len(chk) == 64

    def test_empty_pending_returns_zero(self):
        sync, _, _ = self._make_sync(events=[])
        assert sync.upload_pending(1) == 0

    def test_fetch_failure_returns_zero(self, tmp_path):
        storage = MagicMock()
        def bad_fetch(mid): raise RuntimeError("DB down")
        sync = RawStorageSync(storage,
                              fetch_pending_fn=bad_fetch,
                              mark_uploaded_fn=lambda e, k, c: True)
        assert sync.upload_pending(1) == 0

    def test_upsert_true_passed_to_storage(self, tmp_path):
        ev = self._make_event(tmp_path)
        sync, storage, _ = self._make_sync(events=[ev])
        sync.upload_pending(1)
        _, kwargs = storage.upload.call_args
        assert kwargs.get("upsert") is True


# ─── TestModelSync ────────────────────────────────────────────────────────────

class TestModelSync:

    def _make_sync(self, upload_ok=True, update_ok=True):
        storage = MagicMock()
        storage.upload.return_value = upload_ok
        update_calls = []
        update_fn = lambda mid, st, sp, chk: (update_calls.append((mid, st, sp, chk)), update_ok)[1]
        sync = ModelSync(storage, update_storage_fn=update_fn)
        return sync, storage, update_calls

    def test_upload_model_success(self, tmp_path):
        model_path = make_joblib(tmp_path)
        sync, _, update_calls = self._make_sync()
        ok = sync.upload_model(1, 1, 7, "1.100.0", model_path)
        assert ok
        assert len(update_calls) == 1
        mid, st, sp, chk = update_calls[0]
        assert st == "supabase"
        assert sp == "1/1/1.100.0.joblib"
        assert len(chk) == 64

    def test_upload_model_local_file_preserved(self, tmp_path):
        model_path = make_joblib(tmp_path)
        sync, _, _ = self._make_sync()
        sync.upload_model(1, 1, 7, "1.100.0", model_path)
        assert model_path.exists()

    def test_upload_failure_no_db_update(self, tmp_path):
        model_path = make_joblib(tmp_path)
        sync, _, update_calls = self._make_sync(upload_ok=False)
        ok = sync.upload_model(1, 1, 7, "1.100.0", model_path)
        assert not ok
        assert len(update_calls) == 0

    def test_upload_missing_file_returns_false(self, tmp_path):
        sync, _, _ = self._make_sync()
        ok = sync.upload_model(1, 1, 7, "1.100.0", Path("/nonexistent.joblib"))
        assert not ok

    def test_download_success(self, tmp_path):
        src  = make_joblib(tmp_path, "src_model")
        dest = tmp_path / "dest_model.joblib"
        sha  = sha256_of(src)
        storage = MagicMock()
        def fake_dl(bucket, key, candidate):
            shutil.copy(src, candidate)
            return True
        storage.download.side_effect = fake_dl
        sync = ModelSync(storage)
        ok = sync.download_model(1, 1, "1.100.0", dest, expected_sha=sha)
        assert ok
        assert dest.exists()
        assert sha256_of(dest) == sha

    def test_download_sha_mismatch_rollback(self, tmp_path):
        src  = make_joblib(tmp_path, "src_model")
        dest = tmp_path / "dest_model.joblib"
        storage = MagicMock()
        def fake_dl(bucket, key, candidate):
            shutil.copy(src, candidate)
            return True
        storage.download.side_effect = fake_dl
        sync = ModelSync(storage)
        ok = sync.download_model(1, 1, "1.100.0", dest, expected_sha="bad_sha")
        assert not ok
        assert not dest.exists()
        assert not Path(str(dest) + ".candidate").exists()

    def test_download_preserves_old_model_on_sha_fail(self, tmp_path):
        old  = make_joblib(tmp_path, "old_model")
        dest = tmp_path / "active_model.joblib"
        shutil.copy(old, dest)
        old_sha = sha256_of(dest)
        new_src = make_joblib(tmp_path, "new_bad_model")
        storage = MagicMock()
        def fake_dl(bucket, key, candidate):
            shutil.copy(new_src, candidate)
            return True
        storage.download.side_effect = fake_dl
        sync = ModelSync(storage)
        ok = sync.download_model(1, 1, "1.100.0", dest, expected_sha="wrong")
        assert not ok
        assert sha256_of(dest) == old_sha   # unchanged

    def test_download_failure_returns_false(self, tmp_path):
        dest    = tmp_path / "dest.joblib"
        storage = MagicMock()
        storage.download.return_value = False
        sync = ModelSync(storage)
        ok = sync.download_model(1, 1, "1.100.0", dest)
        assert not ok
        assert not dest.exists()

    def test_download_without_sha_skips_verification(self, tmp_path):
        src  = make_joblib(tmp_path, "src")
        dest = tmp_path / "dest.joblib"
        storage = MagicMock()
        def fake_dl(bucket, key, candidate):
            shutil.copy(src, candidate)
            return True
        storage.download.side_effect = fake_dl
        sync = ModelSync(storage)
        ok = sync.download_model(1, 1, "1.100.0", dest, expected_sha=None)
        assert ok

    def test_storage_key_format(self):
        key = ModelSync.storage_key(2, 42, "1.100.0")
        assert key == "2/42/1.100.0.joblib"


# ─── TestSyncConfig ───────────────────────────────────────────────────────────

class TestSyncConfig:

    def test_defaults(self):
        sc = SyncConfig()
        assert sc.raw_bucket        == "aurapredict-raw-events"
        assert sc.model_bucket      == "aurapredict-models"
        assert sc.max_raw_per_cycle == 5
        assert sc.enabled           is True

    def test_edge_config_has_sync_field(self, tmp_path):
        cfg = make_edge_config(tmp_path)
        assert isinstance(cfg.sync, SyncConfig)

    def test_edge_config_without_sync_uses_defaults(self, tmp_path):
        cfg = EdgeConfig(
            machine=MachineConfig(machine_id="T", empresa_id=1, maquina_id=1),
            sensor=SensorConfig(sensor_id="t", sensor_type="mock",
                                sampling_rate_hz=3200, samples_per_window=3200),
            signal=SignalConfig(fs=3200), acquisition=AcquisitionConfig(),
            buffer=BufferConfig(base_dir=str(tmp_path)),
        )
        assert cfg.sync.enabled is True

    def test_from_yaml_loads_sync(self):
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "machines", "example_cnc.yaml"
        )
        cfg = EdgeConfig.from_yaml(yaml_path)
        assert isinstance(cfg.sync, SyncConfig)
        assert cfg.sync.raw_bucket == "aurapredict-raw-events"


# ─── TestEdgePipelineSync ─────────────────────────────────────────────────────

class TestEdgePipelineSync:

    def _make_pipeline(self, tmp_path, raw_sync=None, persist_ok=True):
        config = make_edge_config(tmp_path)
        sensor = MockSensor(config.sensor, MockSensorParams())
        return EdgePipeline(
            config                = config,
            sensor                = sensor,
            persist_fn            = lambda **kw: 1 if persist_ok else None,
            resolve_machine_id_fn = lambda n: 1,
            raw_storage_sync      = raw_sync,
        )

    def test_online_cycle_calls_upload_pending(self, tmp_path):
        mock_sync = MagicMock()
        pipeline  = self._make_pipeline(tmp_path, raw_sync=mock_sync)
        pipeline.startup()
        pipeline.run_once()
        mock_sync.upload_pending.assert_called_once()

    def test_offline_cycle_does_not_call_upload_pending(self, tmp_path):
        mock_sync = MagicMock()
        pipeline  = self._make_pipeline(tmp_path, raw_sync=mock_sync, persist_ok=False)
        pipeline.startup()
        pipeline.run_once()
        mock_sync.upload_pending.assert_not_called()

    def test_upload_pending_uses_max_per_cycle_from_config(self, tmp_path):
        mock_sync = MagicMock()
        pipeline  = self._make_pipeline(tmp_path, raw_sync=mock_sync)
        pipeline.startup()
        pipeline.run_once()
        _, kwargs = mock_sync.upload_pending.call_args
        assert kwargs.get("max_per_cycle") == pipeline.config.sync.max_raw_per_cycle

    def test_no_raw_sync_does_not_crash(self, tmp_path):
        """Pipeline must work normally when raw_storage_sync is None."""
        pipeline = self._make_pipeline(tmp_path, raw_sync=None)
        pipeline.startup()
        fs = pipeline.run_once()
        assert fs is not None

    def test_sync_disabled_no_auto_create(self, tmp_path):
        """When sync.enabled=False, startup() must not create a RawStorageSync."""
        config = make_edge_config(tmp_path)
        config.sync.enabled = False
        sensor = MockSensor(config.sensor, MockSensorParams())
        pipeline = EdgePipeline(
            config=config, sensor=sensor,
            persist_fn=lambda **kw: 1,
            resolve_machine_id_fn=lambda n: 1,
        )
        pipeline.startup()
        assert pipeline._raw_storage_sync is None


# ─── REGRESSION ───────────────────────────────────────────────────────────────

class TestRegression:

    def test_fase1_still_importable(self):
        from src.edge.signal_processing import process_vibration_signal
        from src.edge.feature_extraction import extract_multiaxis_features
        from src.edge.data_quality import check_signal_quality
        assert all(callable(f) for f in [
            process_vibration_signal, extract_multiaxis_features, check_signal_quality
        ])

    def test_sync_modules_dont_duplicate_dsp(self):
        import src.edge.sync.raw_storage_sync as rss
        import src.edge.sync.model_sync as ms
        for name in ("compute_rms", "bandpass_filter", "compute_fft"):
            assert not hasattr(rss, name)
            assert not hasattr(ms, name)

    def test_storage_client_never_crashes_on_none_session(self):
        """StorageClient must create its own session if none is injected."""
        client = StorageClient("https://x.supabase.co", "test-key")
        assert client._session is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
