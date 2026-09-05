"""
Tests de la Fase 5B — Gestión del ciclo de vida de modelos ML

Cobertura:
  TestModelManagerValidation  — los 5 checks de validación
  TestModelManagerListGet     — list_models, get_active_model
  TestModelManagerActivate    — activate_model: éxito y rechazo
  TestModelManagerRollback    — rollback: éxito, sin candidatos, candidato inválido
  TestModelManagerTrain       — train_model: éxito y sin datos suficientes
  TestRepositoriesModels      — nuevas funciones de repositories.py
  TestApiV2Modelos            — endpoints de modelos con aislamiento empresa
  TestDashboardModelos        — sección Modelos ML presente en dashboard
  TestOfflineModels           — operaciones sin Supabase
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import hashlib
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_valid_model(tmp_path: Path, n_features: int = 8) -> tuple[Path, str]:
    """Create a valid .joblib model file and return (path, sha256)."""
    model = IsolationForest(n_estimators=10, random_state=42)
    model.fit(np.random.randn(20, n_features))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha


def make_model_record(model_id=1, maquina_id=42, empresa_id=1,
                      version="1.50.0", is_active=True,
                      model_path="/tmp/model.joblib", model_checksum=None):
    return {
        "id":                  model_id,
        "maquina_id":          maquina_id,
        "empresa_id":          empresa_id,
        "model_version":       version,
        "algorithm":           "isolation_forest",
        "trained_at":          None,
        "training_samples":    50,
        "contamination":       None,
        "features_used":       None,
        "storage_type":        "local",
        "model_path":          model_path,
        "model_checksum":      model_checksum,
        "is_active":           is_active,
        "notes":               None,
        "performance_metrics": None,
    }


def make_manager(tmp_path=None, *, models=None, active=None,
                  activate_ok=True, previous=None):
    from src.edge.anomaly.model_manager import ModelManager
    mm = ModelManager(
        list_fn         = lambda mid: models or [],
        get_active_fn   = lambda mid: active,
        activate_fn     = lambda mid: activate_ok,
        get_previous_fn = lambda mid, excluir_id=None: previous or [],
    )
    return mm


# ─── TestModelManagerValidation ───────────────────────────────────────────────

class TestModelManagerValidation:

    def test_valid_model_passes_all_checks(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, sha = make_valid_model(tmp_path)
        vr = ModelManager().validate_model(str(path))
        assert vr.valid
        assert vr.error is None
        assert vr.checks.get("file_exists") is True
        assert vr.checks.get("joblib_load") is True
        assert vr.checks.get("features") is True
        assert vr.checks.get("predict") is True

    def test_file_not_found_fails(self):
        from src.edge.anomaly.model_manager import ModelManager
        vr = ModelManager().validate_model("/nonexistent/path.joblib")
        assert not vr.valid
        assert "not found" in vr.error
        assert vr.checks.get("file_exists") is False

    def test_correct_sha256_passes(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, sha = make_valid_model(tmp_path)
        vr = ModelManager().validate_model(str(path), expected_checksum=sha)
        assert vr.valid
        assert vr.checks.get("sha256") is True

    def test_wrong_sha256_fails(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, _ = make_valid_model(tmp_path)
        vr = ModelManager().validate_model(str(path), expected_checksum="wrongsha")
        assert not vr.valid
        assert "SHA-256 mismatch" in vr.error

    def test_no_checksum_skips_sha_check(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, _ = make_valid_model(tmp_path)
        vr = ModelManager().validate_model(str(path), expected_checksum=None)
        assert vr.valid
        assert vr.checks.get("sha256") is True   # skipped = True (non-blocking)

    def test_corrupt_joblib_fails(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        corrupt = tmp_path / "corrupt.joblib"
        corrupt.write_bytes(b"this is not a joblib file")
        vr = ModelManager().validate_model(str(corrupt))
        assert not vr.valid
        assert "joblib.load" in vr.error

    def test_wrong_n_features_fails(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        # 3 features instead of 8
        path, _ = make_valid_model(tmp_path, n_features=3)
        vr = ModelManager().validate_model(str(path))
        assert not vr.valid
        assert "Feature count mismatch" in vr.error

    def test_validation_result_is_dataclass(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager, ValidationResult
        path, _ = make_valid_model(tmp_path)
        vr = ModelManager().validate_model(str(path))
        assert isinstance(vr, ValidationResult)

    def test_predict_check_runs(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, _ = make_valid_model(tmp_path)
        vr = ModelManager().validate_model(str(path))
        assert "predict" in vr.checks


# ─── TestModelManagerListGet ──────────────────────────────────────────────────

class TestModelManagerListGet:

    def test_list_models_returns_list(self):
        records = [make_model_record(1), make_model_record(2, version="1.100.0")]
        mm = make_manager(models=records)
        result = mm.list_models(42)
        assert len(result) == 2

    def test_list_models_empty_on_db_error(self):
        from src.edge.anomaly.model_manager import ModelManager
        mm = ModelManager(list_fn=lambda mid: (_ for _ in ()).throw(Exception("DB down")))
        result = mm.list_models(42)
        assert result == []

    def test_get_active_returns_record(self):
        active = make_model_record(is_active=True)
        mm = make_manager(active=active)
        result = mm.get_active_model(42)
        assert result is not None
        assert result["is_active"] is True

    def test_get_active_returns_none_when_missing(self):
        mm = make_manager(active=None)
        assert mm.get_active_model(42) is None

    def test_list_models_contains_required_fields(self):
        record = make_model_record()
        mm = make_manager(models=[record])
        result = mm.list_models(42)
        expected = ["id", "model_version", "is_active", "storage_type",
                    "model_path", "training_samples"]
        for f in expected:
            assert f in result[0], f"Missing field: {f}"


# ─── TestModelManagerActivate ─────────────────────────────────────────────────

class TestModelManagerActivate:

    def test_activate_valid_model_succeeds(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, sha = make_valid_model(tmp_path)
        record = make_model_record(model_path=str(path), model_checksum=sha)
        activated = []
        mm = ModelManager(activate_fn=lambda mid: (activated.append(mid), True)[1])
        mm._get_model_by_id = lambda mid: record
        result = mm.activate_model(1)
        assert result.success
        assert 1 in activated

    def test_activate_nonexistent_file_fails(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        record = make_model_record(model_path="/nonexistent.joblib")
        mm = ModelManager(activate_fn=lambda mid: True)
        mm._get_model_by_id = lambda mid: record
        result = mm.activate_model(99)
        assert not result.success
        assert "Validation failed" in result.error

    def test_activate_model_not_in_registry_fails(self):
        from src.edge.anomaly.model_manager import ModelManager
        mm = ModelManager(activate_fn=lambda mid: True)
        mm._get_model_by_id = lambda mid: None
        result = mm.activate_model(999)
        assert not result.success
        assert "not found" in result.error

    def test_activate_wrong_sha_fails(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, _ = make_valid_model(tmp_path)
        record = make_model_record(model_path=str(path), model_checksum="badsha")
        mm = ModelManager(activate_fn=lambda mid: True)
        mm._get_model_by_id = lambda mid: record
        result = mm.activate_model(1)
        assert not result.success
        assert "SHA-256" in result.error

    def test_activate_returns_version(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, sha = make_valid_model(tmp_path)
        record = make_model_record(version="2.200.0", model_path=str(path), model_checksum=sha)
        mm = ModelManager(activate_fn=lambda mid: True)
        mm._get_model_by_id = lambda mid: record
        result = mm.activate_model(1)
        assert result.version == "2.200.0"

    def test_skip_validation_activates_without_checks(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        record = make_model_record(model_path="/nonexistent.joblib")
        activated = []
        mm = ModelManager(activate_fn=lambda mid: (activated.append(mid), True)[1])
        mm._get_model_by_id = lambda mid: record
        result = mm.activate_model(1, skip_validation=True)
        assert result.success
        assert 1 in activated


# ─── TestModelManagerRollback ─────────────────────────────────────────────────

class TestModelManagerRollback:

    def test_rollback_to_valid_previous(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, sha = make_valid_model(tmp_path)
        prev = make_model_record(model_id=5, version="1.50.0",
                                  model_path=str(path), model_checksum=sha)
        mm = ModelManager(
            get_active_fn   = lambda mid: make_model_record(model_id=6, version="1.100.0"),
            activate_fn     = lambda mid: True,
            get_previous_fn = lambda mid, excluir_id=None: [prev],
        )
        mm._get_model_by_id = lambda mid: prev
        result = mm.rollback_model(42)
        assert result.success
        assert result.version == "1.50.0"

    def test_rollback_no_candidates_fails(self):
        from src.edge.anomaly.model_manager import ModelManager
        mm = ModelManager(
            get_active_fn   = lambda mid: make_model_record(),
            activate_fn     = lambda mid: True,
            get_previous_fn = lambda mid, excluir_id=None: [],
        )
        result = mm.rollback_model(42)
        assert not result.success
        assert "No previous" in result.error

    def test_rollback_skips_invalid_candidates(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        # First candidate: invalid file; second: valid
        path, sha = make_valid_model(tmp_path)
        invalid = make_model_record(model_id=3, version="1.30.0",
                                     model_path="/nonexistent.joblib")
        valid_c = make_model_record(model_id=2, version="1.20.0",
                                     model_path=str(path), model_checksum=sha)
        activated = []
        mm = ModelManager(
            get_active_fn   = lambda mid: make_model_record(model_id=4),
            activate_fn     = lambda mid: (activated.append(mid), True)[1],
            get_previous_fn = lambda mid, excluir_id=None: [invalid, valid_c],
        )
        mm._get_model_by_id = lambda mid: valid_c if mid == 2 else invalid
        result = mm.rollback_model(42)
        assert result.success
        assert 2 in activated

    def test_rollback_all_invalid_fails(self):
        from src.edge.anomaly.model_manager import ModelManager
        bad = make_model_record(model_id=1, model_path="/nonexistent.joblib")
        mm = ModelManager(
            get_active_fn   = lambda mid: make_model_record(model_id=99),
            activate_fn     = lambda mid: True,
            get_previous_fn = lambda mid, excluir_id=None: [bad],
        )
        result = mm.rollback_model(42)
        assert not result.success
        assert "No valid previous" in result.error


# ─── TestModelManagerTrain ────────────────────────────────────────────────────

class TestModelManagerTrain:

    def test_train_without_baseline_manager_fails(self):
        from src.edge.anomaly.model_manager import ModelManager
        mm = ModelManager()
        result = mm.train_model(42)
        assert not result.success
        assert "No BaselineManager" in result.error

    def test_train_insufficient_data_fails(self):
        from src.edge.anomaly.model_manager import ModelManager
        bm = MagicMock()
        bm.is_baseline_ready   = False
        bm.n_samples           = 5
        bm._baseline_min_samples = 50
        mm = ModelManager(baseline_manager=bm)
        result = mm.train_model(42)
        assert not result.success
        assert "Insufficient" in result.error
        assert "5/50" in result.error

    def test_train_success(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        import numpy as np
        data = np.random.randn(60, 8)
        trained_id = [None]

        bm = MagicMock()
        bm.is_baseline_ready      = True
        bm.n_samples              = 60
        bm._baseline_min_samples  = 50
        bm._baseline_buffer       = data

        before_active = {"id": 1, "model_version": "1.50.0"}
        after_active  = {"id": 2, "model_version": "1.60.0"}
        call_count = [0]

        def get_active_side(mid):
            call_count[0] += 1
            return before_active if call_count[0] == 1 else after_active

        mm = ModelManager(
            baseline_manager = bm,
            get_active_fn    = get_active_side,
        )

        def fake_train(data):
            trained_id[0] = 2

        bm._train_isolation_forest.side_effect = fake_train
        result = mm.train_model(42)
        assert result.success
        assert result.model_id == 2


# ─── TestRepositoriesModels ───────────────────────────────────────────────────

class TestRepositoriesModels:

    def test_obtener_modelos_maquina_exists(self):
        import ast
        src = open(
            os.path.join(os.path.dirname(__file__),
                         "../src/database_v2/repositories.py"),
            encoding="utf-8",
        ).read()
        assert "def obtener_modelos_maquina" in src

    def test_obtener_modelos_anteriores_exists(self):
        src = open(
            os.path.join(os.path.dirname(__file__),
                         "../src/database_v2/repositories.py"),
            encoding="utf-8",
        ).read()
        assert "def obtener_modelos_anteriores_maquina" in src

    def test_obtener_modelo_activo_returns_more_columns(self):
        src = open(
            os.path.join(os.path.dirname(__file__),
                         "../src/database_v2/repositories.py"),
            encoding="utf-8",
        ).read()
        # Must select features_used and model_checksum
        assert "features_used" in src
        assert "model_checksum" in src

    def test_repositories_syntax_valid(self):
        import ast
        src = open(
            os.path.join(os.path.dirname(__file__),
                         "../src/database_v2/repositories.py"),
            encoding="utf-8",
        ).read()
        ast.parse(src)


# ─── TestApiV2Modelos ─────────────────────────────────────────────────────────

class TestApiV2Modelos:

    @pytest.fixture
    def client(self):
        pytest.importorskip("fastapi")
        try:
            from fastapi.testclient import TestClient
        except (RuntimeError, ModuleNotFoundError) as e:
            pytest.skip(f"TestClient unavailable: {e}")
        with patch("database.get_conn"), patch("database.init_db"):
            from api import app
            return TestClient(app, raise_server_exceptions=False)

    def _auth(self, empresa_id=1):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
        from auth import crear_token
        token = crear_token({"sub": "t@t.com", "nombre": "T",
                             "rol": "admin", "empresa_id": empresa_id})
        return {"Authorization": f"Bearer {token}"}

    def test_get_modelos_requires_auth(self, client):
        r = client.get("/v2/maquinas/1/modelos")
        assert r.status_code in (401, 403)

    def test_get_modelo_activo_requires_auth(self, client):
        r = client.get("/v2/maquinas/1/modelos/activo")
        assert r.status_code in (401, 403)

    def test_activar_requires_auth(self, client):
        r = client.post("/v2/maquinas/1/modelos/1/activar")
        assert r.status_code in (401, 403)

    def test_rollback_requires_auth(self, client):
        r = client.post("/v2/maquinas/1/modelos/1/rollback")
        assert r.status_code in (401, 403)

    def test_get_modelos_empresa_isolation(self, client):
        with patch("database_v2.repositories.obtener_empresa_id_de_maquina", return_value=2):
            r = client.get("/v2/maquinas/99/modelos", headers=self._auth(empresa_id=1))
        assert r.status_code == 403

    def test_get_modelo_activo_empresa_isolation(self, client):
        with patch("database_v2.repositories.obtener_empresa_id_de_maquina", return_value=2):
            r = client.get("/v2/maquinas/99/modelos/activo", headers=self._auth(empresa_id=1))
        assert r.status_code == 403

    def test_activar_empresa_isolation(self, client):
        with patch("database_v2.repositories.obtener_empresa_id_de_maquina", return_value=2):
            r = client.post("/v2/maquinas/99/modelos/5/activar", headers=self._auth(empresa_id=1))
        assert r.status_code == 403

    def test_entrenar_returns_guidance(self, client):
        with patch("database_v2.repositories.obtener_empresa_id_de_maquina", return_value=1):
            r = client.post("/v2/maquinas/1/modelos/entrenar", headers=self._auth())
        if r.status_code == 200:
            body = r.json()
            assert "action" in body
            assert body["action"] == "automatic_via_baseline_manager"


# ─── TestDashboardModelos ─────────────────────────────────────────────────────

class TestDashboardModelos:

    def _read(self):
        return open(
            os.path.join(os.path.dirname(__file__), "../src/dashboard.py"),
            encoding="utf-8",
        ).read()

    def test_modelos_tab_present(self):
        assert "🤖 Modelos ML" in self._read()

    def test_render_modelos_ml_defined(self):
        assert "def _render_modelos_ml" in self._read()

    def test_modelo_activo_section(self):
        src = self._read()
        assert "Modelo activo" in src

    def test_historial_section(self):
        src = self._read()
        assert "Historial de versiones" in src

    def test_activar_action_present(self):
        src = self._read()
        assert "Activar modelo" in src or "activar" in src.lower()

    def test_rollback_action_present(self):
        src = self._read()
        assert "rollback" in src.lower()

    def test_confirmation_required(self):
        src = self._read()
        assert "confirm_act" in src or "Confirmar" in src

    def test_dashboard_syntax_valid(self):
        import ast
        ast.parse(self._read())

    def test_api_model_endpoints_called(self):
        src = self._read()
        assert "/modelos/activo" in src
        assert "/modelos" in src


# ─── TestOfflineModels ────────────────────────────────────────────────────────

class TestOfflineModels:

    def test_validate_works_offline(self, tmp_path):
        """validate_model needs no DB — works fully offline."""
        from src.edge.anomaly.model_manager import ModelManager
        path, sha = make_valid_model(tmp_path)
        vr = ModelManager().validate_model(str(path), expected_checksum=sha)
        assert vr.valid

    def test_list_models_db_offline_returns_empty(self):
        from src.edge.anomaly.model_manager import ModelManager
        def fail_fn(mid): raise Exception("Connection refused")
        mm = ModelManager(list_fn=fail_fn)
        result = mm.list_models(42)
        assert result == []

    def test_activate_db_offline_fails_gracefully(self, tmp_path):
        from src.edge.anomaly.model_manager import ModelManager
        path, sha = make_valid_model(tmp_path)
        record = make_model_record(model_path=str(path), model_checksum=sha)
        def fail_activate(mid): raise Exception("DB offline")
        mm = ModelManager(activate_fn=fail_activate)
        mm._get_model_by_id = lambda mid: record
        result = mm.activate_model(1)
        assert not result.success
        assert "DB error" in result.error

    def test_model_store_local_works_without_supabase(self, tmp_path):
        """ModelStore is entirely local — no Supabase needed."""
        from src.edge.anomaly.model_store import ModelStore
        store = ModelStore(str(tmp_path))
        model = IsolationForest(n_estimators=5, random_state=42)
        model.fit(np.random.randn(20, 8))
        saved_path = store.save(model, maquina_id=1, version="1.50.0")
        assert saved_path.exists()
        loaded = store.load(maquina_id=1, version="1.50.0")
        assert loaded is not None
        assert hasattr(loaded, "predict")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
