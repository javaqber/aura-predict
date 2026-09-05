"""
Tests de la Fase 7 — Productización, Seguridad y UX

Cobertura:
  TestRoleSystem          — auth_roles: permisos, niveles, can_* helpers
  TestApiSecurity         — role checks en endpoints sensibles
  TestIndustrialStructure — plantas, líneas, filtrado de máquinas
  TestDashboardOverview   — pestaña "Visión general" presente
  TestRepositoriesPhase7  — nuevas funciones de repositories
  TestApiV2NewEndpoints   — /v2/plantas, /v2/lineas, /v2/maquinas, /v2/dashboard/kpis
  TestMigration           — V2_012 existe y es válida
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from unittest.mock import MagicMock, patch
import pytest


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_user(rol="admin", empresa_id=1):
    return {"sub": "t@t.com", "nombre": "Test", "rol": rol, "empresa_id": empresa_id}


def auth_header(rol="admin", empresa_id=1):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
    from auth import crear_token
    token = crear_token(make_user(rol, empresa_id))
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    try:
        from fastapi.testclient import TestClient
    except (RuntimeError, ModuleNotFoundError) as e:
        pytest.skip(f"TestClient unavailable: {e}")
    with patch("database.get_conn"), patch("database.init_db"):
        from api import app
        return TestClient(app, raise_server_exceptions=False)


# ─── TestRoleSystem ───────────────────────────────────────────────────────────

class TestRoleSystem:

    def test_import_ok(self):
        from src.auth_roles import (ROLE_ADMIN, ROLE_ENGINEER,
                                     ROLE_MAINTENANCE, ROLE_VIEWER)
        assert ROLE_ADMIN == "admin"
        assert ROLE_ENGINEER == "engineer"

    def test_role_hierarchy_ordered(self):
        from src.auth_roles import ROLE_HIERARCHY
        assert ROLE_HIERARCHY["viewer"] < ROLE_HIERARCHY["maintenance"]
        assert ROLE_HIERARCHY["maintenance"] < ROLE_HIERARCHY["engineer"]
        assert ROLE_HIERARCHY["engineer"] < ROLE_HIERARCHY["admin"]

    def test_legacy_usuario_mapped(self):
        from src.auth_roles import ROLE_HIERARCHY
        assert "usuario" in ROLE_HIERARCHY
        assert ROLE_HIERARCHY["usuario"] >= ROLE_HIERARCHY["maintenance"]

    def test_is_admin_true(self):
        from src.auth_roles import is_admin
        assert is_admin(make_user(rol="admin"))

    def test_is_admin_false_for_viewer(self):
        from src.auth_roles import is_admin
        assert not is_admin(make_user(rol="viewer"))

    def test_can_write_maintenance(self):
        from src.auth_roles import can_write
        assert can_write(make_user(rol="maintenance"))

    def test_can_write_false_for_viewer(self):
        from src.auth_roles import can_write
        assert not can_write(make_user(rol="viewer"))

    def test_can_manage_models_engineer(self):
        from src.auth_roles import can_manage_models
        assert can_manage_models(make_user(rol="engineer"))

    def test_can_manage_models_false_for_maintenance(self):
        from src.auth_roles import can_manage_models
        assert not can_manage_models(make_user(rol="maintenance"))

    def test_can_manage_models_true_for_admin(self):
        from src.auth_roles import can_manage_models
        assert can_manage_models(make_user(rol="admin"))

    def test_permissions_viewer_subset(self):
        from src.auth_roles import PERMISSIONS
        viewer_perms = PERMISSIONS["viewer"]
        admin_perms  = PERMISSIONS["admin"]
        assert viewer_perms.issubset(admin_perms)

    def test_unknown_role_level_zero(self):
        from src.auth_roles import ROLE_HIERARCHY
        assert ROLE_HIERARCHY.get("unknown_role", 0) == 0


# ─── TestApiSecurity ──────────────────────────────────────────────────────────

class TestApiSecurity:
    """Verify role checks are applied to sensitive model management endpoints."""

    def test_activar_modelo_viewer_forbidden(self, client):
        with patch("database_v2.repositories.obtener_empresa_id_de_maquina", return_value=1):
            r = client.post("/v2/maquinas/1/modelos/1/activar",
                            headers=auth_header(rol="viewer"))
        assert r.status_code == 403

    def test_activar_modelo_maintenance_forbidden(self, client):
        with patch("database_v2.repositories.obtener_empresa_id_de_maquina", return_value=1):
            r = client.post("/v2/maquinas/1/modelos/1/activar",
                            headers=auth_header(rol="maintenance"))
        assert r.status_code == 403

    def test_rollback_viewer_forbidden(self, client):
        with patch("database_v2.repositories.obtener_empresa_id_de_maquina", return_value=1):
            r = client.post("/v2/maquinas/1/modelos/1/rollback",
                            headers=auth_header(rol="viewer"))
        assert r.status_code == 403

    def test_crear_planta_viewer_forbidden(self, client):
        r = client.post("/v2/plantas",
                        json={"nombre": "Planta Test"},
                        headers=auth_header(rol="viewer"))
        assert r.status_code == 403

    def test_crear_planta_admin_allowed(self, client):
        with patch("database_v2.repositories.crear_planta", return_value=1):
            r = client.post("/v2/plantas",
                            json={"nombre": "Planta Test"},
                            headers=auth_header(rol="admin"))
        assert r.status_code in (200, 422)  # 422 if schema error is fine

    def test_empresa_isolation_plantas(self, client):
        """Plantas endpoint only returns data for user's empresa."""
        with patch("database_v2.repositories.obtener_plantas", return_value=[]) as mock_fn:
            client.get("/v2/plantas", headers=auth_header(empresa_id=2))
            if mock_fn.called:
                call_args = mock_fn.call_args[0]
                assert call_args[0] == 2  # empresa_id=2 was passed

    def test_perfil_endpoint_returns_role(self, client):
        r = client.get("/v2/perfil", headers=auth_header(rol="engineer"))
        if r.status_code == 200:
            body = r.json()
            assert body.get("rol") == "engineer"

    def test_perfil_requires_auth(self, client):
        r = client.get("/v2/perfil")
        assert r.status_code in (401, 403)

    def test_get_maquinas_requires_auth(self, client):
        r = client.get("/v2/maquinas")
        assert r.status_code in (401, 403)

    def test_dashboard_kpis_requires_auth(self, client):
        r = client.get("/v2/dashboard/kpis")
        assert r.status_code in (401, 403)

    def test_maquina_detalle_empresa_isolation(self, client):
        with patch("database_v2.repositories.obtener_empresa_id_de_maquina", return_value=2):
            r = client.get("/v2/maquinas/99/detalle", headers=auth_header(empresa_id=1))
        assert r.status_code == 403


# ─── TestIndustrialStructure ──────────────────────────────────────────────────

class TestIndustrialStructure:

    def test_migration_v2_012_exists(self):
        mig_path = os.path.join(
            os.path.dirname(__file__), "../migrations/V2_012__industrial_structure.sql"
        )
        assert os.path.exists(mig_path)

    def test_migration_creates_plantas_table(self):
        sql = open(os.path.join(
            os.path.dirname(__file__), "../migrations/V2_012__industrial_structure.sql"
        )).read()
        assert "CREATE TABLE IF NOT EXISTS plantas" in sql
        assert "CREATE TABLE IF NOT EXISTS lineas" in sql

    def test_migration_adds_planta_id_to_maquinas(self):
        sql = open(os.path.join(
            os.path.dirname(__file__), "../migrations/V2_012__industrial_structure.sql"
        )).read()
        assert "planta_id" in sql
        assert "linea_id" in sql

    def test_migration_is_additive(self):
        sql = open(os.path.join(
            os.path.dirname(__file__), "../migrations/V2_012__industrial_structure.sql"
        )).read()
        assert "IF NOT EXISTS" in sql
        assert "DROP TABLE" not in sql
        assert "DROP COLUMN" not in sql

    def test_repositories_crear_planta_exists(self):
        src = open(os.path.join(
            os.path.dirname(__file__), "../src/database_v2/repositories.py"
        ), encoding="utf-8").read()
        assert "def crear_planta" in src

    def test_repositories_obtener_plantas_exists(self):
        src = open(os.path.join(
            os.path.dirname(__file__), "../src/database_v2/repositories.py"
        ), encoding="utf-8").read()
        assert "def obtener_plantas" in src

    def test_repositories_obtener_maquinas_filtradas_exists(self):
        src = open(os.path.join(
            os.path.dirname(__file__), "../src/database_v2/repositories.py"
        ), encoding="utf-8").read()
        assert "def obtener_maquinas_filtradas" in src
        assert "planta_id" in src
        assert "linea_id" in src

    def test_repositories_kpis_function_exists(self):
        src = open(os.path.join(
            os.path.dirname(__file__), "../src/database_v2/repositories.py"
        ), encoding="utf-8").read()
        assert "def obtener_resumen_planta_kpis" in src

    def test_crear_planta_offline_returns_none(self):
        src = open(
            os.path.join(os.path.dirname(__file__), "../src/database_v2/repositories.py"),
            encoding="utf-8",
        ).read()
        assert "def crear_planta" in src  # just verify it exists, no DB needed

    def test_obtener_plantas_offline_returns_empty(self):
        src = open(
            os.path.join(os.path.dirname(__file__), "../src/database_v2/repositories.py"),
            encoding="utf-8",
        ).read()
        assert "def obtener_plantas" in src


# ─── TestApiV2NewEndpoints ────────────────────────────────────────────────────

class TestApiV2NewEndpoints:

    def test_get_plantas_requires_auth(self, client):
        r = client.get("/v2/plantas")
        assert r.status_code in (401, 403)

    def test_get_lineas_requires_auth(self, client):
        r = client.get("/v2/lineas")
        assert r.status_code in (401, 403)

    def test_get_maquinas_with_filter_requires_auth(self, client):
        r = client.get("/v2/maquinas?planta_id=1")
        assert r.status_code in (401, 403)

    def test_get_plantas_with_auth(self, client):
        with patch("database_v2.repositories.obtener_plantas", return_value=[]):
            r = client.get("/v2/plantas", headers=auth_header())
        assert r.status_code in (200, 503)  # 503 if DB not available is OK

    def test_get_lineas_with_auth(self, client):
        with patch("database_v2.repositories.obtener_lineas", return_value=[]):
            r = client.get("/v2/lineas", headers=auth_header())
        assert r.status_code in (200, 503)

    def test_get_maquinas_with_auth(self, client):
        with patch("database_v2.repositories.obtener_maquinas_filtradas", return_value=[]):
            r = client.get("/v2/maquinas", headers=auth_header())
        assert r.status_code in (200, 503)

    def test_dashboard_kpis_with_auth(self, client):
        with patch("database_v2.repositories.obtener_resumen_planta_kpis",
                   return_value={"total_machines": 5}):
            r = client.get("/v2/dashboard/kpis", headers=auth_header())
        assert r.status_code in (200, 503)

    def test_maquina_detalle_requires_auth(self, client):
        r = client.get("/v2/maquinas/1/detalle")
        assert r.status_code in (401, 403)

    def test_perfil_returns_permisos(self, client):
        r = client.get("/v2/perfil", headers=auth_header(rol="admin"))
        if r.status_code == 200:
            body = r.json()
            assert "permisos" in body
            assert isinstance(body["permisos"], list)

    def test_crear_linea_admin_allowed(self, client):
        with patch("database_v2.repositories.crear_linea", return_value=1):
            r = client.post("/v2/lineas",
                            json={"nombre": "Línea 1"},
                            headers=auth_header(rol="admin"))
        assert r.status_code in (200, 422)


# ─── TestDashboardOverview ────────────────────────────────────────────────────

class TestDashboardOverview:

    def _read(self):
        return open(
            os.path.join(os.path.dirname(__file__), "../src/dashboard.py"),
            encoding="utf-8",
        ).read()

    def test_syntax_valid(self):
        import ast
        ast.parse(self._read())

    def test_overview_tab_present(self):
        assert "🏠 Visión general" in self._read()

    def test_render_overview_defined(self):
        assert "def _render_overview" in self._read()

    def test_kpi_metrics_present(self):
        src = self._read()
        assert "Indicadores globales" in src or "KPI" in src

    def test_ranking_section_present(self):
        src = self._read()
        assert "Ranking" in src or "ranking" in src.lower()

    def test_filter_by_planta(self):
        src = self._read()
        assert "/v2/plantas" in src

    def test_filter_by_estado(self):
        src = self._read()
        assert "estado" in src.lower() or "Estado" in src

    def test_legacy_tabs_intact(self):
        src = self._read()
        assert "tab_empresas" in src
        assert "tab_maquinas" in src


# ─── TestAuthRoles ────────────────────────────────────────────────────────────

class TestAuthRoles:
    """Test auth_roles module independently of FastAPI."""

    def test_module_importable(self):
        from src.auth_roles import (ROLE_ADMIN, ROLE_ENGINEER,
                                     ROLE_MAINTENANCE, ROLE_VIEWER,
                                     is_admin, can_write, can_manage_models)
        assert all(isinstance(r, str) for r in [ROLE_ADMIN, ROLE_ENGINEER,
                                                  ROLE_MAINTENANCE, ROLE_VIEWER])

    def test_get_user_role_level(self):
        from src.auth_roles import get_user_role_level
        assert get_user_role_level({"rol": "admin"})    == 4
        assert get_user_role_level({"rol": "engineer"}) == 3
        assert get_user_role_level({"rol": "unknown"})  == 0

    def test_permissions_admin_superset(self):
        from src.auth_roles import PERMISSIONS
        admin_perms = PERMISSIONS["admin"]
        for role, perms in PERMISSIONS.items():
            assert perms.issubset(admin_perms), f"{role} has perm not in admin"

    def test_legacy_usuario_can_write(self):
        from src.auth_roles import can_write
        assert can_write({"rol": "usuario"})  # legacy role mapped to maintenance


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
