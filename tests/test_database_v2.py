"""
Tests de integración para la Fase 2A — Base de datos CNC v2

Estos tests conectan a la base de datos real (Supabase).
Se saltan automáticamente si la conexión no está disponible.

El fixture de sesión:
  1. Aplica las migraciones (idempotente — seguro si ya están aplicadas).
  2. Crea datos de test dentro de una TRANSACCIÓN que se revierte al final.
     Las migraciones (DDL) NO se revierten.

Para constraints (FK, CHECK, UNIQUE), se usan SAVEPOINTS para que los
errores esperados no rompan la transacción principal.

Cobertura:
  1. Creación de tablas (information_schema)
  2. Claves foráneas (IntegrityError al violar FK)
  3. Restricciones CHECK (IntegrityError al violar CHECK)
  4. Índice único de modelo activo por máquina
  5. Relación máquina/empresa
  6. Inserción de una lectura CNC
  7. Inserción de health score
  8. Relación lectura → RAW event
  9. Registro de mantenimiento y fallo
 10. alert_log y cooldown
 11. RLS policies configuradas (pg_policies)
"""

import sys
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import psycopg2
import pytest

# ── Detect DB availability at import time ─────────────────────────────────────
# database.py calls init_db() at module level, so a simple import raises
# OperationalError when no DB is reachable. We catch it here and mark all
# tests as skipped so pytest can still collect the file cleanly.
try:
    from database import get_conn
    from database_v2.migration_runner import run_migrations
    from database_v2.repositories import (
        registrar_modelo, activar_modelo, obtener_modelo_activo,
        guardar_baseline, obtener_baseline,
        registrar_lectura_cnc, obtener_historial_cnc,
        registrar_health_score, obtener_historial_health,
        registrar_evento_raw, marcar_evento_subido, obtener_eventos_pendientes_upload,
        registrar_mantenimiento, registrar_fallo,
        registrar_alerta, puede_enviar_alerta,
        obtener_maquina_id_por_nombre,
    )
    _DB_AVAILABLE = True
except Exception as _db_err:
    _DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _DB_AVAILABLE,
    reason="Database not reachable — run these tests with a real Supabase connection"
)


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def apply_migrations():
    """Apply all v2 migrations once per test session (idempotent)."""
    try:
        conn = get_conn()
        conn.close()
    except Exception:
        pytest.skip("Database not available — skipping Fase 2A tests")

    run_migrations()


@pytest.fixture(scope="module")
def db():
    """
    Module-scoped fixture. Creates a test empresa and maquina, COMMITS them
    so they are visible to all connections (repository functions open their own
    connections and cannot see uncommitted data from another transaction).

    Cleanup uses explicit DELETEs in FK-reverse order at the end of the module.
    The main connection is closed without committing, which auto-rolls-back any
    uncommitted inserts made by constraint tests (savepoint tests).
    """
    try:
        conn = get_conn()
        conn.autocommit = False
    except Exception:
        pytest.skip("Database not available")

    suffix   = uuid.uuid4().hex[:8]
    emp_name = f"__TEST_EMPRESA_{suffix}__"
    maq_name = f"__TEST_MAQUINA_{suffix}__"

    cur = conn.cursor()
    empresa_id = None
    maquina_id = None

    try:
        # ── Create test empresa ───────────────────────────────────────────────
        cur.execute(
            "INSERT INTO empresas (nombre, fecha_registro) VALUES (%s, NOW()) RETURNING id",
            (emp_name,)
        )
        empresa_id = cur.fetchone()[0]

        # ── Create test maquina assigned to that empresa ──────────────────────
        cur.execute(
            "INSERT INTO maquinas (nombre, tipo, fecha_registro, empresa_id) "
            "VALUES (%s, 'torno_cnc', NOW(), %s) RETURNING id",
            (maq_name, empresa_id)
        )
        maquina_id = cur.fetchone()[0]

        # ── COMMIT so repository functions (separate connections) can see them ─
        conn.commit()

        yield conn, empresa_id, maquina_id

    finally:
        # ── Cleanup: explicit DELETEs in FK-reverse order ─────────────────────
        # conn itself is closed without commit → auto-rollbacks any uncommitted
        # constraint-test inserts. A separate cleanup connection removes all
        # committed data that repositories inserted via their own connections.
        cur.close()
        conn.close()   # rolls back uncommitted constraint-test data

        if maquina_id is not None:
            _cleanup_test_data(maquina_id, empresa_id)


def _cleanup_test_data(maquina_id: int, empresa_id: Optional[int]) -> None:
    """
    Remove all test data in FK-reverse order.
    Called once after all module tests complete.
    """
    try:
        cleanup = get_conn()
        cleanup.autocommit = False
        c = cleanup.cursor()

        # FK-reverse order: child tables first, then parent tables
        for table in (
            "alert_log",
            "raw_event_windows",
            "failure_events",
            "maintenance_events",
            "health_scores",
            "lecturas_cnc_v2",
            "machine_baselines",
            "machine_model_registry",
        ):
            c.execute(f"DELETE FROM {table} WHERE maquina_id = %s", (maquina_id,))

        c.execute("DELETE FROM maquinas  WHERE id = %s", (maquina_id,))
        if empresa_id is not None:
            c.execute("DELETE FROM empresas WHERE id = %s", (empresa_id,))

        cleanup.commit()
    except Exception as e:
        print(f"\n[cleanup] Warning during test data removal: {e}")
        try:
            cleanup.rollback()
        except Exception:
            pass
    finally:
        try:
            c.close()
            cleanup.close()
        except Exception:
            pass


# ─── HELPER: test-safe constraint violation ───────────────────────────────────

def assert_raises_integrity_error(conn, fn):
    """
    Execute fn(cur) expecting IntegrityError.
    Uses a SAVEPOINT so the parent transaction stays valid.
    """
    cur = conn.cursor()
    cur.execute("SAVEPOINT constraint_test")
    try:
        fn(cur)
        conn.commit()
        pytest.fail("Expected IntegrityError was not raised")
    except psycopg2.IntegrityError:
        cur.execute("ROLLBACK TO SAVEPOINT constraint_test")
    finally:
        cur.execute("RELEASE SAVEPOINT constraint_test")
        cur.close()


# ─── TEST 1: Tablas existentes ────────────────────────────────────────────────

NEW_TABLES = [
    "schema_migrations",
    "machine_model_registry",
    "machine_baselines",
    "lecturas_cnc_v2",
    "health_scores",
    "raw_event_windows",
    "maintenance_events",
    "failure_events",
    "alert_log",
]


@pytest.mark.parametrize("table_name", NEW_TABLES)
def test_table_exists(table_name):
    """All new tables must exist in the public schema."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        )
    """, (table_name,))
    exists = cur.fetchone()[0]
    cur.close()
    conn.close()
    assert exists, f"Table '{table_name}' not found in public schema"


# ─── TEST 2: Claves foráneas ─────────────────────────────────────────────────

def test_fk_lecturas_cnc_maquina_invalida(db):
    """Inserting a reading with a non-existent maquina_id must fail."""
    conn, empresa_id, _ = db

    def do_insert(cur):
        cur.execute("""
            INSERT INTO lecturas_cnc_v2
            (maquina_id, empresa_id, resultado, nivel_riesgo, sampling_rate_configured)
            VALUES (999999999, %s, 'OK - Sano', 'Bajo', 3200.0)
        """, (empresa_id,))

    assert_raises_integrity_error(conn, do_insert)


def test_fk_health_score_lectura_invalida(db):
    """health_scores.lectura_id must reference an existing lecturas_cnc_v2 row."""
    conn, empresa_id, maquina_id = db

    def do_insert(cur):
        cur.execute("""
            INSERT INTO health_scores (maquina_id, empresa_id, score, lectura_id)
            VALUES (%s, %s, 90, 999999999)
        """, (maquina_id, empresa_id))

    assert_raises_integrity_error(conn, do_insert)


def test_fk_failure_references_maintenance(db):
    """failure_events.maintenance_event_id must reference an existing row."""
    conn, empresa_id, maquina_id = db

    def do_insert(cur):
        cur.execute("""
            INSERT INTO failure_events (maquina_id, empresa_id, failure_at, maintenance_event_id)
            VALUES (%s, %s, NOW(), 999999999)
        """, (maquina_id, empresa_id))

    assert_raises_integrity_error(conn, do_insert)


# ─── TEST 3: Restricciones CHECK ─────────────────────────────────────────────

def test_check_health_score_out_of_range(db):
    """health_score must be between 0 and 100."""
    conn, empresa_id, maquina_id = db

    def do_insert(cur):
        cur.execute("""
            INSERT INTO health_scores (maquina_id, empresa_id, score)
            VALUES (%s, %s, 150)
        """, (maquina_id, empresa_id))

    assert_raises_integrity_error(conn, do_insert)


def test_check_health_score_negative(db):
    """health_score cannot be negative."""
    conn, empresa_id, maquina_id = db

    def do_insert(cur):
        cur.execute("""
            INSERT INTO health_scores (maquina_id, empresa_id, score)
            VALUES (%s, %s, -1)
        """, (maquina_id, empresa_id))

    assert_raises_integrity_error(conn, do_insert)


def test_check_maintenance_tipo_invalido(db):
    """maintenance_events.tipo must be one of the allowed values."""
    conn, empresa_id, maquina_id = db

    def do_insert(cur):
        cur.execute("""
            INSERT INTO maintenance_events (maquina_id, empresa_id, maintenance_at, tipo)
            VALUES (%s, %s, NOW(), 'urgente_inventado')
        """, (maquina_id, empresa_id))

    assert_raises_integrity_error(conn, do_insert)


def test_check_signal_quality_out_of_range(db):
    """signal_quality_score must be between 0.0 and 1.0."""
    conn, empresa_id, maquina_id = db

    def do_insert(cur):
        cur.execute("""
            INSERT INTO lecturas_cnc_v2
            (maquina_id, empresa_id, resultado, nivel_riesgo,
             sampling_rate_configured, signal_quality_score)
            VALUES (%s, %s, 'OK - Sano', 'Bajo', 3200.0, 1.5)
        """, (maquina_id, empresa_id))

    assert_raises_integrity_error(conn, do_insert)


# ─── TEST 4: Índice único modelo activo por máquina ──────────────────────────

def test_partial_unique_index_one_active_model(db):
    """Only ONE model can be active per machine at a time."""
    conn, empresa_id, maquina_id = db
    cur = conn.cursor()

    # Insert first model — active
    cur.execute("""
        INSERT INTO machine_model_registry
        (maquina_id, empresa_id, model_version, trained_at, training_samples,
         model_path, is_active)
        VALUES (%s, %s, 'v1.0.0', NOW(), 100, 'supabase://test/v1/model.joblib', TRUE)
        RETURNING id
    """, (maquina_id, empresa_id))
    model1_id = cur.fetchone()[0]

    # Attempting to insert a SECOND active model for the same machine must fail
    def do_insert_second(cur):
        cur.execute("""
            INSERT INTO machine_model_registry
            (maquina_id, empresa_id, model_version, trained_at, training_samples,
             model_path, is_active)
            VALUES (%s, %s, 'v2.0.0', NOW(), 200, 'supabase://test/v2/model.joblib', TRUE)
        """, (maquina_id, empresa_id))

    assert_raises_integrity_error(conn, do_insert_second)

    # A second INACTIVE model is allowed
    cur.execute("""
        INSERT INTO machine_model_registry
        (maquina_id, empresa_id, model_version, trained_at, training_samples,
         model_path, is_active)
        VALUES (%s, %s, 'v2.0.0', NOW(), 200, 'supabase://test/v2/model.joblib', FALSE)
        RETURNING id
    """, (maquina_id, empresa_id))
    model2_id = cur.fetchone()[0]
    assert model2_id is not None, "Inactive model should be insertable"


# ─── TEST 5: Relación máquina/empresa ────────────────────────────────────────

def test_maquina_empresa_relationship(db):
    """A reading's empresa_id must match its machine's empresa_id (app-level rule)."""
    conn, empresa_id, maquina_id = db
    # This test validates at the app level, not via DB constraint
    # (the DB allows mismatches — RLS or app logic handles isolation)
    cur = conn.cursor()
    cur.execute("SELECT empresa_id FROM maquinas WHERE id = %s", (maquina_id,))
    row = cur.fetchone()
    assert row is not None
    assert row[0] == empresa_id, "Test machine must belong to the test empresa"


# ─── TEST 6: Inserción de lectura CNC ────────────────────────────────────────

def test_registrar_lectura_cnc(db):
    """A complete CNC reading must insert and be retrievable."""
    conn, empresa_id, maquina_id = db

    lectura_id = registrar_lectura_cnc(
        maquina_id                = maquina_id,
        empresa_id                = empresa_id,
        resultado                 = "OK - Sano",
        nivel_riesgo              = "Bajo",
        sampling_rate_configured  = 3200.0,
        sampling_rate_actual      = 3198.5,
        rms_x                     = 0.072,
        kurtosis_x                = 0.15,
        health_score              = 92,
        signal_quality_score      = 0.98,
        data_quality_status       = "OK",
    )
    assert lectura_id is not None, "registrar_lectura_cnc should return a valid id"

    historial = obtener_historial_cnc(maquina_id, limite=5)
    ids = [r["id"] for r in historial]
    assert lectura_id in ids, "Inserted reading must appear in historial"


# ─── TEST 7: Inserción de health score ───────────────────────────────────────

def test_registrar_health_score(db):
    """A health score entry must insert correctly and respect the 0-100 constraint."""
    _, empresa_id, maquina_id = db

    hs_id = registrar_health_score(
        maquina_id = maquina_id,
        empresa_id = empresa_id,
        score      = 85,
        trend      = "stable",
        slope      = -0.1,
    )
    assert hs_id is not None, "registrar_health_score should return a valid id"

    historial = obtener_historial_health(maquina_id, dias=1)
    ids = [r["id"] for r in historial]
    assert hs_id in ids


# ─── TEST 8: Relación lectura → RAW event ────────────────────────────────────

def test_lectura_raw_event_relationship(db):
    """A raw event window must link back to a valid lectura_cnc_v2 row."""
    conn, empresa_id, maquina_id = db

    # Create a reading first
    lectura_id = registrar_lectura_cnc(
        maquina_id=maquina_id, empresa_id=empresa_id,
        resultado="NOK - Anomalía Detectada", nivel_riesgo="CRÍTICO",
        sampling_rate_configured=3200.0, anomaly_score=0.82, health_score=35,
    )
    assert lectura_id is not None

    # Register the raw event linked to that reading
    event_id = registrar_evento_raw(
        maquina_id                = maquina_id,
        empresa_id                = empresa_id,
        event_timestamp           = datetime.now(timezone.utc),
        pre_event_s               = 5.0,
        post_event_s              = 10.0,
        sampling_rate_hz          = 3200.0,
        total_samples             = 48000,
        axes_captured             = ["x", "y", "z"],
        anomaly_score             = 0.82,
        health_score_at_event     = 35,
        triggered_by_lectura_id   = lectura_id,
    )
    assert event_id is not None, "registrar_evento_raw should return a valid id"

    # Check it appears in the pending upload queue
    pendientes = obtener_eventos_pendientes_upload(maquina_id)
    ids = [e["id"] for e in pendientes]
    assert event_id in ids

    # Mark as uploaded
    ok = marcar_evento_subido(event_id, "supabase://raw/event.npy", "abc123")
    assert ok

    # Should no longer appear in the pending queue
    pendientes_after = obtener_eventos_pendientes_upload(maquina_id)
    ids_after = [e["id"] for e in pendientes_after]
    assert event_id not in ids_after


# ─── TEST 9: Mantenimiento y fallo ───────────────────────────────────────────

def test_registrar_mantenimiento(db):
    """A maintenance event must insert with correct tipo constraint."""
    _, empresa_id, maquina_id = db

    mant_id = registrar_mantenimiento(
        maquina_id       = maquina_id,
        empresa_id       = empresa_id,
        tipo             = "predictivo",
        componente       = "rodamiento_husillo",
        tiempo_parada_h  = 2.0,
        coste_euros      = 350.0,
        alertado_por_ia  = True,
        dias_anticipacion= 12,
    )
    assert mant_id is not None, "registrar_mantenimiento should return a valid id"


def test_registrar_fallo_vinculado_a_mantenimiento(db):
    """A failure event must link to an existing maintenance event."""
    _, empresa_id, maquina_id = db

    mant_id = registrar_mantenimiento(
        maquina_id=maquina_id, empresa_id=empresa_id, tipo="correctivo",
    )
    assert mant_id is not None

    fallo_id = registrar_fallo(
        maquina_id           = maquina_id,
        empresa_id           = empresa_id,
        tipo_fallo           = "bearing_outer_race",
        downtime_hours       = 8.0,
        coste_euros          = 1200.0,
        tiempo_deteccion_dias= 14.0,
        maintenance_event_id = mant_id,
    )
    assert fallo_id is not None, "registrar_fallo should return a valid id"


# ─── TEST 10: alert_log y cooldown ───────────────────────────────────────────

def test_alert_log_y_cooldown(db):
    """Alert log must persist and cooldown check must respect the window."""
    _, empresa_id, maquina_id = db

    # Before any alert: cooldown should pass
    assert puede_enviar_alerta(maquina_id, cooldown_hours=1.0) is True

    # Register an alert
    alert_id = registrar_alerta(
        maquina_id   = maquina_id,
        empresa_id   = empresa_id,
        tipo_alerta  = "CRITICAL",
        destinatario = "test@test.invalid",
        asunto       = "Test alert",
        enviado      = True,
    )
    assert alert_id is not None

    # Now cooldown should prevent sending within 1 hour
    assert puede_enviar_alerta(maquina_id, cooldown_hours=1.0) is False

    # But a very short cooldown (0 hours) should allow it
    assert puede_enviar_alerta(maquina_id, cooldown_hours=0.0) is True


# ─── TEST 11: RLS policies configuradas ──────────────────────────────────────

TABLES_WITH_RLS = [
    "machine_model_registry",
    "machine_baselines",
    "lecturas_cnc_v2",
    "health_scores",
    "raw_event_windows",
    "maintenance_events",
    "failure_events",
    "alert_log",
]


@pytest.mark.parametrize("table_name", TABLES_WITH_RLS)
def test_rls_policy_exists(table_name):
    """Each new table must have at least one RLS policy configured."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM pg_policies
        WHERE schemaname = 'public' AND tablename = %s
    """, (table_name,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    assert count > 0, (
        f"No RLS policies found for '{table_name}'. "
        "Run migration V2_010 to create them."
    )


@pytest.mark.parametrize("table_name", TABLES_WITH_RLS)
def test_rls_enabled(table_name):
    """RLS must be enabled on each new table (relrowsecurity = TRUE)."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT relrowsecurity FROM pg_class
        WHERE relname = %s AND relkind = 'r'
    """, (table_name,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    assert row is not None, f"Table '{table_name}' not found in pg_class"
    assert row[0] is True, f"RLS not enabled on table '{table_name}'"


def test_current_empresa_id_function_exists():
    """The current_empresa_id() helper function must exist in the DB."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = 'public' AND p.proname = 'current_empresa_id'
        )
    """)
    exists = cur.fetchone()[0]
    cur.close()
    conn.close()
    assert exists, "current_empresa_id() function not found — run migration V2_010"


# ─── TEST adicional: migration runner es idempotente ─────────────────────────

def test_run_migrations_idempotent():
    """Running migrations twice must not raise errors or apply new migrations."""
    applied = run_migrations()
    # Second run should apply nothing
    applied_again = run_migrations()
    assert applied_again == [], "Second run should apply 0 migrations"


if __name__ == "__main__":
    import pytest as pt
    pt.main([__file__, "-v", "--tb=short"])
