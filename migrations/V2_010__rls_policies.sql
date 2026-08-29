-- =============================================================
-- V2_010: Row Level Security — empresa isolation policies
--
-- HOW THE CURRENT AUTH SYSTEM WORKS:
--   - JWT tokens carry empresa_id in the payload
--   - psycopg2 connects as the postgres superuser (DATABASE_URL)
--   - Superuser connections BYPASS RLS by PostgreSQL design
--   - Therefore: existing code is unaffected by these policies
--
-- HOW RLS WILL WORK WHEN ACTIVATED:
--   - Application calls SET LOCAL app.current_empresa_id = <id>
--     at the start of each request (see rls_context.py)
--   - The current_empresa_id() function reads that setting
--   - If NULL (not set / superuser bypass): all rows visible
--   - If set: only rows matching empresa_id are visible
--
-- The policies are PERMISSIVE with a NULL fallback, so:
--   - Current service-role connections: work as before (NULL → all visible)
--   - Future non-superuser role connections with SET LOCAL: properly isolated
--
-- This is a phased approach: infrastructure ready now, enforcement later.
-- =============================================================

-- ── Helper function (idempotent) ──────────────────────────────────────────────

CREATE OR REPLACE FUNCTION current_empresa_id() RETURNS INTEGER AS $$
    SELECT NULLIF(current_setting('app.current_empresa_id', TRUE), '')::INTEGER;
$$ LANGUAGE sql STABLE;

-- ── Enable RLS on all new tables ─────────────────────────────────────────────

ALTER TABLE machine_model_registry  ENABLE ROW LEVEL SECURITY;
ALTER TABLE machine_baselines        ENABLE ROW LEVEL SECURITY;
ALTER TABLE lecturas_cnc_v2          ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_scores            ENABLE ROW LEVEL SECURITY;
ALTER TABLE raw_event_windows        ENABLE ROW LEVEL SECURITY;
ALTER TABLE maintenance_events       ENABLE ROW LEVEL SECURITY;
ALTER TABLE failure_events           ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_log                ENABLE ROW LEVEL SECURITY;

-- ── Policies — one per table (idempotent with DO block) ───────────────────────

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'machine_model_registry' AND policyname = 'ap_empresa_isolation') THEN
        CREATE POLICY ap_empresa_isolation ON machine_model_registry
            AS PERMISSIVE FOR ALL
            USING (current_empresa_id() IS NULL OR empresa_id = current_empresa_id())
            WITH CHECK (current_empresa_id() IS NULL OR empresa_id = current_empresa_id());
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'machine_baselines' AND policyname = 'ap_empresa_isolation') THEN
        CREATE POLICY ap_empresa_isolation ON machine_baselines
            AS PERMISSIVE FOR ALL
            USING (current_empresa_id() IS NULL OR empresa_id = current_empresa_id())
            WITH CHECK (current_empresa_id() IS NULL OR empresa_id = current_empresa_id());
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'lecturas_cnc_v2' AND policyname = 'ap_empresa_isolation') THEN
        CREATE POLICY ap_empresa_isolation ON lecturas_cnc_v2
            AS PERMISSIVE FOR ALL
            USING (current_empresa_id() IS NULL OR empresa_id = current_empresa_id())
            WITH CHECK (current_empresa_id() IS NULL OR empresa_id = current_empresa_id());
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'health_scores' AND policyname = 'ap_empresa_isolation') THEN
        CREATE POLICY ap_empresa_isolation ON health_scores
            AS PERMISSIVE FOR ALL
            USING (current_empresa_id() IS NULL OR empresa_id = current_empresa_id())
            WITH CHECK (current_empresa_id() IS NULL OR empresa_id = current_empresa_id());
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'raw_event_windows' AND policyname = 'ap_empresa_isolation') THEN
        CREATE POLICY ap_empresa_isolation ON raw_event_windows
            AS PERMISSIVE FOR ALL
            USING (current_empresa_id() IS NULL OR empresa_id = current_empresa_id())
            WITH CHECK (current_empresa_id() IS NULL OR empresa_id = current_empresa_id());
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'maintenance_events' AND policyname = 'ap_empresa_isolation') THEN
        CREATE POLICY ap_empresa_isolation ON maintenance_events
            AS PERMISSIVE FOR ALL
            USING (current_empresa_id() IS NULL OR empresa_id = current_empresa_id())
            WITH CHECK (current_empresa_id() IS NULL OR empresa_id = current_empresa_id());
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'failure_events' AND policyname = 'ap_empresa_isolation') THEN
        CREATE POLICY ap_empresa_isolation ON failure_events
            AS PERMISSIVE FOR ALL
            USING (current_empresa_id() IS NULL OR empresa_id = current_empresa_id())
            WITH CHECK (current_empresa_id() IS NULL OR empresa_id = current_empresa_id());
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'alert_log' AND policyname = 'ap_empresa_isolation') THEN
        CREATE POLICY ap_empresa_isolation ON alert_log
            AS PERMISSIVE FOR ALL
            USING (current_empresa_id() IS NULL OR empresa_id = current_empresa_id())
            WITH CHECK (current_empresa_id() IS NULL OR empresa_id = current_empresa_id());
    END IF;
END $$;
