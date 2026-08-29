-- =============================================================
-- V2_005: Health score history
-- Purpose: Time series of the Machine Health Score (0–100) per machine.
--          Enables trend visualisation and slope calculation over time.
-- Depends on: maquinas, empresas, lecturas_cnc_v2 (V2_004)
-- =============================================================

CREATE TABLE IF NOT EXISTS health_scores (
    id          BIGSERIAL   PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    maquina_id  INTEGER     NOT NULL
                    REFERENCES maquinas(id) ON DELETE RESTRICT,
    empresa_id  INTEGER     NOT NULL
                    REFERENCES empresas(id) ON DELETE RESTRICT,

    score       SMALLINT    NOT NULL
                    CHECK (score BETWEEN 0 AND 100),
    trend       TEXT,       -- 'stable'|'degrading'|'improving'|'critical'
    slope       REAL,       -- score change per day; negative = degrading

    -- The reading that generated this health score
    lectura_id  BIGINT
                    REFERENCES lecturas_cnc_v2(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_health_scores_machine_time
    ON health_scores (maquina_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_health_scores_empresa_time
    ON health_scores (empresa_id, timestamp DESC);
