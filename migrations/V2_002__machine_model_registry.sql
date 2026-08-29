-- =============================================================
-- V2_002: Machine model registry
-- Purpose: Versioned registry of ML models per machine.
--          NO BYTEA: model files are stored in Supabase Storage.
--          storage_type = 'supabase' by default.
--          Only ONE model can be active per machine (partial unique index).
-- Depends on: maquinas, empresas (existing tables)
-- =============================================================

CREATE TABLE IF NOT EXISTS machine_model_registry (
    id                  SERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    maquina_id          INTEGER     NOT NULL
                            REFERENCES maquinas(id) ON DELETE RESTRICT,
    empresa_id          INTEGER     NOT NULL
                            REFERENCES empresas(id) ON DELETE RESTRICT,

    -- Version identifier (semver recommended: '1.0.0')
    model_version       TEXT        NOT NULL,
    algorithm           TEXT        NOT NULL DEFAULT 'isolation_forest',

    -- Training metadata
    trained_at          TIMESTAMPTZ NOT NULL,
    training_samples    INTEGER     NOT NULL,
    training_from       TIMESTAMPTZ,
    training_to         TIMESTAMPTZ,
    contamination       REAL,
    features_used       TEXT[],

    -- File storage — NOT stored as BYTEA in the DB
    -- Path is relative key within Supabase Storage bucket 'aurapredict-models'
    -- Format: '{empresa_id}/{maquina_id}/{model_version}/model.joblib'
    storage_type        TEXT        NOT NULL DEFAULT 'supabase',
    model_path          TEXT        NOT NULL,

    -- Activation
    is_active           BOOLEAN     NOT NULL DEFAULT FALSE,
    notes               TEXT,

    -- Validation metrics stored as JSONB for flexibility
    performance_metrics JSONB,

    CONSTRAINT uq_model_registry_version UNIQUE (maquina_id, model_version)
);

-- Enforces at most ONE active model per machine without a trigger.
-- PostgreSQL guarantees uniqueness only over rows matching the WHERE clause.
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_registry_one_active
    ON machine_model_registry (maquina_id)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_model_registry_machine
    ON machine_model_registry (maquina_id, trained_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_registry_empresa
    ON machine_model_registry (empresa_id);
