-- =============================================================
-- V2_003: Machine baselines
-- Purpose: One row per machine. Stores statistical baseline (μ, σ,
--          percentiles) per feature as JSONB for schema flexibility.
--          Links to the active ML model.
-- Depends on: maquinas, empresas, machine_model_registry (V2_002)
-- =============================================================

CREATE TABLE IF NOT EXISTS machine_baselines (
    id                  SERIAL      PRIMARY KEY,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    maquina_id          INTEGER     NOT NULL
                            REFERENCES maquinas(id) ON DELETE RESTRICT,
    empresa_id          INTEGER     NOT NULL
                            REFERENCES empresas(id) ON DELETE RESTRICT,

    -- How many readings were used to build this baseline
    n_samples           INTEGER     NOT NULL,
    baseline_from       TIMESTAMPTZ,
    baseline_to         TIMESTAMPTZ,

    -- {feature_name: {mean, std, p5, p50, p95}} — JSONB for future feature additions
    stats_json          JSONB       NOT NULL,

    -- Which model is currently active for this machine
    active_model_id     INTEGER
                            REFERENCES machine_model_registry(id) ON DELETE SET NULL,

    -- How often (in days) baseline is recomputed
    baseline_period_days INTEGER    DEFAULT 30,

    -- One baseline record per machine
    CONSTRAINT uq_baseline_machine UNIQUE (maquina_id)
);
