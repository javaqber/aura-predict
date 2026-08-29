-- =============================================================
-- V2_008: Failure events
-- Purpose: Record actual machine failures when they occur.
--          Critical data for:
--            - Validating AI prediction accuracy
--            - Calculating ROI (detected X days before failure)
--            - Generating supervised-learning labels for future models
--
--          primera_anomalia_ts + tiempo_deteccion_dias answer:
--          "Did the system predict this failure, and how early?"
-- Depends on: maquinas, empresas, usuarios, maintenance_events (V2_007)
-- =============================================================

CREATE TABLE IF NOT EXISTS failure_events (
    id                      SERIAL      PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    maquina_id              INTEGER     NOT NULL
                                REFERENCES maquinas(id) ON DELETE RESTRICT,
    empresa_id              INTEGER     NOT NULL
                                REFERENCES empresas(id) ON DELETE RESTRICT,

    -- When the failure occurred
    failure_at              TIMESTAMPTZ NOT NULL,

    tipo_fallo              TEXT,       -- 'bearing_outer_race'|'imbalance'|'looseness'|etc.
    componente              TEXT,
    descripcion             TEXT,

    -- Business impact
    downtime_hours          REAL,
    coste_euros             REAL,

    -- AI prediction validation
    -- Was the failure predicted? If yes, when was the first anomaly detected?
    primera_anomalia_ts     TIMESTAMPTZ,     -- timestamp of first AI anomaly alert
    tiempo_deteccion_dias   REAL,            -- (failure_at - primera_anomalia_ts) in days

    -- Link to the maintenance intervention that followed (if any)
    maintenance_event_id    INTEGER
                                REFERENCES maintenance_events(id) ON DELETE SET NULL,

    -- Audit trail
    registrado_por          INTEGER
                                REFERENCES usuarios(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_failure_machine_time
    ON failure_events (maquina_id, failure_at DESC);

CREATE INDEX IF NOT EXISTS idx_failure_empresa_time
    ON failure_events (empresa_id, failure_at DESC);
