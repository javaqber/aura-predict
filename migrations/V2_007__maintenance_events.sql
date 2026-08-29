-- =============================================================
-- V2_007: Maintenance events
-- Purpose: Record all maintenance interventions per machine.
--          Enables ROI calculation, AI prediction validation and
--          future supervised-learning labels.
--
--          alertado_por_ia + dias_anticipacion quantify how much
--          advance warning the system provided.
-- Depends on: maquinas, empresas, usuarios, lecturas_cnc_v2 (V2_004)
-- =============================================================

CREATE TABLE IF NOT EXISTS maintenance_events (
    id                  SERIAL      PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    maquina_id          INTEGER     NOT NULL
                            REFERENCES maquinas(id) ON DELETE RESTRICT,
    empresa_id          INTEGER     NOT NULL
                            REFERENCES empresas(id) ON DELETE RESTRICT,

    -- When the maintenance was performed
    maintenance_at      TIMESTAMPTZ NOT NULL,

    tipo                TEXT        NOT NULL
                            CHECK (tipo IN ('preventivo', 'correctivo', 'predictivo')),
    componente          TEXT,       -- e.g. 'rodamiento_husillo', 'correa'
    descripcion         TEXT,

    -- Cost and downtime
    tiempo_parada_h     REAL,
    coste_euros         REAL,
    tecnico             TEXT,

    -- AI context — was this maintenance prompted by an AuraPredict alert?
    alertado_por_ia     BOOLEAN     NOT NULL DEFAULT FALSE,
    dias_anticipacion   INTEGER,    -- days from first AI warning to this event

    -- Audit trail — SET NULL if user is later deleted (preserve history)
    registrado_por      INTEGER
                            REFERENCES usuarios(id) ON DELETE SET NULL,
    related_lectura_id  BIGINT
                            REFERENCES lecturas_cnc_v2(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_maintenance_machine_time
    ON maintenance_events (maquina_id, maintenance_at DESC);

CREATE INDEX IF NOT EXISTS idx_maintenance_empresa_time
    ON maintenance_events (empresa_id, maintenance_at DESC);
