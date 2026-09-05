-- =============================================================
-- V2_012: Industrial structure — Planta and Línea
--
-- Adds the organizational hierarchy:
--   Empresa → Planta → Línea → Máquina → Sensor
--
-- Design:
--   - Both tables are NULLABLE FK from maquinas (backward-compatible).
--   - Existing machines get planta_id = NULL / linea_id = NULL.
--   - The dashboard can filter by planta/linea when values are set.
--   - planta and linea each belong to a single empresa (enforced by FK).
--
-- Also adds to maquinas:
--   - planta_id, linea_id  (organizational hierarchy)
--   - activa (was: only in legacy table)
--   - notas (free text)
--
-- Safe: fully additive, no data removed, all NULLable additions.
-- Idempotent: IF NOT EXISTS on all objects.
-- =============================================================

-- ── Plantas ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS plantas (
    id              SERIAL      PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    empresa_id      INTEGER     NOT NULL
                        REFERENCES empresas(id) ON DELETE RESTRICT,

    nombre          TEXT        NOT NULL,
    descripcion     TEXT,
    ubicacion       TEXT,        -- e.g. "Polígono Industrial A, Nave 3"
    activa          BOOLEAN     NOT NULL DEFAULT TRUE,

    CONSTRAINT uq_planta_empresa_nombre UNIQUE (empresa_id, nombre)
);

CREATE INDEX IF NOT EXISTS idx_plantas_empresa
    ON plantas (empresa_id);

-- ── Líneas de producción ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lineas (
    id              SERIAL      PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    empresa_id      INTEGER     NOT NULL
                        REFERENCES empresas(id) ON DELETE RESTRICT,
    planta_id       INTEGER
                        REFERENCES plantas(id) ON DELETE SET NULL,

    nombre          TEXT        NOT NULL,
    descripcion     TEXT,
    activa          BOOLEAN     NOT NULL DEFAULT TRUE,

    CONSTRAINT uq_linea_empresa_nombre UNIQUE (empresa_id, nombre)
);

CREATE INDEX IF NOT EXISTS idx_lineas_planta
    ON lineas (planta_id);
CREATE INDEX IF NOT EXISTS idx_lineas_empresa
    ON lineas (empresa_id);

-- ── Extend maquinas with hierarchy and metadata ───────────────────────────────
ALTER TABLE maquinas
    ADD COLUMN IF NOT EXISTS planta_id  INTEGER REFERENCES plantas(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS linea_id   INTEGER REFERENCES lineas(id)  ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS activa     BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS notas      TEXT;

CREATE INDEX IF NOT EXISTS idx_maquinas_planta
    ON maquinas (planta_id) WHERE planta_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_maquinas_linea
    ON maquinas (linea_id) WHERE linea_id IS NOT NULL;

-- Comments for documentation
COMMENT ON TABLE plantas IS
    'Physical plant/factory locations. Each plant belongs to one empresa.';
COMMENT ON TABLE lineas IS
    'Production lines within a plant. Each line belongs to one empresa and optionally one plant.';
COMMENT ON COLUMN maquinas.planta_id IS
    'Organizational hierarchy: plant this machine belongs to (nullable).';
COMMENT ON COLUMN maquinas.linea_id IS
    'Organizational hierarchy: production line this machine belongs to (nullable).';
COMMENT ON COLUMN maquinas.activa IS
    'Whether this machine is currently active (default TRUE).';
