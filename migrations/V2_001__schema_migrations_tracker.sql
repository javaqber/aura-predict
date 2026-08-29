-- =============================================================
-- V2_001: Schema migrations tracker
-- Purpose: Track which migrations have been applied and when.
--          Used by src/database_v2/migration_runner.py
-- Safe: CREATE TABLE IF NOT EXISTS — fully idempotent
-- =============================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    id          SERIAL PRIMARY KEY,
    version     TEXT        NOT NULL UNIQUE,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum    TEXT,
    description TEXT
);
