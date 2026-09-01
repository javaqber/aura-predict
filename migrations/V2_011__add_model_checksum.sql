-- =============================================================
-- V2_011: Add model_checksum to machine_model_registry
--
-- Purpose: Enable SHA-256 integrity verification when uploading
--          or downloading .joblib model files to/from Supabase Storage.
--
--          Without this column there is no way to detect a corrupted
--          or truncated model file after a failed upload/download.
--
-- Design:
--   - Nullable: existing rows keep NULL (local models trained in Fase 2C
--     have no checksum yet; it is set on first upload to Storage).
--   - Populated by ModelSync.upload_model() after a successful upload.
--   - Verified by ModelSync.download_model() before replacing the
--     local active model.
--
-- Safe:    Fully additive — no existing data is modified or removed.
-- Idempotent: IF NOT EXISTS prevents errors on repeated execution.
-- Depends on: machine_model_registry (V2_002)
-- =============================================================

ALTER TABLE machine_model_registry
    ADD COLUMN IF NOT EXISTS model_checksum TEXT;

COMMENT ON COLUMN machine_model_registry.model_checksum IS
    'SHA-256 hex digest of the .joblib file stored in Supabase Storage. '
    'NULL for models that have not yet been uploaded to Storage (storage_type=''local'').';
