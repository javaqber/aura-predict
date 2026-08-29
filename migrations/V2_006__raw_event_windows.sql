-- =============================================================
-- V2_006: Raw event windows
-- Purpose: Metadata for raw vibration signal windows captured by the
--          Edge when an anomaly is detected. The actual .npy file is
--          stored in Supabase Storage (bucket: 'aurapredict-raw-events').
--          Path format: '{empresa_id}/{maquina_id}/{event_ts}_{id}.npy'
--
--          Events may arrive LATER than their event_timestamp because
--          the Edge operates offline and syncs when connectivity returns.
--          created_at tracks insertion time; event_timestamp tracks
--          when the anomaly actually occurred on the Edge.
--
--          No circular FK: this table references lecturas_cnc_v2,
--          not the other way around.
-- Depends on: maquinas, empresas, lecturas_cnc_v2 (V2_004)
-- =============================================================

CREATE TABLE IF NOT EXISTS raw_event_windows (
    id                          BIGSERIAL   PRIMARY KEY,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    maquina_id                  INTEGER     NOT NULL
                                    REFERENCES maquinas(id) ON DELETE RESTRICT,
    empresa_id                  INTEGER     NOT NULL
                                    REFERENCES empresas(id) ON DELETE RESTRICT,

    -- When the anomaly actually occurred on the Edge device
    event_timestamp             TIMESTAMPTZ NOT NULL,

    -- Capture window (seconds before and after the event)
    pre_event_s                 REAL        NOT NULL,
    post_event_s                REAL        NOT NULL,
    sampling_rate_hz            REAL        NOT NULL,
    total_samples               INTEGER     NOT NULL,
    axes_captured               TEXT[]      NOT NULL,  -- e.g. ['x','y','z']

    -- File storage — Supabase Storage, not local filesystem
    storage_type                TEXT        NOT NULL DEFAULT 'supabase',
    file_path                   TEXT,                  -- storage key/path
    file_size_bytes             INTEGER,
    file_checksum               TEXT,                  -- SHA-256 for integrity

    -- Upload state (Edge → Cloud sync)
    is_uploaded                 BOOLEAN     NOT NULL DEFAULT FALSE,
    uploaded_at                 TIMESTAMPTZ,

    -- Anomaly context at the moment of capture
    anomaly_score               REAL,
    health_score_at_event       SMALLINT
                                    CHECK (health_score_at_event IS NULL
                                        OR health_score_at_event BETWEEN 0 AND 100),

    -- The reading that triggered this capture (may be NULL if offline)
    triggered_by_lectura_id     BIGINT
                                    REFERENCES lecturas_cnc_v2(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_events_machine_time
    ON raw_event_windows (maquina_id, event_timestamp DESC);

-- Partial index for offline sync queue — only unuploaded events
CREATE INDEX IF NOT EXISTS idx_raw_events_pending_upload
    ON raw_event_windows (maquina_id, created_at ASC)
    WHERE is_uploaded = FALSE;
