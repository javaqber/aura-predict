-- =============================================================
-- V2_004: CNC readings v2 (extended features)
-- Purpose: Main time-series table for the new CNC Condition Monitoring
--          system. Parallel to lecturas_rodamiento (NOT a replacement —
--          existing table stays untouched).
--
-- Key differences from lecturas_rodamiento:
--   - maquina_id (FK integer) instead of maquina (TEXT)
--   - empresa_id for multi-tenancy enforcement
--   - TIMESTAMPTZ instead of TEXT timestamp
--   - Full feature set: per-axis + frequency domain + order analysis
--   - sampling_rate_configured vs sampling_rate_actual (never assumed equal)
--   - rpm_nominal vs rpm_real (never substituted silently)
--   - Link to the ML model version that produced the result
--   - BIGSERIAL for high-volume time series growth
--
-- Historical data: lecturas_rodamiento is NOT migrated to this table.
--   The new historical record starts from zero when the new system is deployed.
--
-- Depends on: maquinas, empresas, machine_model_registry (V2_002)
-- =============================================================

CREATE TABLE IF NOT EXISTS lecturas_cnc_v2 (
    id                          BIGSERIAL   PRIMARY KEY,
    timestamp                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Machine and company (FK, not TEXT)
    maquina_id                  INTEGER     NOT NULL
                                    REFERENCES maquinas(id) ON DELETE RESTRICT,
    empresa_id                  INTEGER     NOT NULL
                                    REFERENCES empresas(id) ON DELETE RESTRICT,

    -- Sampling rate (configured ≠ actual — see data_quality.py)
    sampling_rate_configured    REAL        NOT NULL,
    sampling_rate_actual        REAL,
    sample_loss_fraction        REAL
                                    CHECK (sample_loss_fraction IS NULL
                                        OR sample_loss_fraction BETWEEN 0.0 AND 1.0),

    -- Operating context
    -- IMPORTANT: rpm_nominal is NEVER used silently as rpm_real
    rpm_nominal                 REAL,
    rpm_real                    REAL,
    rpm_source                  TEXT,       -- 'encoder'|'opc_ua'|'modbus'|NULL
    temperatura_c               REAL,
    carga_pct                   REAL,

    -- Time-domain features — one column per axis (X, Y, Z kept separate)
    rms_x                       REAL,
    rms_y                       REAL,
    rms_z                       REAL,
    peak_x                      REAL,
    peak_y                      REAL,
    peak_z                      REAL,
    peak_to_peak_x              REAL,
    peak_to_peak_y              REAL,
    peak_to_peak_z              REAL,
    kurtosis_x                  REAL,       -- Fisher definition (normal ≈ 0)
    kurtosis_y                  REAL,
    kurtosis_z                  REAL,
    skewness_x                  REAL,
    skewness_y                  REAL,
    skewness_z                  REAL,
    crest_factor_x              REAL,
    crest_factor_y              REAL,
    crest_factor_z              REAL,

    -- Frequency-domain features (primary analysis axis)
    dominant_freq_hz            REAL,
    dominant_amplitude          REAL,
    spectral_energy             REAL,
    band_low_energy             REAL,       -- 10–100 Hz
    band_mid_energy             REAL,       -- 100–500 Hz
    band_high_energy            REAL,       -- 500–1600 Hz

    -- Order analysis — NULL until rpm_real is available from a real source
    order_1x_energy             REAL,
    order_2x_energy             REAL,
    order_3x_energy             REAL,

    -- Data quality from data_quality.py
    signal_quality_score        REAL
                                    CHECK (signal_quality_score IS NULL
                                        OR signal_quality_score BETWEEN 0.0 AND 1.0),
    data_quality_status         TEXT,       -- 'OK'|'DEGRADED'|'POOR'|'SENSOR_ERROR'|'INVALID'

    -- Anomaly detection results
    anomaly_score               REAL,
    health_score                SMALLINT
                                    CHECK (health_score IS NULL
                                        OR health_score BETWEEN 0 AND 100),
    resultado                   TEXT        NOT NULL,
    nivel_riesgo                TEXT        NOT NULL,
    diagnostico                 TEXT        DEFAULT '',

    -- Which model version produced this result
    model_version_id            INTEGER
                                    REFERENCES machine_model_registry(id) ON DELETE SET NULL
);

-- Primary time-series query pattern: machine + time window
CREATE INDEX IF NOT EXISTS idx_lecturas_cnc_v2_machine_time
    ON lecturas_cnc_v2 (maquina_id, timestamp DESC);

-- Admin / multi-tenant queries
CREATE INDEX IF NOT EXISTS idx_lecturas_cnc_v2_empresa_time
    ON lecturas_cnc_v2 (empresa_id, timestamp DESC);

-- Anomaly investigation — only indexes anomalous rows (partial index)
CREATE INDEX IF NOT EXISTS idx_lecturas_cnc_v2_anomalias
    ON lecturas_cnc_v2 (maquina_id, timestamp DESC)
    WHERE resultado != 'OK - Sano';
