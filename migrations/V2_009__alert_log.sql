-- =============================================================
-- V2_009: Alert log
-- Purpose: Persistent record of all alerts sent per machine.
--          Replaces the in-memory _ultimo_envio dict in alertas.py
--          which resets on every process restart (Render free tier
--          restarts every 15 min of inactivity).
--
--          Cooldown query pattern:
--            SELECT MAX(sent_at) FROM alert_log
--            WHERE maquina_id = $1 AND enviado = TRUE
--              AND sent_at > NOW() - INTERVAL '1 hour'
--
--          The existing alert logic in alertas.py continues to work
--          unchanged. alert_log is additive — it does NOT replace
--          the existing code in this phase.
-- Depends on: maquinas, empresas
-- =============================================================

CREATE TABLE IF NOT EXISTS alert_log (
    id              BIGSERIAL   PRIMARY KEY,
    sent_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    maquina_id      INTEGER     NOT NULL
                        REFERENCES maquinas(id) ON DELETE RESTRICT,
    empresa_id      INTEGER     NOT NULL
                        REFERENCES empresas(id) ON DELETE RESTRICT,

    tipo_alerta     TEXT        NOT NULL,   -- 'INFO'|'WARNING'|'CRITICAL'
    destinatario    TEXT        NOT NULL,   -- email address
    asunto          TEXT,

    -- Was the email actually delivered?
    enviado         BOOLEAN     NOT NULL DEFAULT TRUE,
    error_msg       TEXT        -- SMTP error if enviado = FALSE
);

-- Primary query: last alert for a machine (cooldown check)
CREATE INDEX IF NOT EXISTS idx_alert_machine_time
    ON alert_log (maquina_id, sent_at DESC);

CREATE INDEX IF NOT EXISTS idx_alert_empresa_time
    ON alert_log (empresa_id, sent_at DESC);
