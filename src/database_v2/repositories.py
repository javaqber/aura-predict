"""
AuraPredict — Repositories v2
================================
CRUD functions for all new CNC Condition Monitoring tables.
Follows the same coding style as src/database.py — simple functions
that manage their own connections.

During the coexistence period with the legacy system, maquina_id is
resolved from the machine name using obtener_maquina_id_por_nombre().
The Edge scheduler resolves the ID on startup and caches it locally.

NO changes are made to existing tables (lecturas_rodamiento, maquinas,
empresas, usuarios, lecturas_prensa). All functions here are additive.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from database import get_conn


# ─── MACHINE ID RESOLUTION ────────────────────────────────────────────────────

def obtener_maquina_id_por_nombre(nombre: str) -> Optional[int]:
    """
    Resolve machine name to maquina_id (INTEGER PK).

    Used during the coexistence period with the legacy system, where
    the Edge still has a machine name in its config (not yet maquina_id).
    The Edge scheduler calls this once on startup and caches the result.

    Returns None if the machine is not found or has no empresa assigned.
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT id FROM maquinas WHERE nombre = %s",
            (nombre,)
        )
        row = cur.fetchone()
        return int(row[0]) if row else None
    finally:
        cur.close()
        conn.close()


def obtener_empresa_id_de_maquina(maquina_id: int) -> Optional[int]:
    """Return the empresa_id for a given maquina_id."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT empresa_id FROM maquinas WHERE id = %s",
            (maquina_id,)
        )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None
    finally:
        cur.close()
        conn.close()


# ─── MACHINE MODEL REGISTRY ───────────────────────────────────────────────────

def registrar_modelo(
    maquina_id:         int,
    empresa_id:         int,
    model_version:      str,
    trained_at:         datetime,
    training_samples:   int,
    model_path:         str,
    algorithm:          str = "isolation_forest",
    training_from:      Optional[datetime] = None,
    training_to:        Optional[datetime] = None,
    contamination:      Optional[float] = None,
    features_used:      Optional[list[str]] = None,
    storage_type:       str = "supabase",
    notes:              Optional[str] = None,
    performance_metrics: Optional[dict] = None,
) -> Optional[int]:
    """Register a new ML model version. Returns the new model id."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO machine_model_registry
            (maquina_id, empresa_id, model_version, algorithm,
             trained_at, training_samples, training_from, training_to,
             contamination, features_used, storage_type, model_path,
             notes, performance_metrics)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            maquina_id, empresa_id, model_version, algorithm,
            trained_at, training_samples, training_from, training_to,
            contamination, features_used, storage_type, model_path,
            notes,
            json.dumps(performance_metrics) if performance_metrics else None,
        ))
        model_id = cur.fetchone()[0]
        conn.commit()
        return model_id
    except psycopg2.IntegrityError:
        conn.rollback()
        return None
    finally:
        cur.close()
        conn.close()


def activar_modelo(model_id: int) -> bool:
    """
    Set is_active=TRUE for model_id and FALSE for all others of the same machine.
    The partial unique index (idx_model_registry_one_active) enforces
    that only ONE active model exists per machine at the database level.
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        # Get maquina_id first
        cur.execute(
            "SELECT maquina_id FROM machine_model_registry WHERE id = %s",
            (model_id,)
        )
        row = cur.fetchone()
        if not row:
            return False
        maquina_id = row[0]

        # Deactivate all models for this machine, then activate the target
        cur.execute(
            "UPDATE machine_model_registry SET is_active = FALSE WHERE maquina_id = %s",
            (maquina_id,)
        )
        cur.execute(
            "UPDATE machine_model_registry SET is_active = TRUE WHERE id = %s",
            (model_id,)
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def obtener_modelo_activo(maquina_id: int) -> Optional[dict]:
    """
    Return the active model record for a machine, or None.
    Fase 5B: extended to include features_used, model_checksum, empresa_id, is_active.
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT id, maquina_id, empresa_id, model_version, algorithm,
                   trained_at, training_samples, contamination,
                   features_used, storage_type, model_path,
                   model_checksum, is_active, notes, performance_metrics
            FROM machine_model_registry
            WHERE maquina_id = %s AND is_active = TRUE
        """, (maquina_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id":                  row[0],
            "maquina_id":          row[1],
            "empresa_id":          row[2],
            "model_version":       row[3],
            "algorithm":           row[4],
            "trained_at":          row[5],
            "training_samples":    row[6],
            "contamination":       row[7],
            "features_used":       row[8],
            "storage_type":        row[9],
            "model_path":          row[10],
            "model_checksum":      row[11],
            "is_active":           row[12],
            "notes":               row[13],
            "performance_metrics": row[14],
        }
    finally:
        cur.close()
        conn.close()


def obtener_modelos_maquina(maquina_id: int) -> list[dict]:
    """
    Return all model versions for a machine ordered by training date (newest first).
    Used by ModelManager.list_models() and the dashboard model history table.
    Includes all fields needed for display and rollback decisions.
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT id, maquina_id, empresa_id, model_version, algorithm,
                   trained_at, training_samples, contamination,
                   features_used, storage_type, model_path,
                   model_checksum, is_active, notes, performance_metrics
            FROM machine_model_registry
            WHERE maquina_id = %s
            ORDER BY trained_at DESC
        """, (maquina_id,))
        cols = ["id","maquina_id","empresa_id","model_version","algorithm",
                "trained_at","training_samples","contamination",
                "features_used","storage_type","model_path",
                "model_checksum","is_active","notes","performance_metrics"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def obtener_modelos_anteriores_maquina(maquina_id: int, excluir_id: Optional[int] = None) -> list[dict]:
    """
    Return previous model versions for rollback candidates (newest first).
    Excludes the model with excluir_id (typically the currently-active failing model).
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        if excluir_id:
            cur.execute("""
                SELECT id, maquina_id, empresa_id, model_version, algorithm,
                       trained_at, training_samples, contamination,
                       features_used, storage_type, model_path,
                       model_checksum, is_active, notes, performance_metrics
                FROM machine_model_registry
                WHERE maquina_id = %s AND id != %s
                ORDER BY trained_at DESC
            """, (maquina_id, excluir_id))
        else:
            cur.execute("""
                SELECT id, maquina_id, empresa_id, model_version, algorithm,
                       trained_at, training_samples, contamination,
                       features_used, storage_type, model_path,
                       model_checksum, is_active, notes, performance_metrics
                FROM machine_model_registry
                WHERE maquina_id = %s AND is_active = FALSE
                ORDER BY trained_at DESC
            """, (maquina_id,))
        cols = ["id","maquina_id","empresa_id","model_version","algorithm",
                "trained_at","training_samples","contamination",
                "features_used","storage_type","model_path",
                "model_checksum","is_active","notes","performance_metrics"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


# ─── MACHINE BASELINES ────────────────────────────────────────────────────────

def guardar_baseline(
    maquina_id:     int,
    empresa_id:     int,
    n_samples:      int,
    stats_json:     dict,
    active_model_id: Optional[int] = None,
    baseline_from:  Optional[datetime] = None,
    baseline_to:    Optional[datetime] = None,
) -> bool:
    """
    Upsert the baseline for a machine (one row per machine).
    stats_json format: {feature_name: {mean, std, p5, p50, p95}}
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO machine_baselines
            (maquina_id, empresa_id, n_samples, stats_json,
             active_model_id, baseline_from, baseline_to, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (maquina_id) DO UPDATE SET
                n_samples       = EXCLUDED.n_samples,
                stats_json      = EXCLUDED.stats_json,
                active_model_id = EXCLUDED.active_model_id,
                baseline_from   = EXCLUDED.baseline_from,
                baseline_to     = EXCLUDED.baseline_to,
                updated_at      = NOW()
        """, (
            maquina_id, empresa_id, n_samples,
            json.dumps(stats_json),
            active_model_id, baseline_from, baseline_to,
        ))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def obtener_baseline(maquina_id: int) -> Optional[dict]:
    """Return the current baseline for a machine, or None."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT n_samples, stats_json, active_model_id,
                   baseline_from, baseline_to, updated_at
            FROM machine_baselines WHERE maquina_id = %s
        """, (maquina_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "n_samples":       row[0],
            "stats":           row[1],   # already JSONB — returned as dict
            "active_model_id": row[2],
            "baseline_from":   row[3],
            "baseline_to":     row[4],
            "updated_at":      row[5],
        }
    finally:
        cur.close()
        conn.close()


# ─── LECTURAS CNC V2 ──────────────────────────────────────────────────────────

def registrar_lectura_cnc(
    maquina_id:                 int,
    empresa_id:                 int,
    resultado:                  str,
    nivel_riesgo:               str,
    sampling_rate_configured:   float,
    # Optional fields (progressive — not all needed from day 1)
    sampling_rate_actual:       Optional[float] = None,
    sample_loss_fraction:       Optional[float] = None,
    rpm_nominal:                Optional[float] = None,
    rpm_real:                   Optional[float] = None,
    rpm_source:                 Optional[str]   = None,
    temperatura_c:              Optional[float] = None,
    carga_pct:                  Optional[float] = None,
    # Time domain per axis
    rms_x:          Optional[float] = None,
    rms_y:          Optional[float] = None,
    rms_z:          Optional[float] = None,
    peak_x:         Optional[float] = None,
    peak_y:         Optional[float] = None,
    peak_z:         Optional[float] = None,
    peak_to_peak_x: Optional[float] = None,
    peak_to_peak_y: Optional[float] = None,
    peak_to_peak_z: Optional[float] = None,
    kurtosis_x:     Optional[float] = None,
    kurtosis_y:     Optional[float] = None,
    kurtosis_z:     Optional[float] = None,
    skewness_x:     Optional[float] = None,
    skewness_y:     Optional[float] = None,
    skewness_z:     Optional[float] = None,
    crest_factor_x: Optional[float] = None,
    crest_factor_y: Optional[float] = None,
    crest_factor_z: Optional[float] = None,
    # Frequency domain
    dominant_freq_hz:   Optional[float] = None,
    dominant_amplitude: Optional[float] = None,
    spectral_energy:    Optional[float] = None,
    band_low_energy:    Optional[float] = None,
    band_mid_energy:    Optional[float] = None,
    band_high_energy:   Optional[float] = None,
    # Order analysis
    order_1x_energy: Optional[float] = None,
    order_2x_energy: Optional[float] = None,
    order_3x_energy: Optional[float] = None,
    # Quality and results
    signal_quality_score:   Optional[float] = None,
    data_quality_status:    Optional[str]   = None,
    anomaly_score:          Optional[float] = None,
    health_score:           Optional[int]   = None,
    diagnostico:            str             = '',
    model_version_id:       Optional[int]   = None,
) -> Optional[int]:
    """Insert a CNC reading. Returns the new row id."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO lecturas_cnc_v2 (
                maquina_id, empresa_id, resultado, nivel_riesgo,
                sampling_rate_configured, sampling_rate_actual, sample_loss_fraction,
                rpm_nominal, rpm_real, rpm_source,
                temperatura_c, carga_pct,
                rms_x, rms_y, rms_z,
                peak_x, peak_y, peak_z,
                peak_to_peak_x, peak_to_peak_y, peak_to_peak_z,
                kurtosis_x, kurtosis_y, kurtosis_z,
                skewness_x, skewness_y, skewness_z,
                crest_factor_x, crest_factor_y, crest_factor_z,
                dominant_freq_hz, dominant_amplitude, spectral_energy,
                band_low_energy, band_mid_energy, band_high_energy,
                order_1x_energy, order_2x_energy, order_3x_energy,
                signal_quality_score, data_quality_status,
                anomaly_score, health_score, diagnostico, model_version_id
            ) VALUES (
                %s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,
                %s,%s,%s,%s
            ) RETURNING id
        """, (
            maquina_id, empresa_id, resultado, nivel_riesgo,
            sampling_rate_configured, sampling_rate_actual, sample_loss_fraction,
            rpm_nominal, rpm_real, rpm_source,
            temperatura_c, carga_pct,
            rms_x, rms_y, rms_z,
            peak_x, peak_y, peak_z,
            peak_to_peak_x, peak_to_peak_y, peak_to_peak_z,
            kurtosis_x, kurtosis_y, kurtosis_z,
            skewness_x, skewness_y, skewness_z,
            crest_factor_x, crest_factor_y, crest_factor_z,
            dominant_freq_hz, dominant_amplitude, spectral_energy,
            band_low_energy, band_mid_energy, band_high_energy,
            order_1x_energy, order_2x_energy, order_3x_energy,
            signal_quality_score, data_quality_status,
            anomaly_score, health_score, diagnostico, model_version_id,
        ))
        lectura_id = cur.fetchone()[0]
        conn.commit()
        return lectura_id
    except Exception as e:
        conn.rollback()
        print(f"Error registrando lectura CNC: {e}")
        return None
    finally:
        cur.close()
        conn.close()


def obtener_historial_cnc(maquina_id: int, limite: int = 50) -> list[dict]:
    """Return recent CNC readings for a machine (newest first)."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT id, timestamp, resultado, nivel_riesgo, health_score,
                   anomaly_score, rms_x, kurtosis_x, signal_quality_score
            FROM lecturas_cnc_v2
            WHERE maquina_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """, (maquina_id, limite))
        cols = ["id", "timestamp", "resultado", "nivel_riesgo", "health_score",
                "anomaly_score", "rms_x", "kurtosis_x", "signal_quality_score"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def obtener_lecturas_para_baseline(
    maquina_id:      int,
    n:               int  = 200,
    excluir_estados: Optional[list[str]] = None,
) -> list[dict]:
    """
    Return up to N recent 'normal' readings with the full 8-feature vector
    used to build the Isolation Forest baseline.

    Only returns readings that are considered normal operation:
      - resultado NOT in excluir_estados (defaults: NOK, ALERTA, SENSOR_ERROR)
      - signal_quality_score >= 0.5  (unreliable signals excluded)
      - All 8 IF features must be non-NULL

    Columns returned (exactly the FEATURE_NAMES vector for the IF model):
      rms_x, kurtosis_x, crest_factor_x, peak_to_peak_x,
      dominant_freq_hz, band_low_energy, band_mid_energy, band_high_energy

    Plus metadata: id, timestamp, signal_quality_score.

    Args:
        maquina_id:      Machine integer PK.
        n:               Maximum number of readings to return.
        excluir_estados: resultado values to exclude.
                         Defaults to anomaly/error states.

    Returns:
        List of dicts, newest first, with the 8 IF features + metadata.
        Returns an empty list if there are fewer than 1 valid reading.
    """
    if excluir_estados is None:
        excluir_estados = [
            "NOK - Anomalía Detectada",
            "ALERTA",
            "SENSOR_ERROR",
        ]

    conn = get_conn()
    cur  = conn.cursor()
    try:
        # Use a tuple for the IN clause; psycopg2 adapts list→tuple correctly
        cur.execute("""
            SELECT id, timestamp, signal_quality_score,
                   rms_x, kurtosis_x, crest_factor_x, peak_to_peak_x,
                   dominant_freq_hz,
                   band_low_energy, band_mid_energy, band_high_energy
            FROM lecturas_cnc_v2
            WHERE maquina_id = %s
              AND resultado NOT IN %s
              AND signal_quality_score >= 0.5
              AND rms_x          IS NOT NULL
              AND kurtosis_x     IS NOT NULL
              AND crest_factor_x IS NOT NULL
              AND peak_to_peak_x IS NOT NULL
              AND dominant_freq_hz  IS NOT NULL
              AND band_low_energy   IS NOT NULL
              AND band_mid_energy   IS NOT NULL
              AND band_high_energy  IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT %s
        """, (maquina_id, tuple(excluir_estados), n))

        cols = [
            "id", "timestamp", "signal_quality_score",
            "rms_x", "kurtosis_x", "crest_factor_x", "peak_to_peak_x",
            "dominant_freq_hz",
            "band_low_energy", "band_mid_energy", "band_high_energy",
        ]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


# ─── HEALTH SCORES ────────────────────────────────────────────────────────────

def registrar_health_score(
    maquina_id: int,
    empresa_id: int,
    score:      int,
    trend:      Optional[str]   = None,
    slope:      Optional[float] = None,
    lectura_id: Optional[int]   = None,
) -> Optional[int]:
    """Insert a health score record. Returns the new row id."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO health_scores (maquina_id, empresa_id, score, trend, slope, lectura_id)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
        """, (maquina_id, empresa_id, score, trend, slope, lectura_id))
        hs_id = cur.fetchone()[0]
        conn.commit()
        return hs_id
    except Exception:
        conn.rollback()
        return None
    finally:
        cur.close()
        conn.close()


def obtener_historial_health(maquina_id: int, dias: int = 30) -> list[dict]:
    """Return health score history for the last N days."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT id, timestamp, score, trend, slope
            FROM health_scores
            WHERE maquina_id = %s
              AND timestamp >= NOW() - INTERVAL '1 day' * %s
            ORDER BY timestamp DESC
        """, (maquina_id, dias))
        cols = ["id", "timestamp", "score", "trend", "slope"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


# ─── RAW EVENT WINDOWS ────────────────────────────────────────────────────────

def registrar_evento_raw(
    maquina_id:         int,
    empresa_id:         int,
    event_timestamp:    datetime,
    pre_event_s:        float,
    post_event_s:       float,
    sampling_rate_hz:   float,
    total_samples:      int,
    axes_captured:      list[str],
    anomaly_score:      Optional[float] = None,
    health_score_at_event: Optional[int] = None,
    triggered_by_lectura_id: Optional[int] = None,
    storage_type:       str = "supabase",
    file_path:          Optional[str] = None,
    file_size_bytes:    Optional[int] = None,
    file_checksum:      Optional[str] = None,
) -> Optional[int]:
    """Register a raw event window (metadata only). Returns the new row id."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO raw_event_windows
            (maquina_id, empresa_id, event_timestamp,
             pre_event_s, post_event_s, sampling_rate_hz, total_samples, axes_captured,
             anomaly_score, health_score_at_event, triggered_by_lectura_id,
             storage_type, file_path, file_size_bytes, file_checksum)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            maquina_id, empresa_id, event_timestamp,
            pre_event_s, post_event_s, sampling_rate_hz, total_samples, axes_captured,
            anomaly_score, health_score_at_event, triggered_by_lectura_id,
            storage_type, file_path, file_size_bytes, file_checksum,
        ))
        event_id = cur.fetchone()[0]
        conn.commit()
        return event_id
    except Exception:
        conn.rollback()
        return None
    finally:
        cur.close()
        conn.close()


def marcar_evento_subido(event_id: int, file_path: str, file_checksum: Optional[str] = None) -> bool:
    """Mark a raw event window as successfully uploaded to cloud storage."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            UPDATE raw_event_windows
            SET is_uploaded = TRUE, uploaded_at = NOW(),
                file_path = %s, file_checksum = COALESCE(%s, file_checksum)
            WHERE id = %s
        """, (file_path, file_checksum, event_id))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def obtener_eventos_pendientes_upload(maquina_id: Optional[int] = None) -> list[dict]:
    """
    Return raw event windows not yet uploaded to Storage (offline sync queue).

    Returned columns (Fase 2D additions: empresa_id, file_checksum):
      id, maquina_id, empresa_id, event_timestamp, total_samples,
      file_path, file_checksum

    empresa_id is needed to build the deterministic Storage path.
    file_checksum allows RawStorageSync to verify integrity after upload.
    Results ordered oldest-first (FIFO, matching LocalBuffer behaviour).
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        if maquina_id:
            cur.execute("""
                SELECT id, maquina_id, empresa_id,
                       event_timestamp, total_samples,
                       file_path, file_checksum
                FROM raw_event_windows
                WHERE is_uploaded = FALSE AND maquina_id = %s
                ORDER BY created_at ASC
            """, (maquina_id,))
        else:
            cur.execute("""
                SELECT id, maquina_id, empresa_id,
                       event_timestamp, total_samples,
                       file_path, file_checksum
                FROM raw_event_windows
                WHERE is_uploaded = FALSE
                ORDER BY created_at ASC
            """)
        cols = ["id", "maquina_id", "empresa_id",
                "event_timestamp", "total_samples",
                "file_path", "file_checksum"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def actualizar_storage_modelo(
    model_id:     int,
    storage_type: str,
    storage_path: str,
    checksum:     Optional[str],
) -> bool:
    """
    Update the storage location and checksum of a model after uploading
    it to Supabase Storage (or reverting to local on failure).

    Called by ModelSync.upload_model() after a successful upload:
      actualizar_storage_modelo(model_id, 'supabase', key, sha256_hex)

    Returns True on success, False on any error.
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            UPDATE machine_model_registry
            SET storage_type   = %s,
                model_path     = %s,
                model_checksum = %s
            WHERE id = %s
        """, (storage_type, storage_path, checksum, model_id))
        conn.commit()
        return cur.rowcount == 1
    except Exception as exc:
        conn.rollback()
        print(f"[repositories] actualizar_storage_modelo failed: {exc}")
        return False
    finally:
        cur.close()
        conn.close()


def obtener_todas_maquinas_con_health(empresa_id: Optional[int] = None) -> list[dict]:
    """
    Return all machines with their latest health score and trend.
    Used by the dashboard multi-machine status map (Fase 4B).

    Returns one row per machine:
        maquina_id, nombre, tipo, empresa_id, health_score, trend, slope, timestamp
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        if empresa_id:
            cur.execute("""
                SELECT m.id, m.nombre, m.tipo, m.empresa_id,
                       hs.score, hs.trend, hs.slope, hs.timestamp
                FROM maquinas m
                LEFT JOIN LATERAL (
                    SELECT score, trend, slope, timestamp
                    FROM health_scores
                    WHERE maquina_id = m.id
                    ORDER BY timestamp DESC
                    LIMIT 1
                ) hs ON true
                WHERE m.empresa_id = %s
                ORDER BY m.nombre
            """, (empresa_id,))
        else:
            cur.execute("""
                SELECT m.id, m.nombre, m.tipo, m.empresa_id,
                       hs.score, hs.trend, hs.slope, hs.timestamp
                FROM maquinas m
                LEFT JOIN LATERAL (
                    SELECT score, trend, slope, timestamp
                    FROM health_scores
                    WHERE maquina_id = m.id
                    ORDER BY timestamp DESC
                    LIMIT 1
                ) hs ON true
                ORDER BY m.nombre
            """)
        cols = ["maquina_id", "nombre", "tipo", "empresa_id",
                "health_score", "trend", "slope", "timestamp"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()

# ─── MAINTENANCE EVENTS ───────────────────────────────────────────────────────

def registrar_mantenimiento(
    maquina_id:         int,
    empresa_id:         int,
    tipo:               str,        # 'preventivo'|'correctivo'|'predictivo'
    maintenance_at:     Optional[datetime] = None,
    componente:         Optional[str]   = None,
    descripcion:        Optional[str]   = None,
    tiempo_parada_h:    Optional[float] = None,
    coste_euros:        Optional[float] = None,
    tecnico:            Optional[str]   = None,
    alertado_por_ia:    bool            = False,
    dias_anticipacion:  Optional[int]   = None,
    registrado_por:     Optional[int]   = None,
    related_lectura_id: Optional[int]   = None,
) -> Optional[int]:
    """Register a maintenance event. Returns the new row id."""
    if maintenance_at is None:
        maintenance_at = datetime.now(timezone.utc)

    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO maintenance_events
            (maquina_id, empresa_id, maintenance_at, tipo, componente, descripcion,
             tiempo_parada_h, coste_euros, tecnico,
             alertado_por_ia, dias_anticipacion, registrado_por, related_lectura_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
        """, (
            maquina_id, empresa_id, maintenance_at, tipo, componente, descripcion,
            tiempo_parada_h, coste_euros, tecnico,
            alertado_por_ia, dias_anticipacion, registrado_por, related_lectura_id,
        ))
        ev_id = cur.fetchone()[0]
        conn.commit()
        return ev_id
    except Exception as e:
        conn.rollback()
        print(f"Error registrando mantenimiento: {e}")
        return None
    finally:
        cur.close()
        conn.close()


# ─── FAILURE EVENTS ───────────────────────────────────────────────────────────

def registrar_fallo(
    maquina_id:             int,
    empresa_id:             int,
    failure_at:             Optional[datetime] = None,
    tipo_fallo:             Optional[str]   = None,
    componente:             Optional[str]   = None,
    descripcion:            Optional[str]   = None,
    downtime_hours:         Optional[float] = None,
    coste_euros:            Optional[float] = None,
    primera_anomalia_ts:    Optional[datetime] = None,
    tiempo_deteccion_dias:  Optional[float] = None,
    maintenance_event_id:   Optional[int]   = None,
    registrado_por:         Optional[int]   = None,
) -> Optional[int]:
    """Register a real machine failure. Returns the new row id."""
    if failure_at is None:
        failure_at = datetime.now(timezone.utc)

    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO failure_events
            (maquina_id, empresa_id, failure_at, tipo_fallo, componente, descripcion,
             downtime_hours, coste_euros,
             primera_anomalia_ts, tiempo_deteccion_dias,
             maintenance_event_id, registrado_por)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
        """, (
            maquina_id, empresa_id, failure_at, tipo_fallo, componente, descripcion,
            downtime_hours, coste_euros,
            primera_anomalia_ts, tiempo_deteccion_dias,
            maintenance_event_id, registrado_por,
        ))
        fallo_id = cur.fetchone()[0]
        conn.commit()
        return fallo_id
    except Exception:
        conn.rollback()
        return None
    finally:
        cur.close()
        conn.close()


# ─── ALERT LOG ────────────────────────────────────────────────────────────────

def registrar_alerta(
    maquina_id:  int,
    empresa_id:  int,
    tipo_alerta: str,
    destinatario: str,
    asunto:      Optional[str]  = None,
    enviado:     bool           = True,
    error_msg:   Optional[str]  = None,
) -> Optional[int]:
    """Log an alert attempt. Returns the new row id."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO alert_log
            (maquina_id, empresa_id, tipo_alerta, destinatario, asunto, enviado, error_msg)
            VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id
        """, (maquina_id, empresa_id, tipo_alerta, destinatario, asunto, enviado, error_msg))
        alert_id = cur.fetchone()[0]
        conn.commit()
        return alert_id
    except Exception:
        conn.rollback()
        return None
    finally:
        cur.close()
        conn.close()


def puede_enviar_alerta(maquina_id: int, cooldown_hours: float = 1.0) -> bool:
    """
    Check if enough time has passed since the last SENT alert for this machine.
    Replaces the in-memory _ultimo_envio dict in alertas.py.

    Returns True if no alert was sent within the cooldown period.
    cooldown_hours <= 0 always returns True (no cooldown).
    """
    if cooldown_hours <= 0:
        return True
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT MAX(sent_at)
            FROM alert_log
            WHERE maquina_id = %s AND enviado = TRUE
        """, (maquina_id,))
        row = cur.fetchone()
        ultimo = row[0] if row else None
        if ultimo is None:
            return True
        # Make timezone-aware for comparison
        ahora = datetime.now(timezone.utc)
        if ultimo.tzinfo is None:
            ultimo = ultimo.replace(tzinfo=timezone.utc)
        return (ahora - ultimo) >= timedelta(hours=cooldown_hours)
    finally:
        cur.close()
        conn.close()


def ultimo_envio_alerta(maquina_id: int) -> Optional[datetime]:
    """Return the timestamp of the last successful alert for a machine."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT MAX(sent_at) FROM alert_log
            WHERE maquina_id = %s AND enviado = TRUE
        """, (maquina_id,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()
        conn.close()
