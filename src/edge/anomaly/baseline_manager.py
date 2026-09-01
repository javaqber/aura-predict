"""
AuraPredict — MachineBaselineManager (Fase 2C)
================================================
Manages the normal-operation baseline for one machine and decides
which anomaly detector to use (ZScore cold start → IsolationForest production).

Responsibilities:
  1. Extract the 8-feature vector from a FeatureSet (no DSP — reads existing fields).
  2. Accumulate 'normal' readings in a rolling in-memory buffer.
  3. Compute and update baseline statistics (μ, σ, percentiles) per feature.
  4. Persist baseline to the database (machine_baselines) and to a local JSON
     file for offline recovery.
  5. Train/retrain IsolationForest when enough normal samples are available.
  6. Register the trained model in machine_model_registry via repositories.py.
  7. Switch from ZScoreDetector to IsolationForestDetector transparently.
  8. Recover from DB or local JSON after a Raspberry Pi restart.

What this class does NOT do:
  - Access sensor hardware.
  - Call registrar_lectura_cnc() — that is the pipeline's responsibility.
  - Upload the model to Supabase Storage (Fase 2D).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from sklearn.ensemble import IsolationForest

# Add src/ to path so database_v2 is importable (same pattern as repositories.py)
# sys.path for database_v2 lazy imports (inside methods)
_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Relative imports within the anomaly package
from .anomaly_detector import (
    AnomalyDetector,
    ZScoreDetector,
    IsolationForestDetector,
    FEATURE_NAMES,
)
from .model_store import ModelStore

# Avoid circular import: FeatureSet is only referenced in type hints
if TYPE_CHECKING:
    from ..pipeline.models import FeatureSet


class MachineBaselineManager:
    """
    Baseline lifecycle manager for one machine.

    Lifecycle:
      1. startup()     → load_from_db() or offline fallback
      2. per cycle     → extract_feature_vector() → record_reading()
      3. get_active_detector() → the pipeline uses this to analyze each vector

    Detector selection:
      - n_total_normal < baseline_min_samples → ZScoreDetector (cold start)
      - n_total_normal >= baseline_min_samples → IsolationForestDetector
    """

    def __init__(
        self,
        maquina_id:           int,
        empresa_id:           int,
        model_store:          ModelStore,
        primary_axis:         str         = "x",
        baseline_min_samples: int         = 50,
        update_every_n:       int         = 20,
        baseline_window_n:    int         = 200,
        z_threshold:          float       = 3.0,
        if_n_estimators:      int         = 50,
        if_contamination:     str | float = "auto",
    ) -> None:
        self._maquina_id           = maquina_id
        self._empresa_id           = empresa_id
        self._model_store          = model_store
        self._primary_axis         = primary_axis
        self._baseline_min_samples = baseline_min_samples
        self._update_every_n       = update_every_n
        self._baseline_window_n    = baseline_window_n
        self._z_threshold          = z_threshold
        self._if_n_estimators      = if_n_estimators
        self._if_contamination     = if_contamination

        # Rolling in-memory buffer of normal feature vectors
        self._buffer: list[np.ndarray] = []
        self._n_since_last_update: int  = 0

        # Persisted state
        self._baseline_stats: dict      = {}      # {feature: {mean, std, p5, p50, p95}}
        self._n_total_normal: int       = 0
        self._active_model_version_id: Optional[int] = None

        # Detector — starts as ZScore until enough samples
        self._active_detector: AnomalyDetector = ZScoreDetector(z_threshold)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_from_db(self) -> bool:
        """
        Load baseline stats and active model from the database.
        Falls back to local JSON files if the database is unavailable.

        Returns True if a baseline was successfully loaded (allows skipping
        the coldest cold-start period when historical data exists).
        """
        loaded = self._try_load_from_db()
        if not loaded:
            loaded = self._load_baseline_from_local()

        # Try to load the active model regardless of where baseline came from
        if loaded:
            self._try_load_active_model()

        return loaded

    def extract_feature_vector(
        self,
        feature_set: "FeatureSet",
    ) -> Optional[np.ndarray]:
        """
        Extract the 8-element IF feature vector from a FeatureSet.

        Uses the primary axis configured in AnomalyConfig.
        Returns None if the primary axis has no valid features.
        Returns a vector with NaN for individual missing values;
        AnomalyDetector._impute_nans() handles those.
        """
        ma = feature_set.multiaxis
        vf = ma.get_axis(self._primary_axis)
        if vf is None:
            return None

        def _f(obj, *attrs) -> float:
            """Navigate nested attributes safely; return NaN if any is None."""
            for attr in attrs:
                if obj is None:
                    return float("nan")
                obj = getattr(obj, attr, None)
            return float(obj) if obj is not None else float("nan")

        def _band(vf_obj, key: str) -> float:
            if vf_obj is None:
                return float("nan")
            v = vf_obj.freq.band_energies.get(key)
            return float(v) if v is not None else float("nan")

        return np.array([
            _f(vf, "time", "rms"),
            _f(vf, "time", "kurtosis"),
            _f(vf, "time", "crest_factor"),
            _f(vf, "time", "peak_to_peak"),
            _f(vf, "freq", "dominant_freq"),
            _band(vf, "low"),
            _band(vf, "mid"),
            _band(vf, "high"),
        ], dtype=float)

    def record_reading(
        self,
        vector:    np.ndarray,
        is_normal: bool,
    ) -> bool:
        """
        Record one feature vector. Accumulates it only if is_normal=True.

        Args:
            vector:    8-feature vector from extract_feature_vector().
            is_normal: True when resultado is 'OK - Sano' or 'OK - Aprendiendo'.

        Returns:
            True if the baseline or detector was updated after this reading.
        """
        if not is_normal:
            return False

        # Skip vectors with all NaN (sensor error slipped through)
        if np.all(np.isnan(vector)):
            return False

        self._buffer.append(vector)
        # Keep buffer bounded
        if len(self._buffer) > self._baseline_window_n:
            self._buffer.pop(0)

        self._n_total_normal      += 1
        self._n_since_last_update += 1

        # Trigger an update?
        threshold_reached = (self._n_total_normal == self._baseline_min_samples)
        periodic_update   = (self._n_since_last_update >= self._update_every_n)

        if threshold_reached or periodic_update:
            self._update_baseline()
            self._n_since_last_update = 0
            return True

        return False

    def get_active_detector(self) -> AnomalyDetector:
        """Return the currently active anomaly detector."""
        return self._active_detector

    @property
    def baseline_stats(self) -> dict:
        """Current statistical baseline used by ZScoreDetector."""
        return self._baseline_stats

    @property
    def n_samples(self) -> int:
        """Total number of normal readings seen since creation/reset."""
        return self._n_total_normal

    @property
    def is_baseline_ready(self) -> bool:
        """True when enough normal readings exist for IF training."""
        return self._n_total_normal >= self._baseline_min_samples

    # ── Internal: baseline update ──────────────────────────────────────────────

    def _update_baseline(self) -> None:
        """Recompute statistics from the rolling buffer and retrain IF if ready."""
        if len(self._buffer) < 5:
            return  # Too few points for meaningful statistics

        data = np.array(self._buffer)   # shape (n, 8)

        # Update per-feature statistics
        self._baseline_stats = {
            name: {
                "mean": float(np.nanmean(data[:, i])),
                "std":  float(max(np.nanstd(data[:, i]), 1e-9)),
                "p5":   float(np.nanpercentile(data[:, i], 5)),
                "p50":  float(np.nanmedian(data[:, i])),
                "p95":  float(np.nanpercentile(data[:, i], 95)),
            }
            for i, name in enumerate(FEATURE_NAMES)
        }

        # Persist to local JSON (always available for offline recovery)
        self._save_baseline_to_local()

        # Persist to DB (best effort — offline = silent skip)
        self._save_baseline_to_db()

        # Train IF when enough data
        if self._n_total_normal >= self._baseline_min_samples and len(data) >= 10:
            self._train_isolation_forest(data)

    def _train_isolation_forest(self, data: np.ndarray) -> None:
        """Train a new IsolationForest, save it, and activate it."""
        # Replace NaN with feature means for training stability
        clean = data.copy()
        for i in range(data.shape[1]):
            col_mean = np.nanmean(clean[:, i])
            clean[np.isnan(clean[:, i]), i] = col_mean if np.isfinite(col_mean) else 0.0

        contamination = self._if_contamination
        if contamination != "auto":
            contamination = float(contamination)

        model = IsolationForest(
            n_estimators  = self._if_n_estimators,
            contamination = contamination,
            random_state  = 42,
        )
        model.fit(clean)

        version    = f"1.{self._n_total_normal}.0"
        model_path = self._model_store.save(model, self._maquina_id, version)

        model_version_id = self._register_model_in_db(
            model      = model,
            version    = version,
            model_path = str(model_path),
            n_samples  = len(data),
        )

        self._model_store.save_active_metadata(
            maquina_id       = self._maquina_id,
            version          = version,
            model_path       = str(model_path),
            model_version_id = model_version_id,
        )

        self._active_model_version_id = model_version_id
        self._active_detector = IsolationForestDetector(
            model            = model,
            model_version_id = model_version_id,
            z_threshold      = self._z_threshold,
        )

        # Fase 2D: upload model to Supabase Storage (best-effort, offline-safe)
        self._try_upload_model(model_path, version, model_version_id)

    # ── Internal: database interactions ───────────────────────────────────────

    def _try_load_from_db(self) -> bool:
        try:
            from database_v2.repositories import obtener_baseline
            row = obtener_baseline(self._maquina_id)
            if not row or not row.get("stats"):
                return False
            self._baseline_stats = row["stats"]
            self._n_total_normal = int(row.get("n_samples", 0))
            return True
        except Exception:
            return False

    def _try_load_active_model(self) -> None:
        model, meta = self._model_store.load_active_model(self._maquina_id)
        if model is None:
            return
        model_version_id = meta.get("model_version_id") if meta else None
        self._active_model_version_id = model_version_id
        self._active_detector = IsolationForestDetector(
            model            = model,
            model_version_id = model_version_id,
            z_threshold      = self._z_threshold,
        )

    def _try_upload_model(
        self,
        model_path:       Path,
        version:          str,
        model_version_id: Optional[int],
    ) -> None:
        """
        Upload the trained .joblib to Supabase Storage (Fase 2D).
        Completely non-blocking: any failure is logged and ignored.
        The model remains fully operational locally.
        """
        if model_version_id is None:
            return  # No BD record → cannot link the upload
        try:
            from ..sync.connectivity import get_storage_client_from_env
            from ..sync.model_sync import ModelSync
            client = get_storage_client_from_env()
            if client is None:
                return  # SUPABASE_URL/KEY not configured
            sync = ModelSync(client)
            sync.upload_model(
                maquina_id = self._maquina_id,
                empresa_id = self._empresa_id,
                model_id   = model_version_id,
                version    = version,
                local_path = model_path,
            )
        except Exception as exc:
            print(f"[BaselineManager] Model upload skipped: {exc}")

    def _register_model_in_db(
        self,
        model:      IsolationForest,
        version:    str,
        model_path: str,
        n_samples:  int,
    ) -> Optional[int]:
        """Register and activate the new model. Returns model_id or None."""
        try:
            from database_v2.repositories import registrar_modelo, activar_modelo

            # Determine contamination value for storage
            cont = model.contamination
            cont_val = float(cont) if isinstance(cont, float) else None

            model_id = registrar_modelo(
                maquina_id       = self._maquina_id,
                empresa_id       = self._empresa_id,
                model_version    = version,
                trained_at       = datetime.now(timezone.utc),
                training_samples = n_samples,
                model_path       = model_path,
                algorithm        = "isolation_forest",
                features_used    = FEATURE_NAMES,
                storage_type     = "local",
                contamination    = cont_val,
            )
            if model_id:
                activar_modelo(model_id)
            return model_id
        except Exception:
            return None  # DB offline — model saved locally, will sync later

    def _save_baseline_to_db(self) -> None:
        try:
            from database_v2.repositories import guardar_baseline
            guardar_baseline(
                maquina_id      = self._maquina_id,
                empresa_id      = self._empresa_id,
                n_samples       = self._n_total_normal,
                stats_json      = self._baseline_stats,
                active_model_id = self._active_model_version_id,
            )
        except Exception:
            pass  # Offline — local JSON is the fallback

    # ── Internal: local JSON persistence ──────────────────────────────────────

    def _local_baseline_path(self) -> Path:
        d = self._model_store.base_dir / str(self._maquina_id)
        d.mkdir(parents=True, exist_ok=True)
        return d / "baseline.json"

    def _save_baseline_to_local(self) -> None:
        try:
            payload = {
                "n_samples":          self._n_total_normal,
                "stats":              self._baseline_stats,
                "model_version_id":   self._active_model_version_id,
            }
            self._local_baseline_path().write_text(
                json.dumps(payload), encoding="utf-8"
            )
        except Exception:
            pass

    def _load_baseline_from_local(self) -> bool:
        try:
            path = self._local_baseline_path()
            if not path.exists():
                return False
            data = json.loads(path.read_text(encoding="utf-8"))
            self._baseline_stats           = data.get("stats", {})
            self._n_total_normal           = int(data.get("n_samples", 0))
            self._active_model_version_id  = data.get("model_version_id")
            return bool(self._baseline_stats)
        except Exception:
            return False
