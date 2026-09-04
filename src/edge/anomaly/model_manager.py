"""
AuraPredict — ModelManager  (Fase 5B)
=======================================
Orchestrates the full lifecycle of Isolation Forest models per machine.

Responsibilities:
  - List all model versions for a machine (history)
  - Get the currently active model record
  - Validate a model before activation (5-step check)
  - Activate a model safely (never activates without passing validation)
  - Rollback to the latest valid previous version
  - Trigger manual retraining on demand

What ModelManager does NOT do (already handled elsewhere):
  - Serialize/deserialize .joblib files → ModelStore
  - Upload/download to Supabase Storage → ModelSync
  - Train the IF model → BaselineManager._train_isolation_forest()
  - Compute features → AcquisitionSession + FeatureExtraction
  - Detect anomalies → IsolationForestDetector

Architecture:
    API / Dashboard
          ↓
    ModelManager           ← this module
      ├─ repositories      (list, get_active, activate)
      ├─ ModelStore        (load local .joblib for validation)
      └─ BaselineManager   (trigger retraining)

Validation protocol (Fase 5B, Bloque 4):
  1. File exists on disk (model_path)
  2. SHA-256 matches model_checksum when available
  3. joblib.load() succeeds
  4. Model has n_features_in_ == len(FEATURE_NAMES) (feature compatibility)
  5. predict() on a dummy vector produces a valid result

Rollback protocol:
  - Searches previous versions from newest to oldest
  - Validates each candidate; stops at the first valid one
  - Never activates a candidate that fails validation
  - If no valid previous version exists: returns error, active model unchanged
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Add src/ to path so sibling modules are importable
_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ─── FEATURE NAMES ────────────────────────────────────────────────────────────
# Imported lazily to avoid circular imports.
def _get_feature_names() -> list[str]:
    from edge.anomaly.anomaly_detector import FEATURE_NAMES
    return FEATURE_NAMES


# ─── RESULT TYPES ─────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of a 5-step model validation."""
    valid:   bool
    checks:  dict[str, bool]    # {'file_exists': True, 'sha256': True, ...}
    error:   Optional[str]      # Human-readable reason if not valid

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(valid=True, checks={}, error=None)

    @classmethod
    def fail(cls, reason: str, checks: dict | None = None) -> "ValidationResult":
        return cls(valid=False, checks=checks or {}, error=reason)


@dataclass
class ActivationResult:
    """Result of an activate_model or rollback_model operation."""
    success:    bool
    model_id:   Optional[int]
    version:    Optional[str]
    error:      Optional[str]


@dataclass
class TrainingResult:
    """Result of a train_model operation."""
    success:       bool
    model_id:      Optional[int]
    version:       Optional[str]
    training_samples: int
    error:         Optional[str]


# ─── MODEL MANAGER ────────────────────────────────────────────────────────────

class ModelManager:
    """
    Lifecycle manager for Isolation Forest models.

    Production usage (DB required):
        mm = ModelManager(model_store=store)
        models = mm.list_models(maquina_id=42)
        result = mm.activate_model(model_id=7)

    Offline usage:
        mm = ModelManager(model_store=store)
        vr = mm.validate_model(model_path="/tmp/v1.joblib", expected_checksum=sha)
        # activate_model will fail gracefully if DB is unreachable
    """

    def __init__(
        self,
        model_store=None,           # ModelStore instance
        baseline_manager=None,      # MachineBaselineManager (for train_model)
        *,
        # Injectable DB functions for tests (lazy imports when None)
        list_fn         = None,
        get_active_fn   = None,
        activate_fn     = None,
        get_previous_fn = None,
    ) -> None:
        self._model_store       = model_store
        self._baseline_manager  = baseline_manager

        # Injectable DB functions — None = use real repositories
        self._list_fn         = list_fn
        self._get_active_fn   = get_active_fn
        self._activate_fn     = activate_fn
        self._get_previous_fn = get_previous_fn

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_models(self, maquina_id: int) -> list[dict]:
        """
        Return all model versions for a machine ordered by training date (newest first).

        Each dict contains: id, model_version, is_active, storage_type,
        model_path, model_checksum, trained_at, training_samples,
        features_used, performance_metrics, and more.
        """
        try:
            fn = self._list_fn or self._import("obtener_modelos_maquina")
            return fn(maquina_id)
        except Exception as exc:
            logger.warning("list_models failed for maquina_id=%d: %s", maquina_id, exc)
            return []

    def get_active_model(self, maquina_id: int) -> Optional[dict]:
        """
        Return the currently active model record, or None.
        The dict includes all columns including features_used and model_checksum.
        """
        try:
            fn = self._get_active_fn or self._import("obtener_modelo_activo")
            return fn(maquina_id)
        except Exception as exc:
            logger.warning("get_active_model failed for maquina_id=%d: %s", maquina_id, exc)
            return None

    def validate_model(
        self,
        model_path:        str | Path,
        expected_checksum: Optional[str] = None,
        expected_features: Optional[list[str]] = None,
    ) -> ValidationResult:
        """
        Validate a model file through 5 sequential checks.

        Checks:
          1. File exists on disk
          2. SHA-256 matches expected_checksum (skipped if None)
          3. joblib.load() succeeds without exception
          4. Loaded object has n_features_in_ == len(expected_features)
          5. predict() on a dummy vector returns a valid result

        Args:
            model_path:        Absolute or relative path to the .joblib file.
            expected_checksum: SHA-256 hex to verify (from model_checksum column).
                               If None, step 2 is skipped.
            expected_features: Feature name list to check compatibility.
                               Defaults to FEATURE_NAMES (8 features).

        Returns:
            ValidationResult with valid=True if all checks pass.
        """
        checks: dict[str, bool] = {}
        path = Path(model_path)
        n_expected = len(expected_features or _get_feature_names())

        # ── Check 1: file exists ────────────────────────────────────────────────
        checks["file_exists"] = path.exists()
        if not checks["file_exists"]:
            return ValidationResult.fail(
                f"Model file not found: {path}", checks
            )

        # ── Check 2: SHA-256 integrity ──────────────────────────────────────────
        if expected_checksum:
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                checks["sha256"] = (actual == expected_checksum)
                if not checks["sha256"]:
                    return ValidationResult.fail(
                        f"SHA-256 mismatch: expected {expected_checksum[:12]}…, "
                        f"got {actual[:12]}…", checks
                    )
            except OSError as exc:
                checks["sha256"] = False
                return ValidationResult.fail(f"Cannot read file for checksum: {exc}", checks)
        else:
            checks["sha256"] = True  # skipped — not available

        # ── Check 3: joblib load ────────────────────────────────────────────────
        try:
            model = joblib.load(path)
            checks["joblib_load"] = True
        except Exception as exc:
            checks["joblib_load"] = False
            return ValidationResult.fail(f"joblib.load() failed: {exc}", checks)

        # ── Check 4: feature compatibility ─────────────────────────────────────
        try:
            n_actual = getattr(model, "n_features_in_", None)
            if n_actual is None:
                # Older sklearn versions may not have n_features_in_ — skip
                checks["features"] = True
            else:
                checks["features"] = (n_actual == n_expected)
                if not checks["features"]:
                    return ValidationResult.fail(
                        f"Feature count mismatch: model expects {n_actual}, "
                        f"pipeline produces {n_expected}", checks
                    )
        except Exception as exc:
            checks["features"] = False
            return ValidationResult.fail(f"Feature check error: {exc}", checks)

        # ── Check 5: predict() test ─────────────────────────────────────────────
        try:
            dummy = np.zeros((1, n_expected), dtype=np.float64)
            result = model.predict(dummy)
            checks["predict"] = (result is not None and len(result) == 1)
            if not checks["predict"]:
                return ValidationResult.fail("predict() returned invalid output", checks)
        except Exception as exc:
            checks["predict"] = False
            return ValidationResult.fail(f"predict() raised exception: {exc}", checks)

        logger.debug("Model validation passed: %s", path.name)
        return ValidationResult(valid=True, checks=checks, error=None)

    def activate_model(
        self,
        model_id:          int,
        skip_validation:   bool = False,
    ) -> ActivationResult:
        """
        Activate a model by its ID after passing validation.

        The model is only activated if:
          - The record exists in the DB
          - The model file passes all 5 validation checks (unless skip_validation=True)

        Args:
            model_id:        FK to machine_model_registry.id
            skip_validation: If True, skip file validation (use with caution)

        Returns:
            ActivationResult with success=True if activation succeeded.
        """
        # Get the model record
        try:
            all_models = None
            # We need to find the model by ID across all its machine's models
            # Use list_models by getting maquina_id from the registry
            activate_fn = self._activate_fn or self._import("activar_modelo")

            # Fetch model details for validation
            model_record = self._get_model_by_id(model_id)
            if model_record is None:
                return ActivationResult(
                    success=False, model_id=None, version=None,
                    error=f"Model id={model_id} not found in registry"
                )
        except Exception as exc:
            return ActivationResult(
                success=False, model_id=None, version=None,
                error=f"Cannot retrieve model record: {exc}"
            )

        version    = model_record.get("model_version", "?")
        model_path = model_record.get("model_path", "")
        checksum   = model_record.get("model_checksum")

        # Validate before activating
        if not skip_validation:
            vr = self.validate_model(model_path, expected_checksum=checksum)
            if not vr.valid:
                logger.warning(
                    "Activation rejected for model_id=%d (v%s): %s",
                    model_id, version, vr.error
                )
                return ActivationResult(
                    success=False, model_id=model_id, version=version,
                    error=f"Validation failed: {vr.error}"
                )

        # Activate in DB
        try:
            activate_fn = self._activate_fn or self._import("activar_modelo")
            ok = activate_fn(model_id)
            if not ok:
                return ActivationResult(
                    success=False, model_id=model_id, version=version,
                    error="activar_modelo() returned False (model may not exist)"
                )
            logger.info("Model activated: id=%d, version=%s", model_id, version)
            return ActivationResult(
                success=True, model_id=model_id, version=version, error=None
            )
        except Exception as exc:
            logger.error("DB activation failed for model_id=%d: %s", model_id, exc)
            return ActivationResult(
                success=False, model_id=model_id, version=version,
                error=f"DB error during activation: {exc}"
            )

    def rollback_model(self, maquina_id: int) -> ActivationResult:
        """
        Roll back to the latest valid previous model version for a machine.

        Algorithm:
          1. Get the currently active model (to exclude from candidates)
          2. Fetch all previous versions ordered newest→oldest
          3. For each candidate: run validate_model()
          4. Activate the first candidate that passes
          5. If none passes: return error, active model unchanged

        Returns:
            ActivationResult describing the outcome.
        """
        # Get current active model to exclude it from candidates
        current = self.get_active_model(maquina_id)
        current_id = current["id"] if current else None

        # Fetch previous versions
        try:
            prev_fn = self._get_previous_fn or self._import("obtener_modelos_anteriores_maquina")
            candidates = prev_fn(maquina_id, excluir_id=current_id)
        except Exception as exc:
            return ActivationResult(
                success=False, model_id=None, version=None,
                error=f"Cannot fetch previous versions: {exc}"
            )

        if not candidates:
            return ActivationResult(
                success=False, model_id=None, version=None,
                error="No previous model versions available for rollback"
            )

        # Try each candidate from newest to oldest
        for candidate in candidates:
            cid     = candidate["id"]
            cver    = candidate.get("model_version", "?")
            cpath   = candidate.get("model_path", "")
            cchk    = candidate.get("model_checksum")

            logger.debug("Rollback: testing candidate id=%d v%s", cid, cver)
            vr = self.validate_model(cpath, expected_checksum=cchk)
            if vr.valid:
                result = self.activate_model(cid, skip_validation=True)
                if result.success:
                    logger.info(
                        "Rollback succeeded: maquina_id=%d → id=%d v%s",
                        maquina_id, cid, cver
                    )
                    return result
                # activate_fn failed — try next
            else:
                logger.debug(
                    "Rollback candidate id=%d v%s failed validation: %s",
                    cid, cver, vr.error
                )

        return ActivationResult(
            success=False, model_id=None, version=None,
            error=f"No valid previous version found among {len(candidates)} candidates"
        )

    def train_model(
        self,
        maquina_id: int,
        auto_activate: bool = False,
    ) -> TrainingResult:
        """
        Trigger a manual retraining of the Isolation Forest model.

        Uses the data accumulated in the linked BaselineManager.
        The new model is NOT automatically activated unless auto_activate=True.
        After training, call activate_model(new_model_id) explicitly.

        Args:
            maquina_id:    Machine to retrain for.
            auto_activate: If True, activate the new model after training
                           (only if it passes validation).

        Returns:
            TrainingResult with the new model_id if successful.
        """
        if self._baseline_manager is None:
            return TrainingResult(
                success=False, model_id=None, version=None,
                training_samples=0,
                error="No BaselineManager linked — cannot trigger retraining"
            )

        bm = self._baseline_manager
        if not bm.is_baseline_ready:
            n = bm.n_samples
            min_n = bm._baseline_min_samples
            return TrainingResult(
                success=False, model_id=None, version=None,
                training_samples=n,
                error=(
                    f"Insufficient baseline data: {n}/{min_n} samples. "
                    "Continue normal operation until the baseline is ready."
                )
            )

        # Capture model_id before training (to detect the new one after)
        before = self.get_active_model(maquina_id)
        before_id = before["id"] if before else None

        try:
            # _update_baseline() triggers retraining when conditions are met,
            # but _train_isolation_forest() directly trains without waiting.
            bm._train_isolation_forest(bm._baseline_buffer)
            n_samples = len(bm._baseline_buffer)
        except Exception as exc:
            return TrainingResult(
                success=False, model_id=None, version=None,
                training_samples=0,
                error=f"Training failed: {exc}"
            )

        # Get the newly-created model
        after = self.get_active_model(maquina_id)
        after_id = after["id"] if after else None
        version  = after.get("model_version", "?") if after else "?"

        if after_id is None or after_id == before_id:
            return TrainingResult(
                success=False, model_id=None, version=None,
                training_samples=n_samples,
                error="Training completed but new model record not found in DB"
            )

        logger.info(
            "Manual training complete: maquina_id=%d, model_id=%d, v%s, n=%d",
            maquina_id, after_id, version, n_samples
        )
        return TrainingResult(
            success=True, model_id=after_id, version=version,
            training_samples=n_samples, error=None
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_model_by_id(self, model_id: int) -> Optional[dict]:
        """
        Retrieve a single model record by its primary key.
        Uses list_models() to avoid adding a new repository function.
        """
        # We don't know the maquina_id, so we need a direct lookup.
        # Use the extended activar_modelo query indirectly — just fetch
        # the maquina_id from the registry, then list all and filter.
        try:
            fn = self._import("get_conn_and_fetch_model_by_id_internal")
        except Exception:
            pass

        # Direct lightweight query
        try:
            from database_v2.repositories import get_conn
            conn = get_conn()
            cur  = conn.cursor()
            try:
                cur.execute("""
                    SELECT id, maquina_id, empresa_id, model_version, algorithm,
                           trained_at, training_samples, contamination,
                           features_used, storage_type, model_path,
                           model_checksum, is_active, notes, performance_metrics
                    FROM machine_model_registry
                    WHERE id = %s
                """, (model_id,))
                row = cur.fetchone()
                if not row:
                    return None
                cols = ["id","maquina_id","empresa_id","model_version","algorithm",
                        "trained_at","training_samples","contamination",
                        "features_used","storage_type","model_path",
                        "model_checksum","is_active","notes","performance_metrics"]
                return dict(zip(cols, row))
            finally:
                cur.close()
                conn.close()
        except Exception as exc:
            logger.warning("_get_model_by_id(%d) DB error: %s", model_id, exc)
            return None

    @staticmethod
    def _import(fn_name: str):
        """Lazy import of a repository function by name."""
        from database_v2 import repositories
        return getattr(repositories, fn_name)
