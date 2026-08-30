"""
AuraPredict — ModelStore (Fase 2C)
====================================
Local filesystem storage for Isolation Forest model files.

Stores serialized .joblib files and a metadata JSON that survives
Raspberry Pi reboots — enabling offline operation after restart.

File layout (per machine):
  {base_dir}/{maquina_id}/v{version}.joblib   — serialized IsolationForest
  {base_dir}/{maquina_id}/active.json          — metadata of the active model

Supabase Storage upload is Fase 2D.
This module only handles the local filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
from sklearn.ensemble import IsolationForest


class ModelStore:
    """
    Serializes and loads Isolation Forest models to/from the local filesystem.

    Does NOT access the database.
    Does NOT upload to Supabase Storage.
    One ModelStore per Edge device, shared across machines.
    """

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base

    # ── Path helpers ───────────────────────────────────────────────────────────

    def machine_dir(self, maquina_id: int) -> Path:
        """Return (and create) the directory for one machine's models."""
        d = self._base / str(maquina_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def model_path(self, maquina_id: int, version: str) -> Path:
        """Absolute path for a specific model version .joblib file."""
        return self.machine_dir(maquina_id) / f"v{version}.joblib"

    def _active_meta_path(self, maquina_id: int) -> Path:
        return self.machine_dir(maquina_id) / "active.json"

    # ── Save / Load model ──────────────────────────────────────────────────────

    def save(
        self,
        model:      IsolationForest,
        maquina_id: int,
        version:    str,
    ) -> Path:
        """
        Serialize a trained model to disk.

        Args:
            model:      Trained IsolationForest instance.
            maquina_id: Machine integer PK (used for directory structure).
            version:    Version string, e.g. '1.100.0'.

        Returns:
            Absolute Path of the saved .joblib file.
        """
        path = self.model_path(maquina_id, version)
        joblib.dump(model, path)
        return path

    def load(
        self,
        maquina_id: int,
        version:    str,
    ) -> Optional[IsolationForest]:
        """
        Load a specific model version from disk.

        Returns None if the file does not exist or cannot be deserialized.
        """
        path = self.model_path(maquina_id, version)
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    # ── Active model metadata ─────────────────────────────────────────────────

    def save_active_metadata(
        self,
        maquina_id:       int,
        version:          str,
        model_path:       str,
        model_version_id: Optional[int],
    ) -> None:
        """
        Persist the active model's metadata to a JSON file.

        This file is the fallback for offline startup: if the database is
        not reachable, MachineBaselineManager reads this file to load the
        last known good model without a DB query.
        """
        metadata = {
            "version":          version,
            "model_path":       model_path,
            "model_version_id": model_version_id,
        }
        try:
            self._active_meta_path(maquina_id).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
        except OSError:
            pass  # Non-critical — model file itself is the source of truth

    def load_active_metadata(self, maquina_id: int) -> Optional[dict]:
        """
        Load the active model metadata from the local JSON file.

        Returns None if the file does not exist or is corrupted.
        """
        path = self._active_meta_path(maquina_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def load_active_model(
        self,
        maquina_id: int,
    ) -> tuple[Optional[IsolationForest], Optional[dict]]:
        """
        Load the active model and its metadata in one call.

        Returns:
            (model, metadata) if both exist and are loadable.
            (None, metadata)  if metadata exists but model file is missing.
            (None, None)      if no active metadata exists.
        """
        meta = self.load_active_metadata(maquina_id)
        if meta is None:
            return None, None

        model_path = meta.get("model_path", "")
        if not Path(model_path).exists():
            return None, meta

        try:
            model = joblib.load(model_path)
            return model, meta
        except Exception:
            return None, meta
