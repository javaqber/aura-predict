"""
AuraPredict — ModelSync (Fase 2D)
===================================
Synchronises Isolation Forest .joblib model files between the Edge
filesystem (ModelStore) and Supabase Storage.

Upload protocol (after training a new IF model):
  1. SHA-256 of local .joblib  (already saved by ModelStore/baseline_manager)
  2. Build deterministic Storage key
  3. upload() with upsert=True  (idempotent retry)
  4. actualizar_storage_modelo() → storage_type='supabase', key, sha256
  5. Local .joblib is NOT deleted — Edge continues using it offline

Download protocol (e.g. restoring after SD card replacement):
  1. Build Storage key from version
  2. StorageClient.download() → writes to dest_path.candidate (temp)
  3. Verify SHA-256 of candidate against expected_sha
  4. VALID   → os.rename(candidate → dest_path)  [atomic; replaces old model]
  5. INVALID → candidate.unlink(); dest_path (old model) preserved; return False

Safety guarantee:
  The currently active local model is NEVER replaced until the downloaded
  candidate passes SHA-256 verification.  A corrupted or truncated download
  leaves the old model intact and returns False.

Deterministic Storage path (approved in architecture review):
  bucket : aurapredict-models
  key    : {empresa_id}/{maquina_id}/{version}.joblib

Testability:
  update_storage_fn is injectable so tests can mock the BD call without
  a real Supabase connection.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from .storage_client import StorageClient

# Type alias for the injectable DB function
UpdateStorageFn = Callable[[int, str, str, Optional[str]], bool]


class ModelSync:
    """
    Uploads and downloads ML model files (.joblib) to/from Supabase Storage.

    Production:
        client = StorageClient.from_env()
        sync   = ModelSync(client)
        ok     = sync.upload_model(maquina_id, empresa_id, model_id,
                                   version, local_path)

    Tests (no Supabase):
        sync = ModelSync(mock_client, update_storage_fn=lambda *a: True)
    """

    BUCKET: str = "aurapredict-models"

    def __init__(
        self,
        storage_client:   StorageClient,
        update_storage_fn: Optional[UpdateStorageFn] = None,
    ) -> None:
        self._storage        = storage_client
        self._update_storage = update_storage_fn

    # ── Public API ─────────────────────────────────────────────────────────────

    def upload_model(
        self,
        maquina_id: int,
        empresa_id: int,
        model_id:   int,
        version:    str,
        local_path: Path,
    ) -> bool:
        """
        Upload a trained .joblib model to Supabase Storage and update its
        DB record with the Storage key and SHA-256 checksum.

        The local file is NOT deleted — the Edge continues to use it for
        inference even when offline.

        Args:
            maquina_id: Machine integer PK.
            empresa_id: Company integer PK.
            model_id:   FK to machine_model_registry.id.
            version:    Model version string (e.g. '1.100.0').
            local_path: Absolute path to the .joblib file.

        Returns:
            True if both the Storage upload and the DB update succeeded.
            False on any error (local file remains valid and unchanged).
        """
        if not local_path.exists():
            print(f"[ModelSync] upload_model: file not found: {local_path}")
            return False

        # 1. SHA-256 of the local model file
        try:
            sha256 = hashlib.sha256(local_path.read_bytes()).hexdigest()
        except OSError as exc:
            print(f"[ModelSync] upload_model: cannot read file: {exc}")
            return False

        # 2. Deterministic Storage key
        key = self.storage_key(empresa_id, maquina_id, version)

        # 3. Upload with upsert=True (idempotent retry)
        ok = self._storage.upload(
            bucket       = self.BUCKET,
            storage_key  = key,
            file_path    = local_path,
            content_type = "application/octet-stream",
            upsert       = True,
        )
        if not ok:
            print(f"[ModelSync] upload_model: Storage upload failed for {key}")
            return False

        # 4. Update DB: storage_type → 'supabase', model_path → key, checksum
        update_fn = self._get_update_fn()
        try:
            updated = update_fn(model_id, "supabase", key, sha256)
        except Exception as exc:
            print(f"[ModelSync] upload_model: DB update failed: {exc}. "
                  f"File is in Storage but record still shows 'local'.")
            return False

        if not updated:
            print(f"[ModelSync] upload_model: DB update returned False for "
                  f"model_id={model_id}")
            return False

        return True

    def download_model(
        self,
        maquina_id:   int,
        empresa_id:   int,
        version:      str,
        dest_path:    Path,
        expected_sha: Optional[str] = None,
    ) -> bool:
        """
        Download a .joblib model from Supabase Storage to the local filesystem.

        Safety:
          - Downloads to a temporary .candidate file first.
          - Verifies SHA-256 before replacing the existing local model.
          - If verification fails: removes .candidate; dest_path is unchanged.
          - Only on success: atomic os.rename(candidate → dest_path).

        Args:
            maquina_id:   Machine integer PK.
            empresa_id:   Company integer PK.
            version:      Model version string to download.
            dest_path:    Target path for the downloaded .joblib.
            expected_sha: Expected SHA-256 hex. If None, skip verification.

        Returns:
            True  — file downloaded, verified, and renamed to dest_path.
            False — download failed, checksum mismatch, or any other error.
                    In all failure cases dest_path (old model) is preserved.
        """
        key       = self.storage_key(empresa_id, maquina_id, version)
        candidate = Path(str(dest_path) + ".candidate")

        # 1. Download to .candidate (StorageClient handles its own .tmp internally)
        ok = self._storage.download(self.BUCKET, key, candidate)
        if not ok:
            print(f"[ModelSync] download_model: Storage download failed for {key}")
            # StorageClient already cleaned up any partial .tmp
            return False

        # 2. Verify SHA-256 of the downloaded candidate
        if expected_sha:
            try:
                actual_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError as exc:
                print(f"[ModelSync] download_model: cannot read candidate: {exc}")
                _safe_unlink(candidate)
                return False

            if actual_sha != expected_sha:
                print(
                    f"[ModelSync] download_model: SHA-256 mismatch for {key}. "
                    f"Expected {expected_sha[:12]}…, got {actual_sha[:12]}…. "
                    f"Removing candidate; existing model preserved."
                )
                _safe_unlink(candidate)
                return False

        # 3. Atomic replace: candidate → dest_path
        # The old dest_path (if any) is only overwritten here, after verification.
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            os.rename(candidate, dest_path)
        except OSError as exc:
            print(f"[ModelSync] download_model: rename failed: {exc}")
            _safe_unlink(candidate)
            return False

        return True

    @staticmethod
    def storage_key(empresa_id: int, maquina_id: int, version: str) -> str:
        """
        Deterministic Storage path for one model version.

        Format: {empresa_id}/{maquina_id}/{version}.joblib
        """
        return f"{empresa_id}/{maquina_id}/{version}.joblib"

    # ── Lazy imports ───────────────────────────────────────────────────────────

    def _get_update_fn(self) -> UpdateStorageFn:
        if self._update_storage is not None:
            return self._update_storage
        _add_src_to_path()
        from database_v2.repositories import actualizar_storage_modelo
        return actualizar_storage_modelo


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_unlink(path: Path) -> None:
    """Delete a file, ignoring errors if it doesn't exist."""
    try:
        path.unlink()
    except OSError:
        pass


def _add_src_to_path() -> None:
    """Add src/ to sys.path so database_v2 is importable."""
    src = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
    if src not in sys.path:
        sys.path.insert(0, src)
