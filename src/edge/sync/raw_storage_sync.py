"""
AuraPredict — RawStorageSync (Fase 2D)
========================================
Uploads pending .npz raw event files from the Edge filesystem to
Supabase Storage and updates raw_event_windows accordingly.

Safety protocol (order is critical):
  1. Local .npz already exists (saved by RawEventCapture in Fase 2C).
  2. Metadata already registered in raw_event_windows (is_uploaded=FALSE).
  3. upload()  → file goes to Supabase Storage (upsert=True, idempotent).
  4. marcar_evento_subido() → is_uploaded=TRUE, file_path=storage_key.
  5. local_file.unlink() → delete ONLY after step 4 confirmed.

If the process dies between steps 3 and 4:
  - The file is in Storage but is_uploaded is still FALSE.
  - Next retry: upload succeeds again (upsert=True, no duplicate).
  - marcar_evento_subido() is called → consistent state restored.
  - No data loss.

If the process dies between steps 4 and 5:
  - is_uploaded=TRUE (consistent) but local file still exists.
  - On next sync: event not returned by obtener_eventos_pendientes_upload().
  - Local file remains as orphan (harmless, periodic cleanup can remove it).

Deterministic Storage path (approved in architecture review):
  bucket : aurapredict-raw-events
  key    : {empresa_id}/{maquina_id}/{window_id}.npz

The window_id is extracted from the local filename (stem of .npz path),
because it was already embedded there by RawEventCapture (Fase 2C).

Testability:
  fetch_pending_fn and mark_uploaded_fn are injectable so tests can
  mock DB calls without a real Supabase connection.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from .storage_client import StorageClient

# Type aliases for the injectable DB functions
FetchPendingFn  = Callable[[Optional[int]], list[dict]]
MarkUploadedFn  = Callable[[int, str, Optional[str]], bool]


class RawStorageSync:
    """
    Uploads pending raw event windows (.npz) to Supabase Storage.

    One instance per Edge device.  Called from EdgePipeline when online.

    Production:
        client = StorageClient.from_env()
        sync   = RawStorageSync(client)
        n      = sync.upload_pending(maquina_id=42)

    Tests:
        sync = RawStorageSync(
            mock_client,
            fetch_pending_fn=lambda mid: [fake_event],
            mark_uploaded_fn=lambda eid, path, chk: True,
        )
    """

    BUCKET: str = "aurapredict-raw-events"

    def __init__(
        self,
        storage_client:   StorageClient,
        fetch_pending_fn: Optional[FetchPendingFn] = None,
        mark_uploaded_fn: Optional[MarkUploadedFn] = None,
    ) -> None:
        self._storage        = storage_client
        self._fetch_pending  = fetch_pending_fn
        self._mark_uploaded  = mark_uploaded_fn

    # ── Public API ─────────────────────────────────────────────────────────────

    def upload_pending(
        self,
        maquina_id:     int,
        max_per_cycle:  int = 5,
    ) -> int:
        """
        Upload up to max_per_cycle pending raw event files to Storage.

        Behaviour:
          - Fetches events with is_uploaded=FALSE from the DB.
          - For each event (oldest first):
              1. Skip if local file is missing (continue to next).
              2. Compute or reuse SHA-256 checksum.
              3. Upload with upsert=True (idempotent retry).
              4. Mark in DB — STOP here if this fails.
              5. Delete local file — ONLY after step 4 confirmed.
          - Stops at the first upload or DB failure.
          - Respects max_per_cycle to avoid blocking the acquisition loop.

        Args:
            maquina_id:    Filter events to this machine.
            max_per_cycle: Maximum uploads per call.

        Returns:
            Number of events successfully uploaded and marked.
        """
        fetch_fn = self._get_fetch_fn()
        mark_fn  = self._get_mark_fn()

        try:
            pending = fetch_fn(maquina_id)
        except Exception as exc:
            print(f"[RawStorageSync] Cannot fetch pending events: {exc}")
            return 0

        uploaded = 0
        for event in pending[:max_per_cycle]:
            success = self._upload_one(event, mark_fn)
            if success is True:
                uploaded += 1
            elif success is False:
                break           # network/DB failure — stop, retry next cycle
            # success is None → skipped (missing file), continue to next event

        return uploaded

    @staticmethod
    def storage_key(empresa_id: int, maquina_id: int, window_id: str) -> str:
        """
        Deterministic Storage path for one raw event.

        Format: {empresa_id}/{maquina_id}/{window_id}.npz
        The window_id is the UUID embedded in the local filename by RawEventCapture.
        """
        return f"{empresa_id}/{maquina_id}/{window_id}.npz"

    # ── Internal ───────────────────────────────────────────────────────────────

    def _upload_one(
        self,
        event:   dict,
        mark_fn: MarkUploadedFn,
    ):
        """
        Process a single pending event.

        Returns:
          True  — uploaded and marked successfully
          False — upload or DB failure (caller should stop the loop)
          None  — skipped (missing file or invalid data), caller can continue
        """
        event_id   = event["id"]
        maquina_id = event["maquina_id"]
        empresa_id = event.get("empresa_id")
        file_path  = event.get("file_path")

        # ── Guard: must have file_path and empresa_id ─────────────────────────
        if not file_path or empresa_id is None:
            print(f"[RawStorageSync] Event {event_id}: missing file_path or "
                  f"empresa_id — skipping")
            return None  # skip, not a sync failure

        local_path = Path(file_path)

        # ── Guard: local file must exist ──────────────────────────────────────
        if not local_path.exists():
            print(f"[RawStorageSync] Event {event_id}: local file not found "
                  f"({local_path}) — skipping")
            return None  # skip, not a sync failure

        # ── SHA-256: reuse stored checksum if available ───────────────────────
        sha256 = event.get("file_checksum")
        if not sha256:
            try:
                sha256 = hashlib.sha256(local_path.read_bytes()).hexdigest()
            except OSError as exc:
                print(f"[RawStorageSync] Event {event_id}: cannot read file: {exc}")
                return None

        # ── Deterministic Storage key from window_id (filename stem) ──────────
        window_id = local_path.stem   # e.g. '550e8400-e29b-41d4-a716-446655440000'
        key       = self.storage_key(empresa_id, maquina_id, window_id)

        # ── Step 3: upload to Storage (upsert=True → idempotent retry) ────────
        ok = self._storage.upload(self.BUCKET, key, local_path, upsert=True)
        if not ok:
            print(f"[RawStorageSync] Event {event_id}: upload failed → stopping")
            return False  # network/Storage failure — stop the loop

        # ── Step 4: mark as uploaded in DB ────────────────────────────────────
        # CRITICAL: only delete the local file if this step succeeds.
        try:
            marked = mark_fn(event_id, key, sha256)
        except Exception as exc:
            print(f"[RawStorageSync] Event {event_id}: mark_evento_subido "
                  f"raised: {exc} → stopping (file in Storage but DB not updated)")
            return False  # DB failure — stop; next retry will re-upload (idempotent)

        if not marked:
            print(f"[RawStorageSync] Event {event_id}: mark_evento_subido "
                  f"returned False → stopping")
            return False

        # ── Step 5: delete local file ONLY after DB confirmation ──────────────
        try:
            local_path.unlink()
        except OSError:
            pass  # File already deleted by another process — harmless

        return True

    # ── Lazy imports (production only) ─────────────────────────────────────────

    def _get_fetch_fn(self) -> FetchPendingFn:
        if self._fetch_pending is not None:
            return self._fetch_pending
        _add_src_to_path()
        from database_v2.repositories import obtener_eventos_pendientes_upload
        return obtener_eventos_pendientes_upload

    def _get_mark_fn(self) -> MarkUploadedFn:
        if self._mark_uploaded is not None:
            return self._mark_uploaded
        _add_src_to_path()
        from database_v2.repositories import marcar_evento_subido
        return marcar_evento_subido


def _add_src_to_path() -> None:
    """Add src/ to sys.path so database_v2 is importable."""
    src = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
    if src not in sys.path:
        sys.path.insert(0, src)
