"""
AuraPredict — LocalBuffer
==========================
Persistent offline-first buffer for FeatureSet payloads.

One JSON file per entry. Files survive Raspberry Pi reboots because they
reside on the SD card. FIFO order is maintained by ISO timestamp prefix
in the filename (lexicographic sort = chronological order).

Write contract (atomicity):
  1. Serialize payload to {filename}.tmp
  2. os.rename(.tmp → final)  — atomic on POSIX (Linux/macOS)
     On Windows: shutil.move() used as fallback.

Flush contract:
  - Reads entries in FIFO order (oldest first)
  - For each entry: calls send_fn(payload)
  - If send_fn returns non-None → entry confirmed → delete file → continue
  - If send_fn returns None or raises → stop flush immediately
  - Files before the failure are deleted (confirmed received)
  - The failed file remains in the buffer for next retry
  - No entry is deleted before send_fn confirms success
  - No entry can be sent twice (file present = not yet confirmed)

Max entries:
  - When buffer is at capacity before push(), the OLDEST entry is dropped
  - Acquisition is never blocked by a full buffer

Recovery after restart:
  - LocalBuffer.__init__() scans the directory
  - All existing JSON files are immediately available as pending entries
  - No state is kept in memory across restarts; the filesystem is the source of truth
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


class LocalBuffer:
    """
    Persistent, FIFO, offline-first buffer for CNC reading payloads.

    Payload format: dict from FeatureSet.to_lectura_cnc_payload()
    File format:    JSON, one file per entry
    Filename:       {YYYYMMDDTHHMMSS.ffffff}_{window_id}.json

    Example:
        buffer = LocalBuffer('/tmp/aurapredict/buffer', max_entries=500)

        # During offline period:
        buffer.push(payload, window_id='uuid4-string')

        # When Supabase reconnects:
        n = buffer.flush(send_fn=lambda p: registrar_lectura_cnc(**p))
        # n = number of entries successfully flushed
    """

    _SUFFIX:     str = ".json"
    _TMP_SUFFIX: str = ".tmp"

    def __init__(self, base_dir: str, max_entries: int = 500) -> None:
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries

    # ── Public API ─────────────────────────────────────────────────────────────

    def push(self, payload: dict, window_id: str) -> None:
        """
        Persist one payload to disk atomically.

        If the buffer is at max_entries capacity, the oldest entry is dropped
        before writing the new one. Acquisition is never blocked by a full buffer.

        The window_id is stored inside the JSON for traceability (not sent to DB).
        It also appears in the filename for easy identification.

        Args:
            payload:   Dict from FeatureSet.to_lectura_cnc_payload().
                       Must not contain keys starting with '_'.
            window_id: Internal UUID4 string — used in filename and JSON body.
        """
        self._enforce_limit()

        ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
        filename = f"{ts}_{window_id}{self._SUFFIX}"

        final_path = self._dir / filename
        tmp_path   = self._dir / (filename + self._TMP_SUFFIX)

        # Store window_id inside the file body for traceability
        stored = dict(payload)
        stored["_window_id"] = window_id

        try:
            tmp_path.write_text(json.dumps(stored, default=str), encoding="utf-8")
            # Atomic rename: on POSIX this is guaranteed atomic.
            # On Windows, os.rename may fail if the destination exists;
            # shutil.move is used as fallback.
            try:
                os.rename(tmp_path, final_path)
            except OSError:
                shutil.move(str(tmp_path), str(final_path))
        except Exception:
            # Clean up .tmp if rename failed
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def flush(self, send_fn: Callable[[dict], Optional[int]]) -> int:
        """
        Send buffered entries to Supabase in FIFO order.

        For each entry:
          - Loads payload from JSON (strips internal '_' keys before calling send_fn)
          - Calls send_fn(payload) → int (success) or None/raises (failure)
          - Success: deletes the file
          - Failure: stops immediately; remaining files stay in buffer for retry

        No entry is sent twice: a file exists = not yet confirmed.
        A file is deleted ONLY after send_fn returns a non-None value.

        Args:
            send_fn: Callable that accepts the payload dict as keyword arguments
                     and returns a non-None value on success (e.g. a DB row id).

        Returns:
            Number of entries successfully flushed.
        """
        flushed = 0
        for path in self._sorted_entries():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                # Strip internal metadata keys before calling send_fn
                payload = {k: v for k, v in stored.items() if not k.startswith("_")}

                result = send_fn(payload)
                if result is not None:
                    path.unlink()
                    flushed += 1
                else:
                    # send_fn returned None → treat as failure, stop
                    break
            except Exception:
                # Any exception (network, DB, JSON decode) → stop flush
                break

        return flushed

    def pending_count(self) -> int:
        """Number of entries currently in the buffer."""
        return len(self._sorted_entries())

    def is_empty(self) -> bool:
        """True if no entries are pending."""
        return self.pending_count() == 0

    def clear(self) -> int:
        """
        Remove ALL entries from the buffer.
        Use carefully — entries are lost permanently.

        Returns:
            Number of entries removed.
        """
        entries = self._sorted_entries()
        for path in entries:
            try:
                path.unlink()
            except OSError:
                pass
        return len(entries)

    def list_entries(self) -> list[str]:
        """Return filenames of buffered entries in FIFO order."""
        return [p.name for p in self._sorted_entries()]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _sorted_entries(self) -> list[Path]:
        """
        Return buffer entries sorted oldest-first (FIFO).

        Sort key is the filename — ISO timestamp prefix guarantees
        lexicographic order = chronological order.
        Excludes .tmp files (incomplete writes).
        """
        return sorted(self._dir.glob(f"*{self._SUFFIX}"))

    def _enforce_limit(self) -> None:
        """
        If the buffer is at or above max_entries, delete the oldest entry.

        Called before every push() to ensure the buffer never exceeds its limit.
        Drops the oldest entry silently — this is the expected behavior when the
        device has been offline longer than the buffer capacity allows.
        """
        entries = self._sorted_entries()
        while len(entries) >= self.max_entries:
            oldest = entries.pop(0)
            try:
                oldest.unlink()
            except OSError:
                pass
