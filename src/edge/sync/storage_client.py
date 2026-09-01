"""
AuraPredict — StorageClient (Fase 2D)
======================================
Thin wrapper over the Supabase Storage REST API using `requests`.

Why `requests` and not `supabase-py`:
  - `requests` is already in requirements.txt.
  - The Storage REST API is straightforward (upload/exists/download).
  - `supabase-py` adds heavy dependencies (httpx, postgrest-py, etc.)
    that are unnecessary for this use case on a Raspberry Pi.

Authentication:
  All requests use the Supabase Service Role Key as a Bearer token.
  The key is loaded from the SUPABASE_SERVICE_ROLE_KEY environment
  variable — never hardcoded.

Testability:
  The underlying HTTP session is injectable via the `session` parameter.
  Tests pass a mock session; production code uses a real requests.Session.

API endpoints used:
  Upload  : POST   /storage/v1/object/{bucket}/{path}  (x-upsert: true)
  Exists  : HEAD   /storage/v1/object/authenticated/{bucket}/{path}
  Download: GET    /storage/v1/object/authenticated/{bucket}/{path}

All methods return bool (success/failure) or raise nothing —
callers never need to catch exceptions from this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests


class StorageClient:
    """
    Supabase Storage REST API client.

    Example (production):
        client = StorageClient.from_env()
        if client:
            ok = client.upload("aurapredict-raw-events", "1/42/uuid.npz", path)

    Example (tests):
        mock_session = MagicMock()
        mock_session.post.return_value.status_code = 200
        client = StorageClient("https://x.supabase.co", "key", session=mock_session)
    """

    def __init__(
        self,
        supabase_url:      str,
        service_role_key:  str,
        timeout_s:         float = 30.0,
        session:           Optional[requests.Session] = None,
    ) -> None:
        self._base     = supabase_url.rstrip("/")
        self._key      = service_role_key
        self._timeout  = timeout_s

        # Inject session for testability; production creates its own
        self._session  = session or requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {service_role_key}",
        })

    # ── URL helpers ────────────────────────────────────────────────────────────

    @property
    def _storage(self) -> str:
        return f"{self._base}/storage/v1"

    def _object_url(self, bucket: str, storage_key: str) -> str:
        return f"{self._storage}/object/{bucket}/{storage_key}"

    def _authenticated_url(self, bucket: str, storage_key: str) -> str:
        """URL for accessing private-bucket objects with a service role key."""
        return f"{self._storage}/object/authenticated/{bucket}/{storage_key}"

    # ── Public API ─────────────────────────────────────────────────────────────

    def upload(
        self,
        bucket:       str,
        storage_key:  str,
        file_path:    Path,
        content_type: str  = "application/octet-stream",
        upsert:       bool = True,
    ) -> bool:
        """
        Upload a local file to Supabase Storage.

        Args:
            bucket:       Storage bucket name.
            storage_key:  Path within the bucket (e.g. '1/42/uuid.npz').
            file_path:    Local file to upload. Must exist.
            content_type: MIME type sent in Content-Type header.
            upsert:       If True, overwrites an existing object with the same key.
                          This makes uploads idempotent — safe to retry.

        Returns:
            True on HTTP 200 or 201. False on any error (network, HTTP, etc.).
        """
        url = self._object_url(bucket, storage_key)
        headers = {"Content-Type": content_type}
        if upsert:
            headers["x-upsert"] = "true"

        try:
            with open(file_path, "rb") as fh:
                resp = self._session.post(
                    url,
                    data    = fh,
                    headers = headers,
                    timeout = self._timeout,
                )
            if resp.status_code not in (200, 201):
                print(f"[StorageClient] Upload failed: HTTP {resp.status_code} "
                      f"— {bucket}/{storage_key}")
            return resp.status_code in (200, 201)

        except requests.ConnectionError as exc:
            print(f"[StorageClient] Upload connection error: {exc}")
            return False
        except requests.Timeout:
            print(f"[StorageClient] Upload timeout: {bucket}/{storage_key}")
            return False
        except FileNotFoundError:
            print(f"[StorageClient] Local file not found: {file_path}")
            return False
        except Exception as exc:
            print(f"[StorageClient] Upload unexpected error: {exc}")
            return False

    def exists(self, bucket: str, storage_key: str) -> bool:
        """
        Check whether an object exists in Storage without downloading it.

        Uses a HEAD request to the authenticated endpoint.
        Returns False on any error (including network errors).
        A 404 means the object does not exist (not an error).
        """
        url = self._authenticated_url(bucket, storage_key)
        try:
            resp = self._session.head(url, timeout=5.0)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
        except Exception:
            return False

    def download(
        self,
        bucket:      str,
        storage_key: str,
        dest_path:   Path,
    ) -> bool:
        """
        Download an object from Storage to a local file.

        Safety protocol:
          1. Stream response to {dest_path}.tmp
          2. Atomic os.rename(.tmp → dest_path) on success
          3. Delete .tmp on any failure

        If the download fails for any reason, dest_path is NOT created or
        overwritten — the previous file (if any) is preserved.

        Returns:
            True if the file was successfully downloaded and renamed.
            False on HTTP error, network error, or any other failure.
        """
        url      = self._authenticated_url(bucket, storage_key)
        tmp_path = Path(str(dest_path) + ".tmp")

        try:
            with self._session.get(url, stream=True, timeout=self._timeout) as resp:
                if resp.status_code != 200:
                    print(f"[StorageClient] Download failed: HTTP {resp.status_code} "
                          f"— {bucket}/{storage_key}")
                    return False

                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65_536):
                        fh.write(chunk)

            # Atomic rename — dest_path only exists if download completed
            os.rename(tmp_path, dest_path)
            return True

        except requests.ConnectionError as exc:
            print(f"[StorageClient] Download connection error: {exc}")
        except requests.Timeout:
            print(f"[StorageClient] Download timeout: {bucket}/{storage_key}")
        except Exception as exc:
            print(f"[StorageClient] Download unexpected error: {exc}")

        # Clean up partial .tmp file
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return False

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> Optional["StorageClient"]:
        """
        Create a StorageClient from environment variables.

        Reads:
          SUPABASE_URL              — project URL (no trailing slash)
          SUPABASE_SERVICE_ROLE_KEY — service role JWT

        Returns None if either variable is missing or empty.
        """
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url or not key:
            return None
        return cls(supabase_url=url, service_role_key=key)
