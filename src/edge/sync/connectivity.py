"""
AuraPredict — Connectivity (Fase 2D)
======================================
Single point of connectivity detection for Supabase Storage.

Design decisions:
  - One function: is_supabase_reachable().
  - No retry logic here — callers decide when to retry.
  - A 4xx response (e.g. 401 Unauthorized) counts as REACHABLE;
    only 5xx or network errors count as UNREACHABLE.
  - Timeout is short (3s default) so it doesn't block acquisition.

Usage pattern in EdgePipeline:
    if is_supabase_reachable(config.supabase_url):
        raw_sync.upload_pending(maquina_id)
    # else: silent, will retry next cycle
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from .storage_client import StorageClient


def is_supabase_reachable(supabase_url: str, timeout_s: float = 3.0) -> bool:
    """
    Check whether the Supabase Storage API is responding.

    Makes a lightweight HEAD request to /storage/v1/bucket.
    No auth is required to check reachability — a 401 response
    still means the server is up and accepting connections.

    Args:
        supabase_url: Supabase project URL (no trailing slash).
        timeout_s:    Maximum wait before declaring unreachable.

    Returns:
        True  — server responded with any HTTP status < 500.
        False — connection refused, DNS failure, or timeout.
    """
    url = f"{supabase_url.rstrip('/')}/storage/v1/bucket"
    try:
        resp = requests.head(url, timeout=timeout_s)
        return resp.status_code < 500
    except (requests.ConnectionError, requests.Timeout):
        return False
    except Exception:
        return False


def get_storage_client_from_env() -> Optional[StorageClient]:
    """
    Build a StorageClient from environment variables.

    Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
    Returns None if either is missing — callers must handle the None case
    and continue in offline mode.

    This is the canonical factory for production use.
    Tests inject a StorageClient with a mock session instead.
    """
    return StorageClient.from_env()


def supabase_url_from_env() -> str:
    """Return SUPABASE_URL from the environment, or empty string."""
    return os.getenv("SUPABASE_URL", "").strip()
