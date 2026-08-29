"""
AuraPredict — RLS Context Manager
====================================
Helper to set the company context (app.current_empresa_id) for
Row Level Security policies defined in V2_010__rls_policies.sql.

CURRENT BEHAVIOR (Phase 2A):
  The DATABASE_URL connects as the postgres superuser, which bypasses
  RLS by PostgreSQL design. These helpers are PASSIVE — they do nothing
  harmful if called, but RLS is not enforced on the superuser role.

FUTURE BEHAVIOR (when non-superuser role is activated):
  Each API request will call set_company_context(conn, empresa_id)
  using the empresa_id from the validated JWT token. From that point,
  the RLS policies in V2_010 will enforce isolation at the DB level.

Integration point:
  The empresa_id comes from the JWT payload (see auth.py / api.py):
      payload = verificar_token(token)
      empresa_id = payload.get("empresa_id")  # None for admin users

  Admin users (empresa_id=None) do not call set_company_context,
  so current_empresa_id() returns NULL → all rows visible (expected).
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from database import get_conn


def set_company_context(conn, empresa_id: int) -> None:
    """
    Set the company context for the current transaction.
    Must be called WITHIN a transaction (before any SELECT/INSERT).

    Usage:
        conn = get_conn()
        conn.autocommit = False
        set_company_context(conn, empresa_id)
        cur = conn.cursor()
        cur.execute("SELECT * FROM lecturas_cnc_v2 ...")
        ...
        conn.commit()
        conn.close()
    """
    with conn.cursor() as cur:
        cur.execute(
            "SET LOCAL app.current_empresa_id = %s",
            (str(empresa_id),)
        )


def clear_company_context(conn) -> None:
    """Remove the company context setting for the current transaction."""
    with conn.cursor() as cur:
        cur.execute("RESET app.current_empresa_id")


@contextlib.contextmanager
def company_context(empresa_id: Optional[int]):
    """
    Context manager that opens a connection, sets company context,
    yields the connection, and commits/closes on exit.

    empresa_id=None means no filter (admin access — all companies visible).

    Example:
        with company_context(empresa_id=3) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM lecturas_cnc_v2 WHERE maquina_id = %s", (42,))
            rows = cur.fetchall()
    """
    conn = get_conn()
    conn.autocommit = False
    try:
        if empresa_id is not None:
            set_company_context(conn, empresa_id)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
