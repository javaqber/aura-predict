"""
AuraPredict — Migration Runner (v2)
=====================================
Applies versioned SQL migration files from the migrations/ directory.
Tracks applied migrations in the schema_migrations table.

Usage:
    from src.database_v2.migration_runner import run_migrations
    run_migrations()                    # applies all pending migrations
    run_migrations(dry_run=True)        # shows what would be applied
    status()                            # prints current migration status

Design decisions:
  - Each migration file is applied inside a transaction.
    If a migration fails the transaction is rolled back and the runner stops.
  - Migration files are immutable: if a previously applied file's checksum
    changes, the runner raises MigrationError (tampered migration).
  - The schema_migrations table is created on the fly using the V2_001 file;
    subsequent migrations are tracked through it.
  - The migrations/ directory is resolved relative to this file's location.
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys

import psycopg2

# Re-use the existing connection factory — no duplication
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from database import get_conn

MIGRATIONS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '../../migrations')
)


class MigrationError(Exception):
    """Raised when a migration cannot be safely applied."""


# ─── INTERNAL HELPERS ─────────────────────────────────────────────────────────

def _file_checksum(path: str) -> str:
    """MD5 checksum of a migration file (for tamper detection)."""
    with open(path, 'r', encoding='utf-8') as f:
        return hashlib.md5(f.read().encode()).hexdigest()


def _migration_files() -> list[str]:
    """Return migration files sorted by version string (V2_001, V2_002, …)."""
    pattern = os.path.join(MIGRATIONS_DIR, 'V2_*.sql')
    files = sorted(glob.glob(pattern))
    if not files:
        raise MigrationError(
            f"No migration files found in {MIGRATIONS_DIR}. "
            "Check that the migrations/ directory is present."
        )
    return files


def _extract_version(filepath: str) -> str:
    """Extract version string from filename, e.g. 'V2_001'."""
    return os.path.basename(filepath).split('__')[0]


def _tracker_exists(conn) -> bool:
    """True if schema_migrations table already exists."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name   = 'schema_migrations'
            )
        """)
        return bool(cur.fetchone()[0])


def _bootstrap_tracker(conn, v001_path: str) -> None:
    """
    Apply V2_001 (the tracker itself) and record it.
    Called only when schema_migrations does not exist yet.
    """
    with open(v001_path, 'r', encoding='utf-8') as f:
        sql = f.read()
    checksum = _file_checksum(v001_path)
    version  = _extract_version(v001_path)
    desc     = os.path.basename(v001_path)

    with conn:
        cur = conn.cursor()
        cur.execute(sql)
        cur.execute(
            "INSERT INTO schema_migrations (version, checksum, description) "
            "VALUES (%s, %s, %s) ON CONFLICT (version) DO NOTHING",
            (version, checksum, desc)
        )
    print(f"  ✅ {version} (bootstrapped tracker)")


def _applied_migrations(conn) -> dict[str, str]:
    """Returns {version: checksum} for all applied migrations."""
    with conn.cursor() as cur:
        cur.execute("SELECT version, checksum FROM schema_migrations")
        return {row[0]: row[1] for row in cur.fetchall()}


def _apply_migration(conn, filepath: str, version: str, dry_run: bool) -> None:
    """Apply a single migration file within a transaction."""
    checksum = _file_checksum(filepath)
    desc     = os.path.basename(filepath)

    if dry_run:
        print(f"  [DRY RUN] Would apply {version}: {desc}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        sql = f.read()

    try:
        with conn:  # auto-commit / rollback context
            cur = conn.cursor()
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (version, checksum, description) "
                "VALUES (%s, %s, %s)",
                (version, checksum, desc)
            )
        print(f"  ✅ {version}: {desc}")
    except Exception as exc:
        print(f"  ❌ {version} FAILED: {exc}")
        raise MigrationError(f"Migration {version} failed: {exc}") from exc


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def run_migrations(dry_run: bool = False) -> list[str]:
    """
    Apply all pending migrations from the migrations/ directory.

    Args:
        dry_run: If True, print what would be applied without executing.

    Returns:
        List of version strings that were applied (or would be applied).

    Raises:
        MigrationError: if a migration fails or a checksum mismatch is detected.
    """
    print("=== AuraPredict Migration Runner (v2) ===")
    print(f"    Directory : {MIGRATIONS_DIR}")
    print(f"    Mode      : {'DRY RUN' if dry_run else 'APPLY'}")
    print()

    conn  = get_conn()
    files = _migration_files()

    # ── Bootstrap tracker if first run ────────────────────────────────────────
    v001 = files[0]  # always V2_001__schema_migrations_tracker.sql
    if not _tracker_exists(conn):
        if dry_run:
            print(f"  [DRY RUN] Would bootstrap tracker from {os.path.basename(v001)}")
        else:
            _bootstrap_tracker(conn, v001)

    # ── Apply remaining migrations ────────────────────────────────────────────
    applied  = _applied_migrations(conn) if _tracker_exists(conn) else {}
    executed = []

    for filepath in files:
        version = _extract_version(filepath)

        if version in applied:
            # Verify checksum — migrations must be immutable once applied
            stored   = applied[version]
            current  = _file_checksum(filepath)
            if stored and stored != current:
                conn.close()
                raise MigrationError(
                    f"Checksum mismatch for {version}! "
                    f"The migration file was modified after being applied. "
                    f"Stored: {stored[:8]}…  Current: {current[:8]}…"
                )
            continue  # already applied, skip

        _apply_migration(conn, filepath, version, dry_run)
        if not dry_run:
            executed.append(version)

    conn.close()

    if executed:
        print(f"\n  {len(executed)} migration(s) applied successfully.")
    elif not dry_run:
        print("  No pending migrations — database is up to date.")

    return executed


def status() -> list[dict]:
    """
    Print and return the current migration status.

    Returns:
        List of dicts with keys: version, description, applied_at, status.
    """
    print("=== Migration Status ===")

    conn  = get_conn()
    files = _migration_files()

    if not _tracker_exists(conn):
        print("  schema_migrations table does not exist — no migrations applied yet.")
        conn.close()
        return []

    applied = _applied_migrations(conn)
    conn.close()

    rows = []
    for filepath in files:
        version = _extract_version(filepath)
        desc    = os.path.basename(filepath)
        is_app  = version in applied
        rows.append({
            "version":     version,
            "description": desc,
            "status":      "APPLIED" if is_app else "PENDING",
        })
        mark = "✅" if is_app else "⏳"
        print(f"  {mark}  {version}  {desc}")

    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AuraPredict Migration Runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be applied without executing")
    parser.add_argument("--status",  action="store_true",
                        help="Show current migration status")
    args = parser.parse_args()

    if args.status:
        status()
    else:
        run_migrations(dry_run=args.dry_run)
