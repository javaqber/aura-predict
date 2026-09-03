"""
AuraPredict — Edge Logging Configuration  (Fase 5A)
=====================================================
Sets up Python structured logging for the entire Edge process.

Named loggers by subsystem — use getLogger(__name__) in each module:
  edge.sensors.*        → sensor/hardware events
  edge.pipeline.*       → acquisition cycle events
  edge.anomaly.*        → detection + diagnosis events
  edge.sync.*           → Storage sync events
  edge.buffer           → LocalBuffer events
  edge_scheduler        → scheduler lifecycle events

Log levels:
  DEBUG    → detailed internal state (disabled in production)
  INFO     → normal operation events (cycle start/end, values)
  WARNING  → recoverable issues (sensor retry, buffer flush fail)
  ERROR    → non-fatal errors (SMTP failure, DB offline)
  CRITICAL → fatal — used only before shutdown
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level:       str            = "INFO",
    log_file:    Optional[str]  = None,
    rich_format: bool           = False,
) -> None:
    """
    Configure structured logging for the Edge process.

    Call once at scheduler startup before any other imports use logging.

    Args:
        level:       Logging level string: DEBUG / INFO / WARNING / ERROR.
                     Reads LOG_LEVEL env var if not specified.
        log_file:    Optional file path to write logs in addition to stdout.
        rich_format: If True use a richer format with module names (for files).
    """
    import os
    effective_level = os.getenv("LOG_LEVEL", level).upper()

    fmt_console = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    fmt_file    = "%(asctime)s [%(levelname)-8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    datefmt     = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(getattr(logging, effective_level, logging.INFO))

    # Remove existing handlers (avoid duplication on repeated calls)
    root.handlers.clear()

    # Console handler — always present
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt_console, datefmt=datefmt))
    root.addHandler(ch)

    # File handler — optional
    if log_file:
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(fmt_file, datefmt=datefmt))
            root.addHandler(fh)
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Cannot open log file %s: %s", log_file, exc
            )

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper for getLogger."""
    return logging.getLogger(name)
