"""
AuraPredict — Edge Scheduler v2  (Fase 3)
==========================================
Runs EdgePipeline in a continuous 24/7 loop with adaptive acquisition intervals
based on the current health score.

Interval selection (from SchedulerConfig — all values configurable in YAML):
  health >= 75  (OK - Sano)   → interval_normal_minutes  (default 120 min)
  health 50–74  (ADVERTENCIA) → interval_watch_minutes   (default  30 min)
  health < 50   (ALERTA/NOK)  → interval_alert_minutes   (default   5 min)
  health = None (cold start)  → interval_normal_minutes  (conservative)

Design decisions (approved Fase 3):
  - 24/7 operation: no time-of-day restrictions.
  - Resilient loop: exceptions in one cycle are logged and the scheduler
    continues. A single failed cycle never stops the Edge.
  - Configurable intervals from YAML: no hardcoded values in logic.
  - All components injectable for tests (EdgePipeline, sleep_fn).

Usage:
    # Production
    python src/edge_scheduler.py --config config/machines/torno_cnc_1.yaml

    # Or via environment variable
    MAQUINA_CONFIG=config/machines/torno_cnc_1.yaml python src/edge_scheduler.py
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Optional

# Add src/ to path
sys.path.insert(0, os.path.dirname(__file__))


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = os.getenv(
    "MAQUINA_CONFIG",
    os.path.join(os.path.dirname(__file__), "../config/machines/example_cnc.yaml"),
)


# ── State ──────────────────────────────────────────────────────────────────────

class SchedulerState:
    """Mutable state shared across cycles."""
    def __init__(self) -> None:
        self.running:           bool          = True
        self.current_interval:  int           = 120   # minutes, updated each cycle
        self.last_health_score: Optional[int] = None
        self.cycles_ok:         int           = 0
        self.cycles_error:      int           = 0


# ── Interval logic ─────────────────────────────────────────────────────────────

def interval_for_health(
    health_score:             Optional[int],
    interval_normal_minutes:  int,
    interval_watch_minutes:   int,
    interval_alert_minutes:   int,
    health_watch_threshold:   int = 75,
    health_warning_threshold: int = 50,
) -> int:
    """
    Select the acquisition interval based on the current health score.

    All thresholds and intervals come from configuration — nothing hardcoded.

    Args:
        health_score:            Latest health score (None during cold start).
        interval_normal_minutes: Interval for healthy operation.
        interval_watch_minutes:  Interval for degraded-but-watchable state.
        interval_alert_minutes:  Interval for anomaly/critical state.
        health_watch_threshold:  Score above which we use the normal interval.
        health_warning_threshold: Score above which we use the watch interval.

    Returns:
        Interval in minutes.
    """
    if health_score is None:
        return interval_normal_minutes   # cold start → conservative
    if health_score >= health_watch_threshold:
        return interval_normal_minutes
    if health_score >= health_warning_threshold:
        return interval_watch_minutes
    return interval_alert_minutes


# ── Main scheduler ─────────────────────────────────────────────────────────────

def run_scheduler(
    config_path: str,
    *,
    pipeline_factory: Optional[Callable] = None,
    sleep_fn:         Callable           = time.sleep,
) -> None:
    """
    Main scheduler loop. Runs until SIGINT/SIGTERM or state.running=False.

    Args:
        config_path:      Path to the machine YAML config file.
        pipeline_factory: Optional injectable for tests. If None, creates a
                          real EdgePipeline from the config.
        sleep_fn:         Optional injectable for tests (replaces time.sleep).
    """
    from edge.config.edge_config import EdgeConfig
    from edge.sensors.mock_sensor import MockSensor, MockSensorParams

    state = SchedulerState()

    # ── Graceful shutdown on SIGINT/SIGTERM ────────────────────────────────────
    def _handle_signal(signum, frame):
        print(f"\n[Scheduler] Signal {signum} received — shutting down gracefully…")
        state.running = False

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Load configuration ─────────────────────────────────────────────────────
    try:
        config = EdgeConfig.from_yaml(config_path)
    except Exception as exc:
        print(f"[Scheduler] FATAL: cannot load config from {config_path}: {exc}")
        sys.exit(1)

    sc = config.scheduler  # SchedulerConfig

    if not sc.enabled:
        print("[Scheduler] scheduler.enabled=false in config — exiting.")
        return

    state.current_interval = sc.interval_normal_minutes

    _print_banner(config, sc)

    # ── Create pipeline ────────────────────────────────────────────────────────
    if pipeline_factory is not None:
        pipeline = pipeline_factory(config)
    else:
        pipeline = _create_pipeline(config)

    try:
        pipeline.startup()
        print("[Scheduler] Pipeline started successfully.\n")
    except Exception as exc:
        print(f"[Scheduler] FATAL: pipeline.startup() failed: {exc}")
        sys.exit(1)

    # ── Main loop ──────────────────────────────────────────────────────────────
    while state.running:
        cycle_start = datetime.now(timezone.utc)
        ts = cycle_start.strftime("%Y-%m-%d %H:%M:%S UTC")

        print(f"[{ts}] ▶ Starting acquisition cycle…")

        health_score = _run_one_cycle(pipeline, state)

        # Update interval based on new health score
        new_interval = interval_for_health(
            health_score,
            interval_normal_minutes  = sc.interval_normal_minutes,
            interval_watch_minutes   = sc.interval_watch_minutes,
            interval_alert_minutes   = sc.interval_alert_minutes,
            health_watch_threshold   = config.anomaly.health_watch,
            health_warning_threshold = config.anomaly.health_warning,
        )

        if new_interval != state.current_interval:
            print(f"[Scheduler] Interval changed: {state.current_interval}min "
                  f"→ {new_interval}min (health={health_score})")
            state.current_interval = new_interval

        state.last_health_score = health_score

        elapsed_s = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        sleep_s   = max(0, state.current_interval * 60 - elapsed_s)

        print(f"[Scheduler] Cycle done in {elapsed_s:.1f}s. "
              f"Next in {state.current_interval}min. "
              f"(ok={state.cycles_ok}, err={state.cycles_error})")

        # Sleep in 10s chunks so SIGINT is handled promptly
        _interruptible_sleep(sleep_s, state, sleep_fn)

    # ── Shutdown ───────────────────────────────────────────────────────────────
    print("\n[Scheduler] Shutting down pipeline…")
    try:
        pipeline.shutdown()
    except Exception:
        pass
    print("[Scheduler] Stopped.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_one_cycle(pipeline, state: SchedulerState) -> Optional[int]:
    """
    Execute one EdgePipeline.run_once() cycle safely.
    Returns the health_score if available, None otherwise.
    Exceptions are caught so the scheduler never stops from a single cycle.
    """
    try:
        feature_set = pipeline.run_once()
        state.cycles_ok += 1

        if feature_set is None:
            print("[Scheduler] Cycle skipped: sensor error on all axes.")
            return state.last_health_score  # keep previous interval

        ar = feature_set.anomaly_result
        if ar is not None:
            resultado = ar.resultado
            health    = ar.health_score
            riesgo    = ar.nivel_riesgo
            score_str = f"health={health}" if health is not None else "health=?"
            print(f"[Scheduler] {resultado} | {riesgo} | {score_str} "
                  f"| score={ar.anomaly_score:.3f}")
            return health

        print("[Scheduler] Cycle complete (no anomaly result yet).")
        return None

    except Exception as exc:
        state.cycles_error += 1
        print(f"[Scheduler] ⚠ Cycle error (continuing): {type(exc).__name__}: {exc}")
        return state.last_health_score


def _interruptible_sleep(
    total_s: float,
    state:   SchedulerState,
    sleep_fn: Callable,
) -> None:
    """Sleep in 10-second chunks so SIGINT is handled promptly."""
    remaining = total_s
    while remaining > 0 and state.running:
        chunk = min(10.0, remaining)
        sleep_fn(chunk)
        remaining -= chunk


def _create_pipeline(config):
    """Create a real EdgePipeline for production use."""
    from edge.pipeline.pipeline import EdgePipeline
    from edge.sensors.mock_sensor import MockSensor, MockSensorParams

    # Sensor: MockSensor for now (real ADXL345 sensor arrives in a future phase)
    sensor = MockSensor(config.sensor, MockSensorParams())

    return EdgePipeline(config=config, sensor=sensor)


def _print_banner(config, sc) -> None:
    print("=" * 60)
    print("  🔮 AuraPredict — Edge Scheduler v2")
    print("=" * 60)
    print(f"  Machine   : {config.machine.machine_id}")
    print(f"  maquina_id: {config.machine.maquina_id or '(pending resolution)'}")
    print(f"  empresa_id: {config.machine.empresa_id}")
    print(f"  Mode      : 24/7 (no time restrictions)")
    print(f"  Intervals : normal={sc.interval_normal_minutes}min "
          f"/ watch={sc.interval_watch_minutes}min "
          f"/ alert={sc.interval_alert_minutes}min")
    print("=" * 60 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AuraPredict Edge Scheduler v2"
    )
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG,
        help="Path to the machine YAML config file "
             f"(default: $MAQUINA_CONFIG or {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()
    run_scheduler(args.config)


if __name__ == "__main__":
    main()
