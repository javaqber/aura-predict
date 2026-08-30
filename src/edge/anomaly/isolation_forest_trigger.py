"""
AuraPredict — IsolationForestTrigger (Fase 2C)
================================================
Implements AnomalyTrigger to decide when to capture a RAW signal window.

Replaces PlaceholderAnomalyTrigger (which always returned False) with a real
trigger based on health_score and a configurable cooldown.

Trigger conditions (ALL must be true):
  1. feature_set.anomaly_result is set (anomaly detection ran this cycle)
  2. is_cold_start is False  (baseline is ready — not still learning)
  3. health_score is not None (signal quality was sufficient to compute it)
  4. nivel_riesgo is not 'Desconocido' (not a SENSOR_ERROR)
  5. health_score < capture_threshold (score is in WARNING zone or worse)
  6. cooldown period has elapsed since the last capture

Cooldown is tracked in memory (monotonic time). It resets on process restart,
which is acceptable — a restarting Pi is an unusual event and one extra capture
is harmless.
"""

from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

from ..pipeline.models import AnomalyTrigger

if TYPE_CHECKING:
    from ..pipeline.models import FeatureSet


class IsolationForestTrigger(AnomalyTrigger):
    """
    Production AnomalyTrigger that fires when health_score drops below
    a configurable threshold, with a cooldown between captures.

    Configuration (from AnomalyConfig, passed at construction):
      capture_threshold : int   — health_score below which capture fires (default 50)
      cooldown_seconds  : float — minimum seconds between captures (default 3600)

    Example:
        trigger = IsolationForestTrigger(capture_threshold=50, cooldown_seconds=3600)
        if trigger.should_capture(feature_set):
            raw_capture.capture(reading, feature_set, lectura_id)
    """

    def __init__(
        self,
        capture_threshold: int   = 50,
        cooldown_seconds:  float = 3600.0,
    ) -> None:
        self._threshold        = capture_threshold
        self._cooldown_s       = cooldown_seconds
        self._last_capture_at: Optional[float] = None   # time.monotonic() value

    def should_capture(self, feature_set: "FeatureSet") -> bool:
        """
        Return True if a RAW event window should be captured this cycle.

        All conditions are checked in order from cheapest to most specific.
        """
        ar = feature_set.anomaly_result

        # ── Condition 1: anomaly detection must have run ───────────────────────
        if ar is None:
            return False

        # ── Condition 2: not cold start (baseline not ready yet) ──────────────
        if ar.is_cold_start:
            return False

        # ── Condition 3: health_score must be computable ──────────────────────
        if ar.health_score is None:
            return False

        # ── Condition 4: not a sensor error ───────────────────────────────────
        if ar.nivel_riesgo == "Desconocido":
            return False

        # ── Condition 5: score below capture threshold ────────────────────────
        if ar.health_score >= self._threshold:
            return False

        # ── Condition 6: cooldown check ───────────────────────────────────────
        now = time.monotonic()
        if self._last_capture_at is not None:
            elapsed = now - self._last_capture_at
            if elapsed < self._cooldown_s:
                return False

        # All conditions met → update cooldown timestamp and signal capture
        self._last_capture_at = now
        return True

    @property
    def last_capture_at(self) -> Optional[float]:
        """Monotonic timestamp of the last RAW capture (for testing/monitoring)."""
        return self._last_capture_at

    def reset_cooldown(self) -> None:
        """Force-reset the cooldown. Useful for testing."""
        self._last_capture_at = None
