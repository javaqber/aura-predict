"""
AuraPredict — HealthScoreCalculator (Fase 2C)
===============================================
Computes trend and slope from the health score history of a machine.

The instantaneous health_score is computed inside AnomalyDetector.
This module adds the temporal dimension: how is the score evolving over time?

Trend and slope are persisted alongside each health_score row via
database_v2.repositories.registrar_health_score(trend=..., slope=...).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Minimum number of historical scores needed to compute a reliable slope.
MIN_SCORES_FOR_SLOPE: int = 3

# Slope thresholds (points per day) for trend classification.
SLOPE_STABLE_BAND:    float = 1.0   # |slope| < 1.0 pt/day → stable
SLOPE_IMPROVING_MIN:  float = 1.0   # slope ≥ 1.0 → improving
SLOPE_DEGRADING_MAX:  float = -1.0  # slope ≤ -1.0 → degrading
CRITICAL_THRESHOLD:   int   = 25    # score ≤ 25 → critical regardless of slope


class HealthScoreCalculator:
    """
    Computes trend and slope from a window of recent health scores.

    Usage (called once per acquisition cycle, after the new score is known):

        calculator = HealthScoreCalculator()
        history = obtener_historial_health(maquina_id, dias=7)  # newest first
        trend, slope = calculator.compute(history, current_score)
        registrar_health_score(maquina_id, empresa_id,
                               score=current_score, trend=trend, slope=slope,
                               lectura_id=lectura_id)
    """

    def compute(
        self,
        recent_scores: list[dict],
        current_score: Optional[int],
    ) -> tuple[str, Optional[float]]:
        """
        Compute trend and slope from recent health score history.

        Args:
            recent_scores: List of health score dicts from obtener_historial_health().
                           Expected keys: 'score' (int), 'timestamp' (datetime).
                           Sorted newest-first (as the repository returns them).
            current_score: The just-computed health score for this cycle.
                           Used to determine 'critical' trend regardless of slope.

        Returns:
            (trend, slope) where:
              trend : 'stable' | 'improving' | 'degrading' | 'critical' | 'unknown'
              slope : score change per day (negative = health worsening).
                      None if there are not enough data points.
        """
        # Critical takes precedence — report it immediately
        if current_score is not None and current_score <= CRITICAL_THRESHOLD:
            slope = self._compute_slope(recent_scores)
            return "critical", slope

        # Not enough history → unknown
        if len(recent_scores) < MIN_SCORES_FOR_SLOPE:
            return "unknown", None

        slope = self._compute_slope(recent_scores)
        if slope is None:
            return "unknown", None

        if slope >= SLOPE_IMPROVING_MIN:
            trend = "improving"
        elif slope <= SLOPE_DEGRADING_MAX:
            trend = "degrading"
        else:
            trend = "stable"

        return trend, slope

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_slope(scores: list[dict]) -> Optional[float]:
        """
        Compute slope (score points per day) via linear regression.

        Args:
            scores: List of health score records, newest first.
                    Each dict must have 'score' (int) and 'timestamp'.

        Returns:
            Slope in points/day, or None if computation fails.
        """
        if len(scores) < MIN_SCORES_FOR_SLOPE:
            return None

        try:
            # Convert timestamps to days-since-oldest (x axis)
            from datetime import timezone as tz
            timestamps = []
            for row in scores:
                ts = row.get("timestamp")
                if ts is None:
                    return None
                # psycopg2 returns timezone-aware datetimes; handle both
                if hasattr(ts, "timestamp"):
                    timestamps.append(ts.timestamp())
                else:
                    return None

            score_values = [int(row["score"]) for row in scores]

            # scores are newest-first; reverse to oldest-first for polyfit
            timestamps   = list(reversed(timestamps))
            score_values = list(reversed(score_values))

            # Convert to days since first timestamp
            t0   = timestamps[0]
            days = np.array([(t - t0) / 86400.0 for t in timestamps])
            vals = np.array(score_values, dtype=float)

            # Linear regression: slope is the coefficient of the day axis
            if np.ptp(days) < 1e-6:   # all timestamps identical → no slope
                return 0.0

            coeffs = np.polyfit(days, vals, 1)
            return round(float(coeffs[0]), 3)   # points per day

        except Exception:
            return None
