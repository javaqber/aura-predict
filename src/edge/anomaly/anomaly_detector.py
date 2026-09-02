"""
AuraPredict — Anomaly Detector (Fase 2C)
=========================================
Detects anomalies in CNC vibration feature vectors.

Two implementations follow the same AnomalyDetector interface:

  ZScoreDetector         — cold start, no model file required.
                           Works from the first few readings.
                           Uses baseline statistics (μ, σ) to compute
                           per-feature z-scores.

  IsolationForestDetector — production mode, requires a trained model.
                            Uses scikit-learn IsolationForest.
                            Activated when enough 'normal' readings
                            have been accumulated (see MachineBaselineManager).

Design rules:
  - Both return AnomalyResult with identical fields.
  - Neither accesses the database directly.
  - Neither modifies FeatureSet — they consume the extracted feature vector.
  - Classification (resultado / nivel_riesgo) is deterministic and consistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# scikit-learn — already in requirements.txt
from sklearn.ensemble import IsolationForest


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

# The 8-feature vector fed to the Isolation Forest.
# These names match exactly the columns in lecturas_cnc_v2 and the
# keys returned by obtener_lecturas_para_baseline().
FEATURE_NAMES: list[str] = [
    "rms_x",            # overall energy
    "kurtosis_x",       # impulsiveness — key for bearing faults
    "crest_factor_x",   # early fault indicator (rises before RMS)
    "peak_to_peak_x",   # mechanical looseness
    "dominant_freq_hz", # frequency shift
    "band_low_energy",  # 10–100 Hz: imbalance, misalignment
    "band_mid_energy",  # 100–500 Hz: harmonics
    "band_high_energy", # 500–1600 Hz: bearings, resonances
]

# Minimum signal quality score to compute a reliable health score.
# Below this threshold, health_score is set to None ("unreliable measurement").
QUALITY_MIN: float = 0.5

# Default Z-score threshold for ZScoreDetector.
# A feature at Z_THRESHOLD σ produces anomaly_score = 1.0 (fully anomalous).
Z_THRESHOLD_DEFAULT: float = 3.0

# Default denominator for IF score normalisation.
# IsolationForest.score_samples() returns values near [-0.5, 0].
# anomaly_score = clip(-raw_score / IF_SCORE_SCALE, 0, 1)
IF_SCORE_SCALE_DEFAULT: float = 0.5


# ─── OUTPUT DATACLASS ─────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """
    Complete output of one anomaly detection cycle.

    Fields:
        anomaly_score    : float [0.0, 1.0]. 0 = normal, 1 = fully anomalous.
        health_score     : int [0, 100] or None if signal quality is too low.
        resultado        : Human-readable status string (persisted to BD).
        nivel_riesgo     : Risk level string (persisted to BD).
        diagnostico      : Short explanation of top contributing features.
        model_version_id : FK to machine_model_registry (None for ZScore/cold start).
        is_cold_start    : True while baseline is still being built.
        algorithm        : 'zscore' | 'isolation_forest'
    """
    anomaly_score:    float
    health_score:     Optional[int]
    resultado:        str
    nivel_riesgo:     str
    diagnostico:      str
    model_version_id: Optional[int]
    is_cold_start:    bool
    algorithm:        str
    # Fase 4A: structured fault diagnosis (None when no anomaly or uncertain)
    fault_diagnosis:  Optional[object] = field(default=None)


# ─── CLASSIFICATION HELPERS ───────────────────────────────────────────────────

def _classify_health(health_score: int) -> tuple[str, str]:
    """
    Map health_score [0, 100] → (resultado, nivel_riesgo).

    Ranges (configurable via AnomalyConfig in the future):
      90–100 → HEALTHY  / Bajo
      75–89  → WATCH    / Bajo
      50–74  → WARNING  / Medio
      25–49  → ALERT    / Alto
       0–24  → CRITICAL / CRÍTICO
    """
    if health_score >= 75:
        return "OK - Sano", "Bajo"
    elif health_score >= 50:
        return "ADVERTENCIA", "Medio"
    elif health_score >= 25:
        return "ALERTA", "Alto"
    else:
        return "NOK - Anomalía Detectada", "CRÍTICO"


def _describe_contributors(
    z_scores: dict[str, float],
    threshold: float,
) -> str:
    """
    Describe the features most deviated from baseline.

    Returns an empty string if nothing is notably anomalous.
    Returns a brief human-readable description of the top 3 contributors.
    """
    # Sort by z-score descending, keep only those above half the threshold
    significant = [
        (name, z)
        for name, z in sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
        if z > threshold * 0.5
    ]
    if not significant:
        return ""
    parts = [f"{name} ({z:.1f}σ)" for name, z in significant[:3]]
    return "Desviación: " + ", ".join(parts)


def _impute_nans(
    vector: np.ndarray,
    baseline_stats: dict[str, dict],
) -> np.ndarray:
    """
    Replace NaN values in the feature vector with the baseline mean.

    This handles the case where a sensor axis failed the quality check,
    leaving some features as None/NaN. Rather than blocking detection,
    we impute conservatively (mean = no anomaly contribution).
    """
    result = vector.astype(float).copy()
    for i, name in enumerate(FEATURE_NAMES):
        if np.isnan(result[i]):
            stats = baseline_stats.get(name, {})
            result[i] = float(stats.get("mean", 0.0))
    return result


def _compute_health(
    anomaly_score: float,
    signal_quality: float,
) -> Optional[int]:
    """
    Convert anomaly_score and signal_quality to a health_score [0, 100].

    Quality dampening:
      - signal_quality = 1.0 → full trust in the score, no dampening
      - signal_quality = 0.5 → 30% dampening (score blended toward midpoint 50)
      - signal_quality < QUALITY_MIN → return None (unreliable measurement)

    Blending formula:
      health = q * (100 * (1 - anomaly)) + (1 - q) * 50
    This avoids reporting extreme values (0 or 100) with poor signal quality.
    """
    if signal_quality < QUALITY_MIN:
        return None
    raw_health  = 100.0 * (1.0 - anomaly_score)
    blended     = signal_quality * raw_health + (1.0 - signal_quality) * 50.0
    return max(0, min(100, round(blended)))


# ─── ABSTRACT INTERFACE ───────────────────────────────────────────────────────

class AnomalyDetector(ABC):
    """
    Abstract interface for anomaly detectors.

    All implementations must accept the same 8-element feature vector
    (FEATURE_NAMES) and baseline statistics, and return AnomalyResult.

    The detector does NOT access the database.
    The detector does NOT modify FeatureSet.
    """

    @abstractmethod
    def analyze(
        self,
        feature_vector:   np.ndarray,           # shape (8,)
        baseline_stats:   dict[str, dict],       # {feature: {mean, std, p5, p50, p95}}
        signal_quality:   float = 1.0,
        model_version_id: Optional[int] = None,
    ) -> AnomalyResult:
        """
        Analyze one feature vector and return a complete AnomalyResult.

        Args:
            feature_vector:   Numpy array of 8 features (FEATURE_NAMES order).
                              NaN values are imputed with baseline means.
            baseline_stats:   Statistical baseline per feature.
                              Empty dict → cold start with no reference.
            signal_quality:   Signal quality score [0, 1] from DataQualityResult.
            model_version_id: FK to machine_model_registry for the result.
        """
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """
        True if this detector can produce reliable anomaly detection.

        ZScoreDetector: always True (no training needed).
        IsolationForestDetector: True when a trained model is loaded.
        """
        ...


# ─── Z-SCORE DETECTOR (cold start) ────────────────────────────────────────────

class ZScoreDetector(AnomalyDetector):
    """
    Cold-start anomaly detector using Z-scores against baseline statistics.

    Does not require a model file. Works from the first few readings.
    All results have is_cold_start=True to signal that the pipeline is
    still accumulating its normal-operation baseline.

    When baseline_stats is empty (no readings yet), returns a special
    "learning" result with resultado='OK - Aprendiendo'.
    """

    def __init__(self, z_threshold: float = Z_THRESHOLD_DEFAULT) -> None:
        self._z_threshold = z_threshold

    @property
    def is_ready(self) -> bool:
        return True  # No training needed

    def analyze(
        self,
        feature_vector:   np.ndarray,
        baseline_stats:   dict[str, dict],
        signal_quality:   float = 1.0,
        model_version_id: Optional[int] = None,
    ) -> AnomalyResult:

        # No baseline yet → pure learning mode
        if not baseline_stats:
            return AnomalyResult(
                anomaly_score    = 0.0,
                health_score     = None,
                resultado        = "OK - Aprendiendo",
                nivel_riesgo     = "Pendiente",
                diagnostico      = "Acumulando lecturas de operación normal",
                model_version_id = None,
                is_cold_start    = True,
                algorithm        = "zscore",
            )

        # Impute NaN values with baseline means
        vector = _impute_nans(feature_vector, baseline_stats)

        # Compute per-feature z-scores
        z_scores: dict[str, float] = {}
        for i, name in enumerate(FEATURE_NAMES):
            stats = baseline_stats.get(name, {})
            std   = float(stats.get("std", 0.0))
            mean  = float(stats.get("mean", 0.0))
            if std > 1e-9:
                z_scores[name] = abs(float(vector[i]) - mean) / std
            else:
                z_scores[name] = 0.0

        max_z         = max(z_scores.values()) if z_scores else 0.0
        anomaly_score = min(max_z / self._z_threshold, 1.0)

        # Health score with quality adjustment
        health_score = _compute_health(anomaly_score, signal_quality)

        if health_score is None:
            resultado, nivel_riesgo = "SENSOR_ERROR", "Desconocido"
            diagnostico = "Calidad de señal insuficiente"
        else:
            resultado, nivel_riesgo = _classify_health(health_score)
            diagnostico = _describe_contributors(z_scores, self._z_threshold)

        return AnomalyResult(
            anomaly_score    = round(anomaly_score, 4),
            health_score     = health_score,
            resultado        = resultado,
            nivel_riesgo     = nivel_riesgo,
            diagnostico      = diagnostico,
            model_version_id = model_version_id,
            is_cold_start    = True,
            algorithm        = "zscore",
        )


# ─── ISOLATION FOREST DETECTOR (production) ───────────────────────────────────

class IsolationForestDetector(AnomalyDetector):
    """
    Production anomaly detector using scikit-learn IsolationForest.

    Requires a trained model (IsolationForest instance).
    Used when enough 'normal' readings have been accumulated
    (controlled by MachineBaselineManager).

    The IF score_samples() output is in approximately [-0.5, 0]:
      0    → statistically indistinguishable from training data (normal)
      -0.5 → maximum anomaly the model can detect

    Normalisation: anomaly_score = clip(-raw / IF_SCORE_SCALE, 0, 1)

    Z-scores from the baseline are still computed (for diagnostico),
    since IsolationForest does not explain which features drove the score.
    """

    def __init__(
        self,
        model:            IsolationForest,
        model_version_id: Optional[int] = None,
        z_threshold:      float = Z_THRESHOLD_DEFAULT,
        score_scale:      float = IF_SCORE_SCALE_DEFAULT,
    ) -> None:
        self._model             = model
        self._model_version_id  = model_version_id
        self._z_threshold       = z_threshold
        self._score_scale       = score_scale

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def analyze(
        self,
        feature_vector:   np.ndarray,
        baseline_stats:   dict[str, dict],
        signal_quality:   float = 1.0,
        model_version_id: Optional[int] = None,
    ) -> AnomalyResult:

        vid    = model_version_id or self._model_version_id
        vector = _impute_nans(feature_vector, baseline_stats)

        # Isolation Forest score
        raw_score     = float(self._model.score_samples(vector.reshape(1, -1))[0])
        anomaly_score = float(np.clip(-raw_score / self._score_scale, 0.0, 1.0))

        # Z-scores for human-readable diagnostico (IF doesn't explain itself)
        z_scores: dict[str, float] = {}
        for i, name in enumerate(FEATURE_NAMES):
            stats = baseline_stats.get(name, {})
            std   = float(stats.get("std", 0.0))
            mean  = float(stats.get("mean", 0.0))
            if std > 1e-9:
                z_scores[name] = abs(float(vector[i]) - mean) / std
            else:
                z_scores[name] = 0.0

        # Health score with quality adjustment
        health_score = _compute_health(anomaly_score, signal_quality)

        if health_score is None:
            resultado, nivel_riesgo = "SENSOR_ERROR", "Desconocido"
            diagnostico = "Calidad de señal insuficiente"
        else:
            resultado, nivel_riesgo = _classify_health(health_score)
            diagnostico = _describe_contributors(z_scores, self._z_threshold)

        return AnomalyResult(
            anomaly_score    = round(anomaly_score, 4),
            health_score     = health_score,
            resultado        = resultado,
            nivel_riesgo     = nivel_riesgo,
            diagnostico      = diagnostico,
            model_version_id = vid,
            is_cold_start    = False,
            algorithm        = "isolation_forest",
        )
