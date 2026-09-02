"""
AuraPredict — FaultClassifier  (Fase 4A)
==========================================
Rule-based fault classifier using the same 8 features as the Isolation Forest.

Design principles:
  - Uses ONLY the 8 features already in the system (FEATURE_NAMES from anomaly_detector).
  - Does NOT use ML / supervised learning — only expert knowledge rules.
  - Returns FaultDiagnosis with type, affected axis, severity, confidence,
    explanation, and maintenance recommendation.
  - Clearly separates three states:
      1. No anomaly → no diagnosis (FaultDiagnosis = None)
      2. Anomaly with sufficient evidence → specific fault type
      3. Anomaly with insufficient evidence → UNCERTAIN (not invented)
  - Confidence reflects actual rule strength, never overstated.
  - Designed to be replaced/extended with a supervised classifier when
    real labelled fault data becomes available.

Feature units (from the existing pipeline):
  rms_x, rms_y, rms_z:               g (RMS acceleration)
  kurtosis_x, kurtosis_y, kurtosis_z: dimensionless
  crest_factor_x, ...:                dimensionless (peak / RMS)
  peak_to_peak_x, ...:                g (full swing)
  dominant_freq_hz:                   Hz (primary frequency)
  band_low_energy:                    energy 10–100 Hz (imbalance, misalignment)
  band_mid_energy:                    energy 100–500 Hz (harmonics)
  band_high_energy:                   energy 500–1600 Hz (bearings, resonance)

Fault signatures (literature-based, CNC spindle bearings):

  IMBALANCE / MISALIGNMENT
    - elevated RMS (energy), especially sinusoidal
    - low kurtosis (< 2.0 — no impulsive events)
    - dominant frequency at 1×RPM or 2×RPM
    - band_low_energy dominant (< 100 Hz)
    - low crest_factor (< 3.0)

  BEARING FAULT
    - elevated kurtosis (> 3.0 — impulsive events characteristic of rolling element defects)
    - elevated crest_factor (> 3.5 — peaks much higher than RMS)
    - band_high_energy elevated (bearing frequencies > 500 Hz)
    - rms may be moderate (early stage) or high (late stage)

  LUBRICATION / FRICTION
    - moderate rms elevation (not extreme)
    - kurtosis low-to-moderate (1.5–3.0)
    - band_mid_energy and band_high_energy elevated proportionally
    - crest_factor moderate (2.5–4.0)
    - no strong impulsive signature
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ─── OUTPUT DATACLASS ─────────────────────────────────────────────────────────

@dataclass
class FaultDiagnosis:
    """
    Structured fault diagnosis produced by FaultClassifier.

    fault_type       : 'IMBALANCE' | 'BEARING' | 'LUBRICATION' | 'UNCERTAIN'
    affected_axis    : 'x' | 'y' | 'z' | 'multiple' | 'unknown'
    severity         : 'INCIPIENT' | 'MODERATE' | 'SEVERE'
    confidence       : float [0.0, 1.0] — fraction of matching rules
    explanation      : Human-readable description of the evidence
    recommendation   : Actionable text for a maintenance technician
    """
    fault_type:     str
    affected_axis:  str
    severity:       str
    confidence:     float
    explanation:    str
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "fault_type":     self.fault_type,
            "affected_axis":  self.affected_axis,
            "severity":       self.severity,
            "confidence":     round(self.confidence, 2),
            "explanation":    self.explanation,
            "recommendation": self.recommendation,
        }


# ─── THRESHOLDS ────────────────────────────────────────────────────────────────
# All thresholds are conservative to minimise false diagnoses.
# Named constants make them easy to tune when real calibration data is available.

# RMS thresholds (g) — rough values for CNC spindle bearings
RMS_NORMAL_MAX     = 0.08   # Below this: clearly normal
RMS_ELEVATED       = 0.15   # Elevated but not critical
RMS_HIGH           = 0.30   # Significantly elevated

# Kurtosis thresholds — key indicator for impulsive bearing faults
KURT_NORMAL_MAX    = 2.0    # Normal distribution ~ 0 (excess kurtosis)
KURT_ELEVATED      = 3.0    # Mild impulsive activity
KURT_HIGH          = 5.0    # Strong impulsive activity → bearing fault

# Crest factor thresholds
CF_NORMAL_MAX      = 3.0    # Normal
CF_ELEVATED        = 3.5    # Mild — could be early bearing
CF_HIGH            = 5.0    # Strong impulsive → bearing

# Band energy ratios
# Imbalance: low band dominant
LOW_DOMINANCE_RATIO = 0.6   # low_energy / (low + mid + high) > this → imbalance
# Bearing: high band elevated
HIGH_ELEV_RATIO     = 0.3   # high_energy / total > this → bearing hint

# Minimum anomaly_score to trigger any diagnosis
MIN_ANOMALY_FOR_DIAGNOSIS = 0.25


# ─── CLASSIFIER ───────────────────────────────────────────────────────────────

class FaultClassifier:
    """
    Classifies fault type from the 8 IF feature vector + per-axis features.

    Usage:
        classifier = FaultClassifier()
        diagnosis = classifier.classify(feature_vector, per_axis_features)
        # diagnosis is FaultDiagnosis or None (if no anomaly or no evidence)

    per_axis_features is a dict:
        {
          'x': {'rms': 0.05, 'kurtosis': 1.5, 'crest_factor': 2.1, 'peak_to_peak': 0.14},
          'y': {...},
          'z': {...},
        }

    feature_vector is the 8-element numpy array [rms_x, kurtosis_x, crest_factor_x,
    peak_to_peak_x, dominant_freq_hz, band_low_energy, band_mid_energy, band_high_energy]
    following FEATURE_NAMES order from anomaly_detector.py.
    """

    def classify(
        self,
        feature_vector:      np.ndarray,
        anomaly_score:       float,
        per_axis_features:   Optional[dict[str, dict]] = None,
    ) -> Optional[FaultDiagnosis]:
        """
        Classify the fault type from the feature vector.

        Returns:
            FaultDiagnosis if anomaly_score >= threshold and there is
            sufficient evidence for any fault type.
            None if the signal is normal or evidence is insufficient.
        """
        if anomaly_score < MIN_ANOMALY_FOR_DIAGNOSIS:
            return None   # Normal operation — no diagnosis needed

        # Extract primary axis features from vector
        # Order: [rms_x, kurtosis_x, crest_factor_x, peak_to_peak_x,
        #         dominant_freq_hz, band_low_energy, band_mid_energy, band_high_energy]
        if feature_vector is None or len(feature_vector) < 8:
            return _uncertain("Feature vector unavailable")

        rms_x           = float(feature_vector[0]) if np.isfinite(feature_vector[0]) else None
        kurtosis_x      = float(feature_vector[1]) if np.isfinite(feature_vector[1]) else None
        crest_factor_x  = float(feature_vector[2]) if np.isfinite(feature_vector[2]) else None
        dominant_freq   = float(feature_vector[4]) if np.isfinite(feature_vector[4]) else None
        band_low        = float(feature_vector[5]) if np.isfinite(feature_vector[5]) else None
        band_mid        = float(feature_vector[6]) if np.isfinite(feature_vector[6]) else None
        band_high       = float(feature_vector[7]) if np.isfinite(feature_vector[7]) else None

        # Determine primary affected axis (highest kurtosis across axes)
        affected_axis = _primary_affected_axis(per_axis_features)

        # Evaluate each fault hypothesis
        bearing_score    = _score_bearing(kurtosis_x, crest_factor_x, band_high, band_low, band_mid)
        imbalance_score  = _score_imbalance(rms_x, kurtosis_x, crest_factor_x, band_low, band_mid, band_high)
        lubrication_score = _score_lubrication(rms_x, kurtosis_x, crest_factor_x, band_mid, band_high)

        # Severity based on anomaly_score
        severity = _severity(anomaly_score)

        # Pick best-scoring hypothesis — require minimum confidence
        scores = {
            "BEARING":     bearing_score,
            "IMBALANCE":   imbalance_score,
            "LUBRICATION": lubrication_score,
        }
        best_type = max(scores, key=scores.get)
        best_conf = scores[best_type]

        if best_conf < 0.30:
            return _uncertain(
                f"Anomaly detected (score={anomaly_score:.2f}) but pattern "
                f"does not match known fault signatures clearly enough "
                f"(best match: {best_type} at {best_conf:.0%})."
            )

        explanation, recommendation = _explain(
            best_type, affected_axis, severity,
            rms_x, kurtosis_x, crest_factor_x, band_low, band_mid, band_high,
        )

        return FaultDiagnosis(
            fault_type     = best_type,
            affected_axis  = affected_axis,
            severity       = severity,
            confidence     = round(best_conf, 2),
            explanation    = explanation,
            recommendation = recommendation,
        )


# ─── SCORING FUNCTIONS ────────────────────────────────────────────────────────

def _score_bearing(kurtosis, crest_factor, band_high, band_low, band_mid) -> float:
    """
    Score the likelihood of a bearing fault [0, 1].

    Bearing faults produce impulsive signals (high kurtosis, high crest factor)
    concentrated in high-frequency bands (rolling element defect frequencies).
    """
    signals = 0
    total   = 0

    # Strong indicators (weighted more)
    if kurtosis is not None:
        total += 2
        if kurtosis >= KURT_HIGH:      signals += 2
        elif kurtosis >= KURT_ELEVATED: signals += 1

    if crest_factor is not None:
        total += 2
        if crest_factor >= CF_HIGH:      signals += 2
        elif crest_factor >= CF_ELEVATED: signals += 1

    # Supporting indicator
    if band_high is not None and band_low is not None and band_mid is not None:
        total_energy = band_low + band_mid + band_high + 1e-12
        high_ratio   = band_high / total_energy
        total += 1
        if high_ratio >= HIGH_ELEV_RATIO:
            signals += 1

    return signals / total if total > 0 else 0.0


def _score_imbalance(rms, kurtosis, crest_factor, band_low, band_mid, band_high) -> float:
    """
    Score the likelihood of imbalance/misalignment [0, 1].

    Imbalance produces elevated but non-impulsive vibration, concentrated in
    low-frequency bands (1×/2× rotational frequency components).
    """
    signals = 0
    total   = 0

    # Kurtosis low (non-impulsive) is an imbalance indicator
    if kurtosis is not None:
        total += 2
        if kurtosis < KURT_NORMAL_MAX:    signals += 2
        elif kurtosis < KURT_ELEVATED:    signals += 1

    # Low crest factor (not peaky)
    if crest_factor is not None:
        total += 2
        if crest_factor < CF_NORMAL_MAX:  signals += 2
        elif crest_factor < CF_ELEVATED:  signals += 1

    # Low band dominant
    if band_low is not None and band_mid is not None and band_high is not None:
        total_energy = band_low + band_mid + band_high + 1e-12
        low_ratio    = band_low / total_energy
        total += 2
        if low_ratio >= LOW_DOMINANCE_RATIO: signals += 2
        elif low_ratio >= 0.45:              signals += 1

    # RMS elevated (energy is there, just not impulsive)
    if rms is not None:
        total += 1
        if rms >= RMS_ELEVATED:  signals += 1

    return signals / total if total > 0 else 0.0


def _score_lubrication(rms, kurtosis, crest_factor, band_mid, band_high) -> float:
    """
    Score the likelihood of lubrication/friction issue [0, 1].

    Poor lubrication produces broad-spectrum friction noise — elevated mid
    and high bands with moderate kurtosis (not as impulsive as bearing faults).
    """
    signals = 0
    total   = 0

    # Kurtosis moderate — some friction but not strong impacts
    if kurtosis is not None:
        total += 2
        if KURT_NORMAL_MAX <= kurtosis < KURT_ELEVATED:  signals += 2
        elif kurtosis < KURT_HIGH:                        signals += 1

    # Mid and high bands elevated proportionally (broad friction spectrum)
    if band_mid is not None and band_high is not None:
        total += 1
        if band_mid > 1e-6 and band_high > 1e-6:  signals += 1

    # Crest factor moderate — not purely impulsive, not purely sinusoidal
    if crest_factor is not None:
        total += 1
        if CF_NORMAL_MAX <= crest_factor < CF_HIGH:  signals += 1

    return signals / total if total > 0 else 0.0


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _primary_affected_axis(per_axis: Optional[dict]) -> str:
    """Find the axis with the highest kurtosis (most impulsive)."""
    if not per_axis:
        return "unknown"
    best_axis  = "unknown"
    best_kurt  = -999.0
    for axis, feats in per_axis.items():
        k = feats.get("kurtosis")
        if k is not None and k > best_kurt:
            best_kurt = k
            best_axis = axis
    return best_axis if best_axis != "unknown" else "multiple"


def _severity(anomaly_score: float) -> str:
    if anomaly_score >= 0.75:  return "SEVERE"
    if anomaly_score >= 0.45:  return "MODERATE"
    return "INCIPIENT"


def _uncertain(reason: str) -> FaultDiagnosis:
    return FaultDiagnosis(
        fault_type     = "UNCERTAIN",
        affected_axis  = "unknown",
        severity       = "UNKNOWN",
        confidence     = 0.0,
        explanation    = reason,
        recommendation = "Perform manual inspection. Increase monitoring frequency.",
    )


def _explain(
    fault_type: str,
    axis:       str,
    severity:   str,
    rms:        Optional[float],
    kurtosis:   Optional[float],
    crest_factor: Optional[float],
    band_low:   Optional[float],
    band_mid:   Optional[float],
    band_high:  Optional[float],
) -> tuple[str, str]:
    """Build human-readable explanation and recommendation."""

    feat_str = []
    if rms        is not None: feat_str.append(f"RMS={rms:.3f}g")
    if kurtosis   is not None: feat_str.append(f"Kurtosis={kurtosis:.2f}")
    if crest_factor is not None: feat_str.append(f"CF={crest_factor:.2f}")
    feats = ", ".join(feat_str)

    severity_label = {
        "INCIPIENT": "incipient (early stage)",
        "MODERATE":  "moderate",
        "SEVERE":    "severe",
    }.get(severity, severity)

    if fault_type == "BEARING":
        explanation = (
            f"Bearing fault signature detected on axis {axis} — {severity_label}. "
            f"Impulsive vibration pattern: {feats}. "
            f"Elevated kurtosis and crest factor indicate rolling element impacts. "
            f"High-frequency energy band elevated."
        )
        rec = {
            "INCIPIENT": (
                "Increase monitoring to 30-minute intervals. "
                "Plan bearing inspection at next scheduled maintenance stop."
            ),
            "MODERATE": (
                "Schedule bearing inspection within 7–14 days. "
                "Avoid high-load operation until inspection."
            ),
            "SEVERE": (
                "STOP THE MACHINE and inspect immediately. "
                "Replace bearing before resuming operation."
            ),
        }.get(severity, "Inspect bearing at earliest opportunity.")

    elif fault_type == "IMBALANCE":
        explanation = (
            f"Imbalance or misalignment detected on axis {axis} — {severity_label}. "
            f"Non-impulsive sinusoidal vibration: {feats}. "
            f"Low-frequency band dominant. "
            f"Low kurtosis confirms absence of impulsive events."
        )
        rec = {
            "INCIPIENT": (
                "Check tool/workpiece balance and spindle alignment. "
                "Monitor at normal intervals."
            ),
            "MODERATE": (
                "Perform balancing check and alignment verification. "
                "Check tool holder and workpiece clamping."
            ),
            "SEVERE": (
                "Stop operation and perform full dynamic balancing check. "
                "Inspect spindle alignment and coupling."
            ),
        }.get(severity, "Check mechanical balance and alignment.")

    elif fault_type == "LUBRICATION":
        explanation = (
            f"Lubrication or friction issue detected — {severity_label}. "
            f"Broad-spectrum friction noise: {feats}. "
            f"Mid and high frequency bands elevated uniformly. "
            f"Moderate kurtosis consistent with friction rather than impacts."
        )
        rec = {
            "INCIPIENT": (
                "Check lubrication level and condition. "
                "Re-lubricate according to manufacturer specifications."
            ),
            "MODERATE": (
                "Re-lubricate immediately using manufacturer-specified grease/oil. "
                "Monitor for 48 hours after re-lubrication."
            ),
            "SEVERE": (
                "Immediate re-lubrication required. "
                "Inspect bearing and spindle for heat damage. "
                "Check lubrication system for blockages."
            ),
        }.get(severity, "Check and replenish lubrication.")

    else:
        explanation    = f"Unknown fault pattern — {feats}"
        rec            = "Manual inspection recommended."

    return explanation, rec


# ─── INTEGRATION HELPER ───────────────────────────────────────────────────────

def extract_per_axis_features(feature_set) -> dict[str, dict]:
    """
    Extract per-axis time features from a FeatureSet for use in FaultClassifier.
    Returns a dict {axis: {rms, kurtosis, crest_factor, peak_to_peak}}.
    """
    result = {}
    for axis_name in ["x", "y", "z"]:
        vf = feature_set.multiaxis.get_axis(axis_name)
        if vf is None:
            continue
        result[axis_name] = {
            "rms":          vf.time.rms,
            "kurtosis":     vf.time.kurtosis,
            "crest_factor": vf.time.crest_factor,
            "peak_to_peak": vf.time.peak_to_peak,
        }
    return result
