"""
Metrics-based pro swing comparison.

Compares a user's computed swing metrics against a database of pro golfer
metric profiles using weighted normalized distance across each metric dimension.
"""

import math
from app.schemas.analysis import SwingMetrics
from app.schemas.comparison import SwingMatchResult, ProProfile
from ml.pro_comparison.pro_database import PRO_PROFILES, TOUR_AVERAGES

# How much each metric matters for overall similarity (sum to 1.0)
METRIC_WEIGHTS = {
    "hip_rotation_degrees": 0.12,
    "shoulder_rotation_degrees": 0.12,
    "x_factor_degrees": 0.15,
    "spine_tilt_address_degrees": 0.12,
    "spine_tilt_change_degrees": 0.10,
    "lead_knee_flex_address_degrees": 0.08,
    "lead_knee_flex_impact_degrees": 0.08,
    "tempo_ratio": 0.23,
}

# Normalization ranges — expected spread of each metric across all skill levels
METRIC_RANGES = {
    "hip_rotation_degrees": 30,        # 20-50 range
    "shoulder_rotation_degrees": 40,   # 60-100 range
    "x_factor_degrees": 30,           # 20-50 range
    "spine_tilt_address_degrees": 20,  # 20-40 range
    "spine_tilt_change_degrees": 15,   # 0-15 range
    "lead_knee_flex_address_degrees": 30,  # 130-160 range
    "lead_knee_flex_impact_degrees": 30,   # 150-180 range
    "tempo_ratio": 2.0,               # 1.5-3.5 range
}

# Labels for metric categories
METRIC_LABELS = {
    "hip_rotation_degrees": "Hip rotation",
    "shoulder_rotation_degrees": "Shoulder turn",
    "x_factor_degrees": "X-Factor",
    "spine_tilt_address_degrees": "Spine tilt at address",
    "spine_tilt_change_degrees": "Spine angle maintenance",
    "lead_knee_flex_address_degrees": "Knee flex at address",
    "lead_knee_flex_impact_degrees": "Knee extension at impact",
    "tempo_ratio": "Tempo ratio",
}


def _get_metric_val(metrics: SwingMetrics, key: str) -> float | None:
    return getattr(metrics, key, None)


def _similarity_score(user_metrics: SwingMetrics, pro_metrics: dict) -> float:
    """
    Compute weighted similarity score (0-1) between user metrics and pro profile.
    Higher = more similar.
    """
    total_weight = 0.0
    weighted_sim = 0.0

    for key, weight in METRIC_WEIGHTS.items():
        user_val = _get_metric_val(user_metrics, key)
        pro_val = pro_metrics.get(key)
        if user_val is None or pro_val is None:
            continue

        norm_range = METRIC_RANGES[key]
        diff = abs(user_val - pro_val) / norm_range
        sim = max(0.0, 1.0 - diff)
        weighted_sim += sim * weight
        total_weight += weight

    if total_weight < 0.01:
        return 0.0

    return weighted_sim / total_weight


def _find_similarities_and_differences(
    user_metrics: SwingMetrics, pro_metrics: dict
) -> tuple[list[str], list[str]]:
    """Find key similarities (close metrics) and differences (divergent metrics)."""
    comparisons = []

    for key in METRIC_WEIGHTS:
        user_val = _get_metric_val(user_metrics, key)
        pro_val = pro_metrics.get(key)
        if user_val is None or pro_val is None:
            continue

        norm_range = METRIC_RANGES[key]
        diff = abs(user_val - pro_val) / norm_range
        label = METRIC_LABELS[key]
        comparisons.append((diff, key, label, user_val, pro_val))

    comparisons.sort(key=lambda x: x[0])

    similarities = []
    differences = []

    for diff, key, label, user_val, pro_val in comparisons:
        if diff < 0.15:
            similarities.append(f"{label}: {user_val:.0f} (pro: {pro_val:.0f})")
        elif diff > 0.3:
            direction = "higher" if user_val > pro_val else "lower"
            differences.append(f"{label}: {user_val:.0f} vs {pro_val:.0f} ({direction})")

    return similarities[:4], differences[:4]


def _classify_archetype(user_metrics: SwingMetrics) -> str:
    """Classify the user's swing style archetype based on their metrics."""
    x_factor = _get_metric_val(user_metrics, "x_factor_degrees")
    hip_rot = _get_metric_val(user_metrics, "hip_rotation_degrees")
    shoulder_rot = _get_metric_val(user_metrics, "shoulder_rotation_degrees")
    tempo = _get_metric_val(user_metrics, "tempo_ratio")
    spine_change = _get_metric_val(user_metrics, "spine_tilt_change_degrees")

    if x_factor is not None and x_factor > 48:
        if hip_rot and hip_rot > 44:
            return "Modern Rotational"
        return "X-Factor Dominant"
    if shoulder_rot is not None and shoulder_rot < 80:
        return "Compact"
    if tempo is not None and tempo > 3.3:
        return "Deliberate / Classic"
    if tempo is not None and tempo < 2.5:
        return "Quick Tempo"
    if spine_change is not None and spine_change < 1.5:
        return "Stack & Tilt Influenced"

    return "Balanced Modern"


def _matching_phases(user_metrics: SwingMetrics, pro_metrics: dict) -> list[str]:
    """Identify which swing phases the user matches the pro most closely."""
    phase_metrics = {
        "Setup": ["spine_tilt_address_degrees", "lead_knee_flex_address_degrees"],
        "Backswing": ["hip_rotation_degrees", "shoulder_rotation_degrees", "x_factor_degrees"],
        "Transition": ["tempo_ratio"],
        "Impact": ["lead_knee_flex_impact_degrees", "spine_tilt_change_degrees"],
    }

    matches = []
    for phase, keys in phase_metrics.items():
        sims = []
        for key in keys:
            user_val = _get_metric_val(user_metrics, key)
            pro_val = pro_metrics.get(key)
            if user_val is not None and pro_val is not None:
                diff = abs(user_val - pro_val) / METRIC_RANGES[key]
                sims.append(max(0, 1 - diff))
        if sims and sum(sims) / len(sims) > 0.7:
            matches.append(phase)

    return matches


def compare_to_pros(user_metrics: SwingMetrics, top_k: int = 5) -> list[SwingMatchResult]:
    """Compare user metrics against all pro profiles, return top K matches."""
    scored = []

    for pro_data in PRO_PROFILES:
        score = _similarity_score(user_metrics, pro_data["metrics"])
        similarities, differences = _find_similarities_and_differences(
            user_metrics, pro_data["metrics"]
        )
        phases = _matching_phases(user_metrics, pro_data["metrics"])

        profile = ProProfile(
            pro_id=pro_data["pro_id"],
            name=pro_data["name"],
            tour=pro_data["tour"],
            swing_style=pro_data["swing_style"],
            known_for=pro_data["known_for"],
        )

        match = SwingMatchResult(
            pro=profile,
            similarity_score=round(score, 3),
            matching_phases=phases,
            key_similarities=similarities,
            key_differences=differences,
        )
        scored.append((score, match))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


def get_tour_comparison(user_metrics: SwingMetrics, tour: str = "PGA") -> dict:
    """Compare user metrics against tour averages."""
    avg = TOUR_AVERAGES.get(tour, TOUR_AVERAGES["PGA"])
    result = {}

    for key in METRIC_WEIGHTS:
        user_val = _get_metric_val(user_metrics, key)
        tour_val = avg.get(key)
        if user_val is not None and tour_val is not None:
            result[key] = {
                "user": round(user_val, 1),
                "tour_avg": tour_val,
                "diff": round(user_val - tour_val, 1),
                "label": METRIC_LABELS[key],
            }

    return result


def classify_swing(user_metrics: SwingMetrics) -> str:
    return _classify_archetype(user_metrics)
