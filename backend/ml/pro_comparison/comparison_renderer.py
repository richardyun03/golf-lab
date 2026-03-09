"""
Render comparison overlay frames showing user's skeleton with pro angle annotations.

Draws the user's skeleton plus color-coded angle comparison labels showing
the difference between the user's measured angles and the selected pro's values.
"""

import math
import cv2
import numpy as np
from app.schemas.analysis import PoseKeypoint
from ml.swing_analysis.skeleton_renderer import (
    CONNECTIONS, BONE_COLOR, BONE_THICKNESS, MIN_CONF,
    _kp_dict, _has, _px, _mid, _angle_at, _draw_arc, _run_measurement,
    PHASE_GUIDES,
)

# Colors (BGR)
USER_JOINT = (0, 255, 180)       # green - user joints
PRO_COLOR = (255, 160, 40)       # blue/cyan - pro reference
MATCH_COLOR = (0, 220, 100)      # green - angles match well
DIFF_COLOR = (0, 120, 255)       # orange - angles differ
LABEL_BG = (0, 0, 0)

# Which metrics map to which phase measurements
METRIC_TO_PHASE_MEASUREMENT = {
    "address": {
        "spine_tilt_address_degrees": "spine_tilt",
        "lead_knee_flex_address_degrees": "left_knee",
    },
    "takeaway": {
        "spine_tilt_address_degrees": "spine_tilt",
    },
    "backswing": {
        "spine_tilt_address_degrees": "spine_tilt",
    },
    "top": {
        "spine_tilt_address_degrees": "spine_tilt",
    },
    "downswing": {
        "spine_tilt_address_degrees": "spine_tilt",
    },
    "impact": {
        "spine_tilt_address_degrees": "spine_tilt",
        "lead_knee_flex_impact_degrees": "left_knee",
    },
    "follow_through": {
        "spine_tilt_address_degrees": "spine_tilt",
    },
    "finish": {
        "lead_knee_flex_impact_degrees": "left_knee",
    },
}

# Label names for display
MEASUREMENT_LABELS = {
    "spine_tilt": "Spine",
    "left_elbow": "Lead elbow",
    "right_elbow": "Trail elbow",
    "left_wrist": "Wrist",
    "left_knee": "Lead knee",
    "right_knee": "Trail knee",
}


def _label_bg(canvas, text, x, y, color, scale=0.42):
    """Draw text with a background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(canvas, (x - 2, y - th - 4), (x + tw + 4, y + 4), LABEL_BG, -1)
    cv2.putText(canvas, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def _comparison_label(canvas, label, user_val, pro_val, x, y):
    """Draw a two-line comparison label: user angle vs pro angle."""
    diff = abs(user_val - pro_val)
    is_close = diff < 8  # within 8 degrees = good match
    color = MATCH_COLOR if is_close else DIFF_COLOR

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    line1 = f"{label}"
    line2 = f"You: {user_val:.0f}  Pro: {pro_val:.0f}"

    (tw1, th1), _ = cv2.getTextSize(line1, font, scale, 1)
    (tw2, th2), _ = cv2.getTextSize(line2, font, scale, 1)
    tw = max(tw1, tw2)
    total_h = th1 + th2 + 10

    # Background
    cv2.rectangle(canvas, (x - 3, y - th1 - 5), (x + tw + 6, y + th2 + 8), LABEL_BG, -1)
    cv2.rectangle(canvas, (x - 3, y - th1 - 5), (x + tw + 6, y + th2 + 8), color, 1)

    # Text
    cv2.putText(canvas, line1, (x, y), font, scale, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (x, y + th2 + 6), font, scale, color, 1, cv2.LINE_AA)

    # Difference indicator
    if not is_close:
        arrow = "+" if user_val > pro_val else ""
        diff_text = f"{arrow}{user_val - pro_val:.0f}"
        dx = x + tw + 10
        cv2.putText(canvas, diff_text, (dx, y + th2 + 6), font, 0.35, DIFF_COLOR, 1, cv2.LINE_AA)


def draw_comparison_frame(
    frame: np.ndarray,
    keypoints: list[PoseKeypoint],
    phase: str,
    pro_metrics: dict,
) -> np.ndarray:
    """
    Draw the user's skeleton with pro comparison angle annotations.

    Shows the user's measured angle vs the pro's known angle at each
    relevant joint for the given phase.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    kps = _kp_dict(keypoints)

    # Draw bones (slightly dimmed)
    for name_a, name_b in CONNECTIONS:
        a = kps.get(name_a)
        b = kps.get(name_b)
        if a and b and a.confidence >= MIN_CONF and b.confidence >= MIN_CONF:
            cv2.line(canvas, _px(a, w, h), _px(b, w, h), BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    # Draw joints
    for kp in keypoints:
        if kp.confidence >= MIN_CONF:
            pt = _px(kp, w, h)
            cv2.circle(canvas, pt, 5, USER_JOINT, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)

    # Get phase-specific guides and draw comparison annotations
    guides = PHASE_GUIDES.get(phase, [])

    # Also collect any pro metric values that map to measurements in this phase
    phase_metric_map = METRIC_TO_PHASE_MEASUREMENT.get(phase, {})

    # Build a dict of pro values for each measurement name
    pro_vals_by_measurement = {}
    for metric_key, meas_name in phase_metric_map.items():
        if metric_key in pro_metrics:
            pro_vals_by_measurement[meas_name] = pro_metrics[metric_key]

    for meas_name, lbl, ideal_lo, ideal_hi, off_x, off_y in guides:
        result = _run_measurement(meas_name, kps, w, h)
        if result is None:
            continue

        user_angle, vertex, pt_a, pt_c = result

        # Get pro value: prefer direct metric mapping, fall back to ideal range midpoint
        pro_val = pro_vals_by_measurement.get(meas_name)
        if pro_val is None:
            pro_val = (ideal_lo + ideal_hi) / 2.0

        diff = abs(user_angle - pro_val)
        is_close = diff < 8
        arc_color = MATCH_COLOR if is_close else DIFF_COLOR

        # Draw arc at vertex
        _draw_arc(canvas, vertex, pt_a, pt_c, arc_color, radius=28)

        # Draw reference line for spine tilt
        if meas_name == "spine_tilt":
            hm = vertex
            for i in range(0, 200, 12):
                y1 = hm[1] - i
                y2 = hm[1] - min(i + 6, 200)
                cv2.line(canvas, (hm[0], y1), (hm[0], y2), (255, 200, 80), 1, cv2.LINE_AA)
            cv2.line(canvas, vertex, pt_a, arc_color, 2, cv2.LINE_AA)

        # Comparison label
        display_label = MEASUREMENT_LABELS.get(meas_name, lbl)
        lx = vertex[0] + off_x
        ly = vertex[1] + off_y
        _comparison_label(canvas, display_label, user_angle, pro_val, lx, ly)

    # Pro name watermark (top-right)
    pro_name = pro_metrics.get("_name", "Pro")
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(f"vs {pro_name}", font, 0.5, 1)
    cv2.rectangle(canvas, (w - tw - 14, 6), (w - 4, th + 16), LABEL_BG, -1)
    cv2.putText(canvas, f"vs {pro_name}", (w - tw - 10, th + 10), font, 0.5, PRO_COLOR, 1, cv2.LINE_AA)

    return canvas


def render_comparison_phase_frames(
    raw_frames: list[np.ndarray],
    keypoints_by_frame: list[list[PoseKeypoint]],
    swing_phases: dict,
    pro_metrics: dict,
    pro_name: str,
) -> dict[str, np.ndarray]:
    """
    For each swing phase, render the user's skeleton with pro comparison annotations.
    Returns {phase_value: rendered_frame}.
    """
    metrics_with_name = {**pro_metrics, "_name": pro_name}
    result = {}
    for phase, frame_idx in swing_phases.items():
        phase_key = phase.value if hasattr(phase, "value") else str(phase)
        idx = min(frame_idx, len(raw_frames) - 1)
        frame = raw_frames[idx]
        kps = keypoints_by_frame[idx] if idx < len(keypoints_by_frame) else []
        result[phase_key] = draw_comparison_frame(frame, kps, phase=phase_key, pro_metrics=metrics_with_name)
    return result
