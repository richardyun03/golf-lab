"""Draw pose skeleton overlays with angle guide annotations on video frames."""

import math
import cv2
import numpy as np
from app.schemas.analysis import PoseKeypoint

# MediaPipe pose connections
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Colors (BGR)
JOINT_COLOR = (0, 255, 180)
BONE_COLOR = (0, 220, 160)
GUIDE_GOOD = (0, 200, 100)       # green - within ideal range
GUIDE_WARN = (0, 180, 255)       # orange - outside ideal range
GUIDE_LINE = (255, 200, 80)      # light blue - reference lines
LABEL_BG = (0, 0, 0)
JOINT_RADIUS = 5
BONE_THICKNESS = 2
MIN_CONF = 0.3


def _kp_dict(keypoints: list[PoseKeypoint]) -> dict[str, PoseKeypoint]:
    return {kp.name: kp for kp in keypoints}


def _has(kps: dict, *names: str) -> bool:
    return all(n in kps and kps[n].confidence >= MIN_CONF for n in names)


def _px(kp: PoseKeypoint, w: int, h: int) -> tuple[int, int]:
    return (int(kp.x * w), int(kp.y * h))


def _mid(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def _angle_at(a: tuple, vertex: tuple, c: tuple) -> float:
    """Angle in degrees at the vertex point, formed by rays vertex->a and vertex->c."""
    va = (a[0] - vertex[0], a[1] - vertex[1])
    vc = (c[0] - vertex[0], c[1] - vertex[1])
    dot = va[0] * vc[0] + va[1] * vc[1]
    m1 = math.hypot(*va)
    m2 = math.hypot(*vc)
    if m1 < 1 or m2 < 1:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (m1 * m2)))
    return math.degrees(math.acos(cos_a))


def _in_range(val: float, lo: float, hi: float) -> bool:
    return lo <= val <= hi


def _draw_arc(canvas, center, pt_a, pt_b, color, radius=30):
    """Draw a small arc at center between the directions to pt_a and pt_b."""
    ang_a = math.degrees(math.atan2(-(pt_a[1] - center[1]), pt_a[0] - center[0]))
    ang_b = math.degrees(math.atan2(-(pt_b[1] - center[1]), pt_b[0] - center[0]))
    start = min(ang_a, ang_b)
    end = max(ang_a, ang_b)
    if end - start > 180:
        start, end = end, start + 360
    cv2.ellipse(canvas, center, (radius, radius), 0, -end, -start, color, 2, cv2.LINE_AA)


def _label(canvas, text, x, y, color, scale=0.42):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(canvas, (x - 2, y - th - 4), (x + tw + 4, y + 4), LABEL_BG, -1)
    cv2.putText(canvas, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


# ── Measurement functions ────────────────────────────────────────────
# Each returns (angle_degrees, vertex_point, point_a, point_c) or None.
# The angle is measured AT the vertex, between rays vertex->a and vertex->c.

def _spine_tilt(kps, w, h):
    """Angle of spine (shoulder-mid to hip-mid) from vertical. Measured at hip midpoint."""
    if not _has(kps, "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        return None
    sm = _mid(_px(kps["left_shoulder"], w, h), _px(kps["right_shoulder"], w, h))
    hm = _mid(_px(kps["left_hip"], w, h), _px(kps["right_hip"], w, h))
    vert = (hm[0], hm[1] - 200)  # straight up from hips
    angle = _angle_at(sm, hm, vert)
    return angle, hm, sm, vert


def _elbow_angle(kps, w, h, side):
    """Angle at the elbow (shoulder-ELBOW-wrist). Straighter = higher angle."""
    s, e, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    if not _has(kps, s, e, wr):
        return None
    ps = _px(kps[s], w, h)
    pe = _px(kps[e], w, h)
    pw = _px(kps[wr], w, h)
    angle = _angle_at(ps, pe, pw)
    return angle, pe, ps, pw


def _wrist_angle(kps, w, h, side):
    """Angle at the wrist (elbow-WRIST-index). Measures wrist cock/hinge."""
    e, wr, idx = f"{side}_elbow", f"{side}_wrist", f"{side}_index"
    if not _has(kps, e, wr, idx):
        return None
    pe = _px(kps[e], w, h)
    pw = _px(kps[wr], w, h)
    pi = _px(kps[idx], w, h)
    angle = _angle_at(pe, pw, pi)
    return angle, pw, pe, pi


def _knee_angle(kps, w, h, side):
    """Angle at the knee (hip-KNEE-ankle)."""
    hip, knee, ankle = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
    if not _has(kps, hip, knee, ankle):
        return None
    ph = _px(kps[hip], w, h)
    pk = _px(kps[knee], w, h)
    pa = _px(kps[ankle], w, h)
    angle = _angle_at(ph, pk, pa)
    return angle, pk, ph, pa


# ── Phase-specific guide config ──────────────────────────────────────
# Each entry: (measurement_func_name, label, ideal_lo, ideal_hi, label_offset_x, label_offset_y)

PHASE_GUIDES: dict[str, list[tuple]] = {
    "address": [
        ("spine_tilt",  "Spine tilt",     25, 40,   40, -50),
        ("left_knee",   "Lead knee",     140, 165,   30,   0),
    ],
    "takeaway": [
        ("spine_tilt",  "Spine tilt",     25, 40,   40, -50),
        ("left_elbow",  "Lead elbow",    160, 180,   30, -10),
    ],
    "backswing": [
        ("spine_tilt",  "Spine tilt",     25, 40,   40, -50),
        ("left_elbow",  "Lead elbow",    160, 180,   30, -10),
    ],
    "top": [
        ("spine_tilt",  "Spine tilt",     25, 45,   40, -50),
        ("left_elbow",  "Lead elbow",    155, 180,   30, -10),
        ("left_wrist",  "Wrist hinge",    70, 130,   25, -20),
    ],
    "downswing": [
        ("spine_tilt",  "Spine tilt",     25, 40,   40, -50),
        ("right_elbow", "Trail elbow",    60, 110,  -10,  25),
    ],
    "impact": [
        ("spine_tilt",  "Spine tilt",     25, 40,   40, -50),
        ("left_elbow",  "Lead elbow",    155, 180,   30, -10),
        ("left_knee",   "Lead knee",     155, 180,   30,   0),
    ],
    "follow_through": [
        ("spine_tilt",  "Spine tilt",     15, 40,   40, -50),
        ("left_elbow",  "Lead elbow",    140, 180,   30, -10),
    ],
    "finish": [
        ("left_knee",   "Lead knee",     155, 180,   30,   0),
    ],
}


def _run_measurement(name, kps, w, h):
    """Dispatch to the right measurement function by name."""
    if name == "spine_tilt":
        return _spine_tilt(kps, w, h)
    elif name == "left_elbow":
        return _elbow_angle(kps, w, h, "left")
    elif name == "right_elbow":
        return _elbow_angle(kps, w, h, "right")
    elif name == "left_wrist":
        return _wrist_angle(kps, w, h, "left")
    elif name == "right_wrist":
        return _wrist_angle(kps, w, h, "right")
    elif name == "left_knee":
        return _knee_angle(kps, w, h, "left")
    elif name == "right_knee":
        return _knee_angle(kps, w, h, "right")
    return None


def draw_guides(canvas, kps, w, h, phase):
    """Draw angle guide annotations for the given phase."""
    guides = PHASE_GUIDES.get(phase, [])

    for meas_name, lbl, ideal_lo, ideal_hi, off_x, off_y in guides:
        result = _run_measurement(meas_name, kps, w, h)
        if result is None:
            continue

        angle, vertex, pt_a, pt_c = result
        good = _in_range(angle, ideal_lo, ideal_hi)
        color = GUIDE_GOOD if good else GUIDE_WARN

        # Draw arc at the vertex joint
        _draw_arc(canvas, vertex, pt_a, pt_c, color, radius=28)

        # Draw reference line for spine tilt (vertical dashed line)
        if meas_name == "spine_tilt":
            hm = vertex
            for i in range(0, 200, 12):
                y1 = hm[1] - i
                y2 = hm[1] - min(i + 6, 200)
                cv2.line(canvas, (hm[0], y1), (hm[0], y2), GUIDE_LINE, 1, cv2.LINE_AA)
            cv2.line(canvas, vertex, pt_a, color, 2, cv2.LINE_AA)

        # Label with angle and ideal range
        lx = vertex[0] + off_x
        ly = vertex[1] + off_y
        text = f"{lbl}: {angle:.0f} ({ideal_lo}-{ideal_hi})"
        _label(canvas, text, lx, ly, color)


def draw_skeleton(
    frame: np.ndarray,
    keypoints: list[PoseKeypoint],
    phase: str | None = None,
) -> np.ndarray:
    """Draw skeleton overlay with optional phase-specific angle guides."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    kps = _kp_dict(keypoints)

    # Draw bones
    for name_a, name_b in CONNECTIONS:
        a = kps.get(name_a)
        b = kps.get(name_b)
        if a and b and a.confidence >= MIN_CONF and b.confidence >= MIN_CONF:
            cv2.line(canvas, _px(a, w, h), _px(b, w, h), BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    # Draw joints
    for kp in keypoints:
        if kp.confidence >= MIN_CONF:
            pt = _px(kp, w, h)
            cv2.circle(canvas, pt, JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, JOINT_RADIUS + 1, (0, 0, 0), 1, cv2.LINE_AA)

    # Phase-specific guides
    if phase:
        draw_guides(canvas, kps, w, h, phase)

    return canvas


def render_phase_frames(
    raw_frames: list[np.ndarray],
    keypoints_by_frame: list[list[PoseKeypoint]],
    swing_phases: dict,
) -> dict[str, np.ndarray]:
    """
    For each swing phase, extract the frame and render skeleton + angle guides.
    Returns {phase_value: rendered_frame}.
    """
    result = {}
    for phase, frame_idx in swing_phases.items():
        phase_key = phase.value if hasattr(phase, "value") else str(phase)
        idx = min(frame_idx, len(raw_frames) - 1)
        frame = raw_frames[idx]
        kps = keypoints_by_frame[idx] if idx < len(keypoints_by_frame) else []
        result[phase_key] = draw_skeleton(frame, kps, phase=phase_key)
    return result
