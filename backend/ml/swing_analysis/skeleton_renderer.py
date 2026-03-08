"""Draw pose skeleton overlays on video frames."""

import cv2
import numpy as np
from app.schemas.analysis import PoseKeypoint

# MediaPipe pose connections (pairs of landmark names)
CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Left arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Right arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Colors (BGR)
JOINT_COLOR = (0, 255, 180)       # cyan-green
BONE_COLOR = (0, 220, 160)        # slightly darker
JOINT_RADIUS = 5
BONE_THICKNESS = 2
MIN_CONFIDENCE = 0.3


def draw_skeleton(
    frame: np.ndarray,
    keypoints: list[PoseKeypoint],
) -> np.ndarray:
    """Draw skeleton overlay on a frame. Returns a copy with the overlay."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    kp_dict: dict[str, PoseKeypoint] = {kp.name: kp for kp in keypoints}

    # Draw bones first (under joints)
    for name_a, name_b in CONNECTIONS:
        a = kp_dict.get(name_a)
        b = kp_dict.get(name_b)
        if a and b and a.confidence >= MIN_CONFIDENCE and b.confidence >= MIN_CONFIDENCE:
            pt_a = (int(a.x * w), int(a.y * h))
            pt_b = (int(b.x * w), int(b.y * h))
            cv2.line(canvas, pt_a, pt_b, BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    # Draw joints
    for kp in keypoints:
        if kp.confidence >= MIN_CONFIDENCE:
            pt = (int(kp.x * w), int(kp.y * h))
            cv2.circle(canvas, pt, JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, JOINT_RADIUS + 1, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def render_phase_frames(
    raw_frames: list[np.ndarray],
    keypoints_by_frame: list[list[PoseKeypoint]],
    swing_phases: dict,  # SwingPhase -> frame_idx
) -> dict[str, np.ndarray]:
    """
    For each swing phase, extract the frame and render the skeleton.
    Returns {phase_value: rendered_frame}.
    """
    result = {}
    for phase, frame_idx in swing_phases.items():
        phase_key = phase.value if hasattr(phase, "value") else str(phase)
        idx = min(frame_idx, len(raw_frames) - 1)
        frame = raw_frames[idx]
        kps = keypoints_by_frame[idx] if idx < len(keypoints_by_frame) else []
        result[phase_key] = draw_skeleton(frame, kps)
    return result
