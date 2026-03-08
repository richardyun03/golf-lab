"""Render ideal skeleton reference diagrams for each swing phase."""

import cv2
import numpy as np
from ml.swing_analysis.ideal_poses import IDEAL_POSES, PHASE_CHECKPOINTS

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

# Colors
BG_COLOR = (20, 20, 25)
BONE_COLOR = (180, 220, 100)      # lime green
JOINT_COLOR = (200, 240, 120)
GUIDE_COLOR = (80, 80, 90)        # subtle grid lines
TEXT_COLOR = (200, 200, 200)
CHECKPOINT_COLOR = (140, 180, 100)
TITLE_COLOR = (220, 220, 220)


def render_ideal_frame(phase: str, width: int = 400, height: int = 540) -> np.ndarray | None:
    """Render an ideal skeleton diagram for a given phase."""
    pose = IDEAL_POSES.get(phase)
    if pose is None:
        return None

    canvas = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)

    # Draw body area
    body_area_h = height - 120  # leave room for text at bottom
    body_area_y = 50  # top margin for title

    # Draw subtle ground line
    ground_y = body_area_y + int(0.88 * body_area_h)
    cv2.line(canvas, (40, ground_y), (width - 40, ground_y), GUIDE_COLOR, 1, cv2.LINE_AA)

    # Draw subtle center line
    center_x = width // 2
    cv2.line(canvas, (center_x, body_area_y + 20), (center_x, ground_y),
             (35, 35, 40), 1, cv2.LINE_AA)

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = int(x * (width - 80) + 40)
        py = int(y * body_area_h + body_area_y)
        return px, py

    # Draw bones
    for a_name, b_name in CONNECTIONS:
        if a_name in pose and b_name in pose:
            pt_a = to_px(*pose[a_name])
            pt_b = to_px(*pose[b_name])
            cv2.line(canvas, pt_a, pt_b, BONE_COLOR, 3, cv2.LINE_AA)

    # Draw joints
    for name, (x, y) in pose.items():
        pt = to_px(x, y)
        cv2.circle(canvas, pt, 6, JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 7, BG_COLOR, 1, cv2.LINE_AA)

    # Phase title
    phase_label = phase.replace("_", " ").title()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"Ideal: {phase_label}", (15, 30),
                font, 0.65, TITLE_COLOR, 2, cv2.LINE_AA)

    # Checkpoints
    checkpoints = PHASE_CHECKPOINTS.get(phase, [])
    y_start = height - 100
    for i, cp in enumerate(checkpoints):
        y_pos = y_start + i * 22
        # Bullet
        cv2.circle(canvas, (25, y_pos - 4), 3, CHECKPOINT_COLOR, -1, cv2.LINE_AA)
        cv2.putText(canvas, cp, (35, y_pos),
                    font, 0.42, TEXT_COLOR, 1, cv2.LINE_AA)

    return canvas
