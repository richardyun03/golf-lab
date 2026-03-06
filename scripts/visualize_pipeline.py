"""
Visualize the full pipeline: skeleton overlay on video + signal plots.

Outputs:
  1. Annotated video with skeleton + phase labels → output_skeleton.mp4
  2. Wrist Y + body velocity plot with phase boundaries → output_signals.png

Usage:
    cd backend
    python ../scripts/visualize_pipeline.py ../data/sample_videos/IMG_6181.mov
"""
import sys
import asyncio
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.services.video_processor import VideoProcessor
from ml.pose_estimation.extractor import PoseExtractor
from ml.swing_analysis.classifier import SwingPhaseClassifier, _smooth, _velocity

# Skeleton connections between our 13 keypoints (by name)
SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]

PHASE_COLORS = {
    "address":        (200, 200, 200),  # gray
    "takeaway":       (255, 200, 0),    # cyan-ish
    "backswing":      (0, 255, 255),    # yellow
    "top":            (0, 165, 255),    # orange
    "downswing":      (0, 0, 255),      # red
    "impact":         (0, 0, 200),      # dark red
    "follow_through": (0, 255, 0),      # green
    "finish":         (255, 0, 0),      # blue
}


class FakeUpload:
    def __init__(self, path: Path):
        self.filename = path.name
        self._data = path.read_bytes()
    async def read(self):
        return self._data


def get_phase_at_frame(phases, frame_idx):
    """Return the phase name for a given frame index."""
    current_phase = None
    for phase, start_frame in sorted(phases.items(), key=lambda x: x[1]):
        if frame_idx >= start_frame:
            current_phase = phase.value
    return current_phase or "unknown"


def draw_skeleton(frame, keypoints, phase_name, frame_idx, fps):
    """Draw keypoints, skeleton, and phase label on a frame."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    if not keypoints:
        return annotated

    kp_dict = {kp.name: kp for kp in keypoints}
    color = PHASE_COLORS.get(phase_name, (255, 255, 255))

    # Draw connections
    for name_a, name_b in SKELETON:
        a, b = kp_dict.get(name_a), kp_dict.get(name_b)
        if a and b and a.confidence > 0.3 and b.confidence > 0.3:
            pt_a = (int(a.x * w), int(a.y * h))
            pt_b = (int(b.x * w), int(b.y * h))
            cv2.line(annotated, pt_a, pt_b, color, 3)

    # Draw keypoints
    for kp in keypoints:
        if kp.confidence > 0.3:
            pt = (int(kp.x * w), int(kp.y * h))
            cv2.circle(annotated, pt, 6, color, -1)
            cv2.circle(annotated, pt, 6, (0, 0, 0), 1)

    # Phase label
    timestamp = frame_idx / fps if fps > 0 else 0
    label = f"{phase_name.upper()}  frame={frame_idx}  t={timestamp:.2f}s"
    cv2.putText(annotated, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(annotated, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    return annotated


def plot_signals(keypoints_by_frame, phases, fps, output_path):
    """Plot wrist Y and body velocity with phase boundary lines."""
    n = len(keypoints_by_frame)
    wrist_y = np.full(n, np.nan)
    body_vel = np.zeros(n)
    prev_positions = None

    for i, kps in enumerate(keypoints_by_frame):
        if not kps:
            prev_positions = None
            continue
        d = {kp.name: kp for kp in kps}
        lw, rw = d.get("left_wrist"), d.get("right_wrist")
        if lw and rw:
            wrist_y[i] = (lw.y + rw.y) / 2.0
        elif lw:
            wrist_y[i] = lw.y
        elif rw:
            wrist_y[i] = rw.y

        positions = np.array([[kp.x, kp.y] for kp in kps])
        if prev_positions is not None and positions.shape == prev_positions.shape:
            diff = positions - prev_positions
            body_vel[i] = np.sqrt((diff ** 2).sum(axis=1)).sum()
        prev_positions = positions

    # Interpolate and smooth
    valid = ~np.isnan(wrist_y)
    if valid.sum() > 0:
        wrist_y = np.interp(np.arange(n), np.where(valid)[0], wrist_y[valid])
    wrist_y_smooth = _smooth(wrist_y, 7)
    body_vel_smooth = _smooth(body_vel, 7)

    time = np.arange(n) / fps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Wrist Y (inverted so "up" is up on the plot)
    ax1.plot(time, wrist_y, alpha=0.3, color="gray", label="Raw wrist Y")
    ax1.plot(time, wrist_y_smooth, color="blue", linewidth=1.5, label="Smoothed wrist Y")
    ax1.set_ylabel("Wrist Y (lower = hands higher)")
    ax1.invert_yaxis()
    ax1.legend(loc="upper right")
    ax1.set_title("Wrist Y Position Over Time")

    # Body velocity
    ax2.plot(time, body_vel, alpha=0.3, color="gray", label="Raw body velocity")
    ax2.plot(time, body_vel_smooth, color="red", linewidth=1.5, label="Smoothed body velocity")
    ax2.set_ylabel("Body Velocity (sum of keypoint displacements)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")
    ax2.set_title("Total Body Velocity Over Time")

    # Draw phase boundaries
    phase_colors_plt = {
        "address": "gray", "takeaway": "cyan", "backswing": "gold",
        "top": "orange", "downswing": "red", "impact": "darkred",
        "follow_through": "green", "finish": "blue",
    }
    for phase, frame in phases.items():
        t = frame / fps
        c = phase_colors_plt.get(phase.value, "black")
        for ax in (ax1, ax2):
            ax.axvline(t, color=c, linestyle="--", alpha=0.8, linewidth=1.5)
            ax.text(t, ax.get_ylim()[1 if ax == ax2 else 0], f" {phase.value}",
                    rotation=90, fontsize=7, color=c, va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Signal plot saved to: {output_path}")


async def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print("Usage: python visualize_pipeline.py <video_path> [--output-dir DIR]")
        sys.exit(1)

    video_path = Path(args[0])
    output_dir = Path(args[args.index("--output-dir") + 1]) if "--output-dir" in args else video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"File not found: {video_path}")
        sys.exit(1)

    settings = get_settings()

    # Step 1: Load video
    print("Loading video...")
    processor = VideoProcessor(settings)
    video = await processor.load(FakeUpload(video_path))
    print(f"  {video.frame_count} frames, {video.width}x{video.height}, {video.fps}fps, {video.duration_seconds:.2f}s")

    # Step 2: Pose extraction
    print("Running pose extraction...")
    extractor = PoseExtractor(settings)
    keypoints_by_frame = await extractor.extract(video)
    detected = sum(1 for kps in keypoints_by_frame if kps)
    print(f"  Pose detected in {detected}/{len(keypoints_by_frame)} frames")
    extractor.close()

    # Step 3: Phase segmentation
    print("Running phase segmentation...")
    classifier = SwingPhaseClassifier(settings)
    phases = classifier.classify(keypoints_by_frame)
    for phase, frame in sorted(phases.items(), key=lambda x: x[1]):
        t = frame / video.fps if video.fps > 0 else 0
        print(f"  {phase.value:20s}  frame={frame:4d}  time={t:.2f}s")

    # Step 4: Generate signal plot
    print("\nGenerating signal plot...")
    plot_path = output_dir / "output_signals.png"
    plot_signals(keypoints_by_frame, phases, video.fps, str(plot_path))

    # Step 5: Generate annotated video
    print("Generating annotated video...")
    video_out_path = output_dir / "output_skeleton.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_out_path), fourcc, video.fps, (video.width, video.height))

    for i, frame in enumerate(video.raw_frames):
        kps = keypoints_by_frame[i] if i < len(keypoints_by_frame) else []
        phase_name = get_phase_at_frame(phases, i)
        annotated = draw_skeleton(frame, kps, phase_name, i, video.fps)
        out.write(annotated)

    out.release()
    print(f"Annotated video saved to: {video_out_path}")
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
