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


def draw_skeleton(frame, keypoints, phase_name, frame_idx, fps, phase_faults=None):
    """Draw keypoints, skeleton, phase label, and any active faults on a frame."""
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

    # Draw fault labels if any are active for this phase
    if phase_faults:
        y_offset = 110
        for fault in phase_faults:
            severity_pct = f"{fault.severity:.0%}"
            fault_label = f"FAULT: {fault.fault_type} ({severity_pct})"
            cv2.putText(annotated, fault_label, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(annotated, fault_label, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            y_offset += 40

    return annotated


def build_fault_summary(video_frames, keypoints_by_frame, pf, faults, phases, fps, output_path):
    """
    Generate a summary image: one panel per phase showing the representative
    frame with skeleton overlay, plus fault annotations on relevant phases.
    """
    from app.schemas.analysis import SwingPhase

    phase_order = [
        SwingPhase.ADDRESS, SwingPhase.TAKEAWAY, SwingPhase.BACKSWING,
        SwingPhase.TOP, SwingPhase.DOWNSWING, SwingPhase.IMPACT,
        SwingPhase.FOLLOW_THROUGH, SwingPhase.FINISH,
    ]

    # Map faults to their phases
    fault_by_phase = {}
    for f in faults:
        fault_by_phase.setdefault(f.phase.value, []).append(f)

    panels = []
    for phase in phase_order:
        if phase not in phases:
            continue
        rep = pf.rep_frame(phase)
        frame = video_frames[rep].copy()
        kps = keypoints_by_frame[rep] if rep < len(keypoints_by_frame) else []
        phase_faults = fault_by_phase.get(phase.value, [])
        annotated = draw_skeleton(frame, kps, phase.value, rep, fps, phase_faults)

        # Resize for the summary grid (fit to consistent height)
        target_h = 640
        scale = target_h / annotated.shape[0]
        target_w = int(annotated.shape[1] * scale)
        panel = cv2.resize(annotated, (target_w, target_h))
        panels.append(panel)

    if not panels:
        return

    # Arrange in 2 rows of 4
    row1 = np.hstack(panels[:4])
    row2 = np.hstack(panels[4:])
    # Pad row2 if fewer than 4 panels
    if row2.shape[1] < row1.shape[1]:
        pad = np.zeros((row2.shape[0], row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
        row2 = np.hstack([row2, pad])
    summary = np.vstack([row1, row2])

    cv2.imwrite(str(output_path), summary)
    print(f"Fault summary saved to: {output_path}")


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

    # Step 4: Metrics + representative frames + fault detection
    from ml.swing_analysis.fault_detector import FaultDetector, PhaseFrames
    from ml.swing_analysis.metrics import compute_metrics
    from app.schemas.analysis import SwingMetrics

    print("\nComputing swing metrics...")
    metrics = compute_metrics(keypoints_by_frame, phases, video.fps)
    for field, value in metrics.model_dump().items():
        if value is not None:
            label = field.replace("_", " ").title()
            print(f"  {label:40s}  {value}")

    print("\nRepresentative frames (highest confidence per phase):")
    pf = PhaseFrames(keypoints_by_frame, phases)
    for phase in sorted(phases, key=lambda p: phases[p]):
        rep = pf.rep_frame(phase)
        orig = phases[phase]
        t = rep / video.fps if video.fps > 0 else 0
        label = f"  (was {orig})" if rep != orig else ""
        print(f"  {phase.value:20s}  rep_frame={rep:4d}  time={t:.2f}s{label}")

    print("\nRunning fault detection...")
    fault_detector = FaultDetector(settings)
    faults = fault_detector.detect(keypoints_by_frame, phases, SwingMetrics())

    if faults:
        for f in faults:
            print(f"  [{f.severity:.0%} severity] {f.fault_type}")
            print(f"    {f.description}")
            print(f"    Fix: {f.correction}")
            print()
    else:
        print("  No faults detected.")

    # Step 5: Generate signal plot
    print("Generating signal plot...")
    plot_path = output_dir / "output_signals.png"
    plot_signals(keypoints_by_frame, phases, video.fps, str(plot_path))

    # Step 6: Build fault-by-phase lookup for video overlay
    fault_by_phase = {}
    for f in faults:
        fault_by_phase.setdefault(f.phase.value, []).append(f)

    # Step 7: Generate annotated video with fault overlays
    print("Generating annotated video...")
    video_out_path = output_dir / "output_skeleton.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_out_path), fourcc, video.fps, (video.width, video.height))

    for i, frame in enumerate(video.raw_frames):
        kps = keypoints_by_frame[i] if i < len(keypoints_by_frame) else []
        phase_name = get_phase_at_frame(phases, i)
        phase_faults = fault_by_phase.get(phase_name, [])
        annotated = draw_skeleton(frame, kps, phase_name, i, video.fps, phase_faults)
        out.write(annotated)

    out.release()
    print(f"Annotated video saved to: {video_out_path}")

    # Step 8: Generate fault summary image (representative frames grid)
    print("Generating fault summary...")
    summary_path = output_dir / "output_fault_summary.png"
    build_fault_summary(video.raw_frames, keypoints_by_frame, pf, faults, phases, video.fps, str(summary_path))

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
