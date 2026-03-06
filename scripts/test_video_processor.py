"""
Quick test: run VideoProcessor + PoseExtractor + Phase Segmentation on a video file.

Usage:
    cd backend
    python ../scripts/test_video_processor.py /path/to/swing.mp4
    python ../scripts/test_video_processor.py /path/to/swing.mp4 --pose
    python ../scripts/test_video_processor.py /path/to/swing.mp4 --pose --phases
"""
import sys
import asyncio
from pathlib import Path

# Add backend to path so imports work from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.services.video_processor import VideoProcessor


class FakeUpload:
    """Mimics FastAPI UploadFile for local testing."""
    def __init__(self, path: Path):
        self.filename = path.name
        self._data = path.read_bytes()

    async def read(self):
        return self._data


async def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print("Usage: python test_video_processor.py <video_path> [--pose] [--phases]")
        sys.exit(1)

    video_path = Path(args[0])
    run_pose = "--pose" in args
    run_phases = "--phases" in args

    if not video_path.exists():
        print(f"File not found: {video_path}")
        sys.exit(1)

    settings = get_settings()
    processor = VideoProcessor(settings)

    upload = FakeUpload(video_path)
    video = await processor.load(upload)

    print("=== Video Processing ===")
    print(f"Session ID:  {video.session_id}")
    print(f"Resolution:  {video.width}x{video.height}")
    print(f"FPS:         {video.fps}")
    print(f"Duration:    {video.duration_seconds:.2f}s")
    print(f"Frames:      {video.frame_count}")
    print(f"Frame shape: {video.raw_frames[0].shape if video.raw_frames else 'N/A'}")

    if not run_pose:
        print("\nAdd --pose flag to also run pose extraction.")
        return

    from ml.pose_estimation.extractor import PoseExtractor

    print("\n=== Pose Extraction ===")
    extractor = PoseExtractor(settings)
    keypoints_by_frame = await extractor.extract(video)

    detected = sum(1 for kps in keypoints_by_frame if kps)
    print(f"Frames with pose detected: {detected}/{len(keypoints_by_frame)}")

    if detected > 0:
        first_detected = next(kps for kps in keypoints_by_frame if kps)
        print(f"Keypoints per frame: {len(first_detected)}")
        print("\nSample keypoints (first detected frame):")
        for kp in first_detected:
            print(f"  {kp.name:20s}  x={kp.x:.3f}  y={kp.y:.3f}  z={kp.z:.4f}  conf={kp.confidence:.2f}")

    extractor.close()

    if not run_phases:
        print("\nAdd --phases flag to also run swing phase segmentation.")
        return

    from ml.swing_analysis.classifier import SwingPhaseClassifier

    print("\n=== Swing Phase Segmentation ===")
    classifier = SwingPhaseClassifier(settings)
    phases = classifier.classify(keypoints_by_frame)

    if not phases:
        print("Could not detect swing phases.")
        return

    fps = video.fps
    for phase, frame in phases.items():
        timestamp = frame / fps if fps > 0 else 0
        print(f"  {phase.value:20s}  frame={frame:4d}  time={timestamp:.2f}s")

    # Print phase durations
    phase_list = sorted(phases.items(), key=lambda x: x[1])
    print("\nPhase durations:")
    for i in range(len(phase_list) - 1):
        phase_name = phase_list[i][0].value
        duration_frames = phase_list[i + 1][1] - phase_list[i][1]
        duration_sec = duration_frames / fps if fps > 0 else 0
        print(f"  {phase_name:20s}  {duration_frames:3d} frames  ({duration_sec:.2f}s)")


if __name__ == "__main__":
    asyncio.run(main())
