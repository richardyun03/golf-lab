"""
Batch-run the analysis pipeline on all videos in a directory.

For each video, outputs:
  - Fault summary image (phase grid with skeleton + faults)
  - Signal plot (wrist Y + body velocity with phase boundaries)
  - JSON results (phases, metrics, faults, score)

Usage:
    cd backend
    python ../scripts/batch_analyze.py                          # all videos in data/sample_videos/
    python ../scripts/batch_analyze.py ../data/my_new_videos/   # custom input dir
    python ../scripts/batch_analyze.py video1.mov video2.mp4    # specific files
"""
import sys
import asyncio
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.services.video_processor import VideoProcessor
from ml.pose_estimation.extractor import PoseExtractor
from ml.swing_analysis.classifier import SwingPhaseClassifier
from ml.swing_analysis.fault_detector import FaultDetector, PhaseFrames
from ml.swing_analysis.metrics import compute_metrics
from app.schemas.analysis import SwingMetrics

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v", ".mkv"}


class FakeUpload:
    def __init__(self, path: Path):
        self.filename = path.name
        self._data = path.read_bytes()

    async def read(self):
        return self._data


def collect_videos(args: list[str]) -> list[Path]:
    """Resolve CLI args to a list of video file paths."""
    if not args:
        default_dir = Path(__file__).resolve().parents[1] / "data" / "sample_videos"
        args = [str(default_dir)]

    videos = []
    for arg in args:
        p = Path(arg).resolve()
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in VIDEO_EXTENSIONS and not f.stem.startswith("output"):
                    videos.append(f)
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)
        else:
            print(f"  Skipping: {arg}")
    return videos


async def analyze_one(video_path: Path, output_dir: Path, settings):
    """Run the full pipeline on a single video and save outputs."""
    stem = video_path.stem
    vid_output = output_dir / stem
    vid_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {video_path.name}")
    print(f"{'='*60}")

    # Load video
    processor = VideoProcessor(settings)
    video = await processor.load(FakeUpload(video_path))
    print(f"  {video.frame_count} frames, {video.width}x{video.height}, {video.fps}fps, {video.duration_seconds:.2f}s")

    # Pose extraction
    print("  Extracting poses...")
    extractor = PoseExtractor(settings)
    keypoints_by_frame = await extractor.extract(video)
    detected = sum(1 for kps in keypoints_by_frame if kps)
    print(f"  Pose detected in {detected}/{len(keypoints_by_frame)} frames")

    # Phase segmentation
    print("  Classifying phases...")
    classifier = SwingPhaseClassifier(settings)
    phases = classifier.classify(keypoints_by_frame)

    if not phases:
        print("  WARNING: No swing phases detected. Skipping.")
        summary = {"video": video_path.name, "error": "No swing phases detected"}
        (vid_output / "result.json").write_text(json.dumps(summary, indent=2))
        return summary

    for phase, frame in sorted(phases.items(), key=lambda x: x[1]):
        t = frame / video.fps if video.fps > 0 else 0
        print(f"    {phase.value:20s}  frame={frame:4d}  time={t:.2f}s")

    # Metrics
    print("  Computing metrics...")
    metrics = compute_metrics(keypoints_by_frame, phases, video.fps)

    # Fault detection
    print("  Detecting faults...")
    pf = PhaseFrames(keypoints_by_frame, phases)
    fault_detector = FaultDetector(settings)
    faults = fault_detector.detect(keypoints_by_frame, phases, SwingMetrics())

    if faults:
        for f in faults:
            print(f"    [{f.severity:.0%}] {f.fault_type}: {f.description}")
    else:
        print("    No faults detected.")

    # Score
    score = 85.0 - sum(f.severity * 12 for f in faults)
    score = max(0.0, min(100.0, score))
    print(f"  Score: {score:.0f}/100")

    # Save JSON results
    result = {
        "video": video_path.name,
        "duration_seconds": video.duration_seconds,
        "fps": video.fps,
        "frame_count": video.frame_count,
        "resolution": f"{video.width}x{video.height}",
        "phases": {p.value: f for p, f in sorted(phases.items(), key=lambda x: x[1])},
        "metrics": {k: v for k, v in metrics.model_dump().items() if v is not None},
        "faults": [
            {"type": f.fault_type, "phase": f.phase.value, "severity": round(f.severity, 2), "description": f.description}
            for f in faults
        ],
        "score": round(score, 1),
    }
    (vid_output / "result.json").write_text(json.dumps(result, indent=2))

    # Generate visualizations (import here to avoid opencv/matplotlib overhead if not needed)
    from visualize_pipeline import plot_signals, build_fault_summary

    print("  Generating signal plot...")
    plot_signals(keypoints_by_frame, phases, video.fps, str(vid_output / "signals.png"))

    print("  Generating fault summary...")
    build_fault_summary(
        video.raw_frames, keypoints_by_frame, pf, faults,
        phases, video.fps, str(vid_output / "fault_summary.png"),
    )

    return result


async def main():
    args = sys.argv[1:]
    videos = collect_videos(args)

    if not videos:
        print("No videos found.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s):")
    for v in videos:
        print(f"  {v.name}")

    settings = get_settings()
    output_dir = Path(__file__).resolve().parents[1] / "data" / "batch_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for video_path in videos:
        try:
            result = await analyze_one(video_path, output_dir, settings)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR processing {video_path.name}: {e}")
            results.append({"video": video_path.name, "error": str(e)})

    # Save combined summary
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {len(results)} videos processed")
    print(f"  Results: {output_dir}")
    print(f"{'='*60}")

    # Print comparison table
    print(f"\n{'Video':<25} {'Score':>6} {'Tempo':>6} {'Shoulder':>9} {'X-Factor':>9} {'Faults'}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['video']:<25} {'ERROR':>6}")
            continue
        m = r.get("metrics", {})
        tempo = m.get("tempo_ratio", "—")
        shoulder = m.get("shoulder_rotation_degrees", "—")
        xfactor = m.get("x_factor_degrees", "—")
        fault_names = ", ".join(f["type"] for f in r.get("faults", [])) or "none"
        print(f"{r['video']:<25} {r['score']:>6.1f} {str(tempo):>6} {str(shoulder):>9} {str(xfactor):>9} {fault_names}")


if __name__ == "__main__":
    asyncio.run(main())
