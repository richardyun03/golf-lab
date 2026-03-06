from app.core.config import Settings
from app.services.video_processor import VideoData
from app.schemas.analysis import AnalysisResult, SwingMetrics
from app.ml.pose_estimation.extractor import PoseExtractor
from app.ml.swing_analysis.classifier import SwingPhaseClassifier
from app.ml.swing_analysis.fault_detector import FaultDetector


class SwingAnalyzer:
    """
    Orchestrates the full swing analysis pipeline:
      1. Pose extraction (keypoints per frame)
      2. Phase segmentation (address → finish)
      3. Metric computation (angles, tempo, etc.)
      4. Fault detection + coaching cues
      5. Scoring
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pose_extractor = PoseExtractor(settings)
        self.phase_classifier = SwingPhaseClassifier(settings)
        self.fault_detector = FaultDetector(settings)

    async def analyze(self, video: VideoData) -> AnalysisResult:
        keypoints_by_frame = await self.pose_extractor.extract(video)
        swing_phases = self.phase_classifier.classify(keypoints_by_frame)
        metrics = self._compute_metrics(keypoints_by_frame, swing_phases)
        faults = self.fault_detector.detect(keypoints_by_frame, swing_phases, metrics)
        score = self._compute_score(faults, metrics)

        return AnalysisResult(
            session_id=video.session_id,
            video_duration_seconds=video.duration_seconds,
            fps=video.fps,
            swing_phases=swing_phases,
            keypoints_by_frame=keypoints_by_frame,
            metrics=metrics,
            faults=faults,
            overall_score=score,
            summary=self._generate_summary(faults, metrics, score),
        )

    def _compute_metrics(self, keypoints_by_frame, swing_phases) -> SwingMetrics:
        # TODO: Implement angle calculations using keypoint geometry
        return SwingMetrics()

    def _compute_score(self, faults, metrics) -> float:
        # TODO: Weighted scoring based on fault severity and metric benchmarks
        if not faults:
            return 80.0
        penalty = sum(f.severity * 10 for f in faults)
        return max(0.0, min(100.0, 80.0 - penalty))

    def _generate_summary(self, faults, metrics, score) -> str:
        if not faults:
            return f"Clean swing mechanics. Overall score: {score:.0f}/100."
        fault_names = ", ".join(f.fault_type for f in faults[:3])
        return f"Score: {score:.0f}/100. Key areas to work on: {fault_names}."
