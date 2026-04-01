from app.core.config import Settings
from app.services.video_processor import VideoData
from app.schemas.analysis import AnalysisResult, SwingMetrics, SwingPhase
from ml.pose_estimation.extractor import PoseExtractor
from ml.swing_analysis.classifier import SwingPhaseClassifier
from ml.swing_analysis.fault_detector import FaultDetector
from ml.swing_analysis.metrics import compute_metrics


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
        metrics = compute_metrics(keypoints_by_frame, swing_phases, video.fps)
        faults = self.fault_detector.detect(keypoints_by_frame, swing_phases, metrics, club_type=video.club_type)
        score = self._compute_score(faults, metrics)
        phase_scores = self._compute_phase_scores(faults)

        return AnalysisResult(
            session_id=video.session_id,
            video_duration_seconds=video.duration_seconds,
            fps=video.fps,
            swing_phases=swing_phases,
            keypoints_by_frame=keypoints_by_frame,
            metrics=metrics,
            faults=faults,
            overall_score=score,
            phase_scores=phase_scores,
            summary=self._generate_summary(faults, metrics, score),
        )

    def _compute_score(self, faults, metrics) -> float:
        if not faults:
            return 85.0
        penalty = sum(f.severity * 12 for f in faults)
        return max(0.0, min(100.0, 85.0 - penalty))

    @staticmethod
    def _compute_phase_scores(faults) -> dict[str, float]:
        faults_by_phase: dict[str, list] = {}
        for f in faults:
            key = f.phase.value if hasattr(f.phase, "value") else str(f.phase)
            faults_by_phase.setdefault(key, []).append(f)

        phase_scores = {}
        for phase in SwingPhase:
            phase_faults = faults_by_phase.get(phase.value, [])
            penalty = sum(f.severity * 12 for f in phase_faults)
            phase_scores[phase.value] = max(0.0, min(100.0, 85.0 - penalty))
        return phase_scores

    def _generate_summary(self, faults, metrics, score) -> str:
        parts = [f"Score: {score:.0f}/100."]

        if metrics.tempo_ratio is not None:
            parts.append(f"Tempo ratio: {metrics.tempo_ratio}:1.")
        if metrics.x_factor_degrees is not None:
            parts.append(f"X-factor: {metrics.x_factor_degrees} deg.")

        if faults:
            fault_names = ", ".join(f.fault_type.replace("_", " ") for f in faults[:3])
            parts.append(f"Key areas to work on: {fault_names}.")
        else:
            parts.append("Clean swing mechanics.")

        return " ".join(parts)
