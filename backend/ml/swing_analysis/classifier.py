from app.core.config import Settings
from app.schemas.analysis import SwingPhase, PoseKeypoint

PHASES_IN_ORDER = [
    SwingPhase.ADDRESS,
    SwingPhase.TAKEAWAY,
    SwingPhase.BACKSWING,
    SwingPhase.TOP,
    SwingPhase.DOWNSWING,
    SwingPhase.IMPACT,
    SwingPhase.FOLLOW_THROUGH,
    SwingPhase.FINISH,
]


class SwingPhaseClassifier:
    """
    Segments a swing into its canonical phases by frame index.

    Approach options:
    1. Rule-based: heuristics on wrist/club-head velocity and angles
    2. ML: trained sequence model (LSTM / Transformer) on labeled swing data

    Returns a dict mapping each SwingPhase to its start frame index.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def classify(self, keypoints_by_frame: list[list[PoseKeypoint]]) -> dict[SwingPhase, int]:
        """
        Identify the frame index where each swing phase begins.

        TODO: Implement phase detection logic. Suggested approach:
        - Detect address as first N stable frames (low keypoint variance)
        - Detect top of swing as the frame with max right-wrist elevation + min velocity
        - Detect impact as the frame where club-head is lowest + max hand speed
        - Use velocity profiles to segment remaining phases
        """
        if not keypoints_by_frame:
            return {}

        n = len(keypoints_by_frame)
        step = max(1, n // len(PHASES_IN_ORDER))
        return {phase: i * step for i, phase in enumerate(PHASES_IN_ORDER)}
