from app.core.config import Settings
from app.schemas.analysis import SwingFault, SwingPhase, SwingMetrics, PoseKeypoint

# Fault definitions: (fault_type, phase, description, correction)
KNOWN_FAULTS = [
    {
        "fault_type": "early_extension",
        "phase": SwingPhase.DOWNSWING,
        "description": "Hips thrust toward the ball through impact, losing spine angle.",
        "correction": "Focus on maintaining your hip-to-ball distance through impact. Drill: impact bag work.",
    },
    {
        "fault_type": "over_the_top",
        "phase": SwingPhase.DOWNSWING,
        "description": "Club swings outside-to-in, causing pulls and slices.",
        "correction": "Feel like the right elbow drops into the slot first. Drill: headcover under right arm.",
    },
    {
        "fault_type": "chicken_wing",
        "phase": SwingPhase.FOLLOW_THROUGH,
        "description": "Lead arm breaks down post-impact instead of extending through.",
        "correction": "Practice keeping the lead arm connected to your chest through impact.",
    },
    {
        "fault_type": "sway",
        "phase": SwingPhase.BACKSWING,
        "description": "Lateral hip slide away from target instead of rotation.",
        "correction": "Feel weight stay on the inside of your trail foot. Drill: swing with foot against a wall.",
    },
    {
        "fault_type": "casting",
        "phase": SwingPhase.DOWNSWING,
        "description": "Early release of wrist angle (loss of lag) from the top.",
        "correction": "Hold the angle longer — feel the butt of the club pointing at the ball longer.",
    },
]


class FaultDetector:
    """
    Detects common swing faults from pose keypoints and computed metrics.

    Each fault check is an independent rule or small classifier. Over time
    these can be replaced with a unified learned model.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def detect(
        self,
        keypoints_by_frame: list[list[PoseKeypoint]],
        swing_phases: dict[SwingPhase, int],
        metrics: SwingMetrics,
    ) -> list[SwingFault]:
        """
        Run all fault checks and return detected faults.

        TODO: Implement each check using keypoint geometry:
        - early_extension: track hip-to-ball distance across downswing frames
        - over_the_top: measure club path angle at top → impact
        - chicken_wing: measure lead elbow angle at follow-through
        - sway: measure lateral displacement of lead hip in backswing
        - casting: measure wrist angle (lag) from top through P7 (impact)
        """
        # Placeholder — no real detection yet
        return []
