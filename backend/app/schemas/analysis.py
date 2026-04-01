from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ClubType(str, Enum):
    DRIVER = "driver"
    WOOD = "wood"
    HYBRID = "hybrid"
    LONG_IRON = "long_iron"
    MID_IRON = "mid_iron"
    SHORT_IRON = "short_iron"
    WEDGE = "wedge"
    PUTTER = "putter"


class SwingPhase(str, Enum):
    ADDRESS = "address"
    TAKEAWAY = "takeaway"
    BACKSWING = "backswing"
    TOP = "top"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    FINISH = "finish"


class PoseKeypoint(BaseModel):
    name: str
    x: float
    y: float
    z: Optional[float] = None
    confidence: float


class SwingFault(BaseModel):
    fault_type: str
    phase: SwingPhase
    description: str
    severity: float  # 0.0 - 1.0
    correction: str


class SwingMetrics(BaseModel):
    # Rotation (relative to address baseline, measured at top of backswing)
    hip_rotation_degrees: Optional[float] = None
    shoulder_rotation_degrees: Optional[float] = None
    x_factor_degrees: Optional[float] = None  # shoulder turn - hip turn at top

    # Spine
    spine_tilt_address_degrees: Optional[float] = None  # spine angle from vertical at address
    spine_tilt_change_degrees: Optional[float] = None    # change from address to impact (+ = stood up)

    # Lower body
    lead_knee_flex_address_degrees: Optional[float] = None
    lead_knee_flex_impact_degrees: Optional[float] = None

    # Tempo
    tempo_ratio: Optional[float] = None         # backswing:downswing time ratio (pro avg ~3:1)
    backswing_duration_seconds: Optional[float] = None
    downswing_duration_seconds: Optional[float] = None
    total_swing_duration_seconds: Optional[float] = None  # address to finish


class AnalysisResult(BaseModel):
    session_id: str
    club_type: Optional[ClubType] = None
    video_duration_seconds: float
    fps: float
    swing_phases: dict[SwingPhase, int]  # phase -> frame index
    keypoints_by_frame: list[list[PoseKeypoint]]
    metrics: SwingMetrics
    faults: list[SwingFault]
    overall_score: float  # 0-100
    phase_scores: dict[str, float] = {}  # phase -> 0-100 score
    summary: str


class AnalysisResponse(BaseModel):
    """Lighter response model for the API — excludes per-frame keypoints."""
    session_id: str
    club_type: Optional[ClubType] = None
    video_duration_seconds: float
    fps: float
    swing_phases: dict[SwingPhase, int]
    metrics: SwingMetrics
    ideal_ranges: dict[str, tuple[float, float]] = {}
    faults: list[SwingFault]
    overall_score: float
    phase_scores: dict[str, float] = {}
    summary: str
    frame_count: int


class SessionTrendPoint(BaseModel):
    session_id: str
    created_at: str
    club_type: Optional[ClubType] = None
    overall_score: float
    phase_scores: dict[str, float]
    metrics: SwingMetrics
    fault_count: int
    fault_types: list[str]
