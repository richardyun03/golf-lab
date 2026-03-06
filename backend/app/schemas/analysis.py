from pydantic import BaseModel
from typing import Optional
from enum import Enum


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
    hip_rotation_degrees: Optional[float] = None
    shoulder_rotation_degrees: Optional[float] = None
    spine_angle_degrees: Optional[float] = None
    knee_flex_degrees: Optional[float] = None
    tempo_ratio: Optional[float] = None  # backswing:downswing time ratio
    attack_angle_degrees: Optional[float] = None


class AnalysisResult(BaseModel):
    session_id: str
    video_duration_seconds: float
    fps: float
    swing_phases: dict[SwingPhase, int]  # phase -> frame index
    keypoints_by_frame: list[list[PoseKeypoint]]
    metrics: SwingMetrics
    faults: list[SwingFault]
    overall_score: float  # 0-100
    summary: str
