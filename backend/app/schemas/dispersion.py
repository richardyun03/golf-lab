from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ClubType(str, Enum):
    DRIVER = "driver"
    FAIRWAY_WOOD = "fairway_wood"
    HYBRID = "hybrid"
    IRON = "iron"
    WEDGE = "wedge"
    PUTTER = "putter"


class ShotShape(str, Enum):
    STRAIGHT = "straight"
    DRAW = "draw"
    FADE = "fade"
    HOOK = "hook"
    SLICE = "slice"
    PUSH = "push"
    PULL = "pull"


class ShotPrediction(BaseModel):
    predicted_shape: ShotShape
    shape_confidence: float
    carry_yards_estimate: Optional[float] = None
    lateral_deviation_yards: float  # + = right, - = left
    dispersion_radius_yards: float


class DispersionResult(BaseModel):
    session_id: str
    club: ClubType
    shots: list[ShotPrediction]
    dominant_shot_shape: ShotShape
    average_lateral_deviation: float
    dispersion_radius_yards: float
    recommendations: list[str]
