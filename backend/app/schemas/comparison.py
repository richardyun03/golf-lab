from pydantic import BaseModel
from typing import Optional


class ProProfile(BaseModel):
    pro_id: str
    name: str
    tour: str  # "PGA", "LPGA", "DP World", etc.
    swing_style: str  # e.g., "upright", "flat", "rotational"
    known_for: list[str]
    thumbnail_url: Optional[str] = None


class SwingMatchResult(BaseModel):
    pro: ProProfile
    similarity_score: float  # 0.0 - 1.0
    matching_phases: list[str]
    key_similarities: list[str]
    key_differences: list[str]


class ComparisonResult(BaseModel):
    session_id: str
    top_matches: list[SwingMatchResult]
    primary_match: SwingMatchResult
    swing_archetype: str  # e.g., "Stack & Tilt", "Classic", "Modern Rotational"
