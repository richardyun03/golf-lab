from app.core.config import Settings
from app.schemas.comparison import ComparisonResult, SwingMatchResult, ProProfile
from app.services.storage import AnalysisStorage
from ml.pro_comparison.matcher import compare_to_pros, classify_swing, get_tour_comparison
from ml.pro_comparison.pro_database import PRO_PROFILES


class ProMatcher:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = AnalysisStorage(settings)

    async def match(self, session_id: str) -> ComparisonResult | None:
        session = self.storage.get(session_id)
        if session is None:
            return None

        matches = compare_to_pros(session.metrics, top_k=5)
        if not matches:
            return None

        archetype = classify_swing(session.metrics)

        return ComparisonResult(
            session_id=session_id,
            top_matches=matches,
            primary_match=matches[0],
            swing_archetype=archetype,
        )

    async def get_tour_comparison(self, session_id: str, tour: str = "PGA") -> dict | None:
        session = self.storage.get(session_id)
        if session is None:
            return None
        return get_tour_comparison(session.metrics, tour)

    async def list_pros(self) -> list[ProProfile]:
        return [
            ProProfile(
                pro_id=p["pro_id"],
                name=p["name"],
                tour=p["tour"],
                swing_style=p["swing_style"],
                known_for=p["known_for"],
            )
            for p in PRO_PROFILES
        ]
