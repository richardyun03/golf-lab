from app.core.config import Settings
from app.schemas.comparison import ComparisonResult, SwingMatchResult, ProProfile
from ml.pro_comparison.embedder import SwingEmbedder
from ml.pro_comparison.matcher import SwingSimilarityMatcher


class ProMatcher:
    """
    Matches a user's swing to the closest pros in the reference database.

    Pipeline:
      1. Load the user's swing embedding from the analysis session
      2. Compare against pre-computed pro embeddings via cosine similarity + DTW
      3. Return ranked matches with per-phase similarity breakdowns
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = SwingEmbedder(settings)
        self.matcher = SwingSimilarityMatcher(settings)

    async def match(self, session_id: str) -> ComparisonResult | None:
        # TODO: Load session keypoints from storage
        # user_embedding = self.embedder.embed(keypoints)
        # matches = self.matcher.find_top_k(user_embedding, k=5)
        return None

    async def list_pros(self) -> list[ProProfile]:
        # TODO: Load from pro_swings database / JSON index
        return []
