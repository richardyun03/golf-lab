import torch
from app.core.config import Settings
from app.schemas.comparison import SwingMatchResult, ProProfile


class SwingSimilarityMatcher:
    """
    Retrieves the most similar pro swings from a pre-built embedding index.

    Matching strategy:
    1. Fast cosine similarity search over all pro embeddings
    2. Rerank top-K candidates using phase-aligned DTW distance
    3. Return ranked SwingMatchResult list

    Pro embeddings are computed offline and stored in data/pro_swings/.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pro_embeddings: dict[str, torch.Tensor] = {}
        self.pro_profiles: dict[str, ProProfile] = {}
        self._load_index()

    def _load_index(self):
        """Load pre-computed pro embeddings and profiles from disk."""
        # TODO: Load embeddings.pt and profiles.json from pro_swings_dir
        pass

    def find_top_k(self, user_embedding: torch.Tensor, k: int = 5) -> list[SwingMatchResult]:
        """
        Return the k most similar pros to the user's swing.

        Steps:
        1. Compute cosine similarity: user_embedding @ pro_matrix.T
        2. Sort descending, take top 2*k
        3. Rerank with DTW on phase-aligned keypoint sequences
        4. Build SwingMatchResult for each
        """
        if not self.pro_embeddings:
            return []

        # TODO: implement similarity search
        return []
