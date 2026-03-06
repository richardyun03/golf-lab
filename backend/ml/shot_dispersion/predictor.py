from app.core.config import Settings
from app.schemas.dispersion import ShotPrediction, ShotShape, ClubType


class DispersionPredictor:
    """
    [Future] Predicts shot shape and dispersion from swing mechanics.

    Training data needed:
    - Paired (pose keypoints, launch monitor data) recordings
    - Launch monitor fields: ball speed, launch angle, spin rate, spin axis,
      carry distance, offline distance (lateral deviation)

    Model idea:
    - Input:  swing embedding + club type
    - Output: shot shape classification + regression on lateral deviation
    - Architecture: small MLP or fine-tuned from SwingEncoder
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def predict(self, swing_embedding, club: ClubType) -> ShotPrediction:
        """Predict shot shape and dispersion from a swing embedding."""
        raise NotImplementedError("Shot dispersion model not yet trained.")
