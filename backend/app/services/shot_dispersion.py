from app.core.config import Settings
from app.schemas.dispersion import DispersionResult, ClubType
from app.ml.shot_dispersion.predictor import DispersionPredictor


class ShotDispersionService:
    """
    Predicts shot shape and dispersion from swing mechanics.

    This is a future feature. The predictor will be trained on
    paired (swing data, Trackman/launch monitor) datasets.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.predictor = DispersionPredictor(settings)

    async def predict(self, session_id: str, club: ClubType) -> DispersionResult | None:
        # TODO: Load session, run predictor, return dispersion result
        return None
