from fastapi import APIRouter, HTTPException, Depends
from app.schemas.dispersion import DispersionResult, ClubType
from app.services.shot_dispersion import ShotDispersionService
from app.core.dependencies import get_settings
from app.core.config import Settings

router = APIRouter()


@router.post("/{session_id}", response_model=DispersionResult)
async def predict_dispersion(
    session_id: str,
    club: ClubType = ClubType.DRIVER,
    settings: Settings = Depends(get_settings),
):
    """
    [Coming Soon] Predict shot shape and dispersion patterns from an analyzed swing.

    Uses swing mechanics data from a completed analysis session to estimate:
    - Dominant shot shape (draw, fade, hook, slice, etc.)
    - Lateral deviation and dispersion radius
    - Personalized fix recommendations
    """
    raise HTTPException(
        status_code=501,
        detail="Shot dispersion is coming soon. Complete swing analysis first.",
    )
