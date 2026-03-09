from fastapi import APIRouter, HTTPException, Depends
from app.schemas.comparison import ComparisonResult, ProProfile
from app.services.pro_matcher import ProMatcher
from app.core.dependencies import get_pro_matcher

router = APIRouter()


@router.get("/{session_id}", response_model=ComparisonResult)
async def compare_to_pros(
    session_id: str,
    pro_matcher: ProMatcher = Depends(get_pro_matcher),
):
    """Match the user's swing metrics against pro golfer profiles."""
    result = await pro_matcher.match(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found or no metrics available.")
    return result


@router.get("/{session_id}/tour/{tour}")
async def tour_comparison(
    session_id: str,
    tour: str = "PGA",
    pro_matcher: ProMatcher = Depends(get_pro_matcher),
):
    """Compare user metrics against tour averages."""
    result = await pro_matcher.get_tour_comparison(session_id, tour)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return result


@router.get("/pros/list", response_model=list[ProProfile])
async def list_pros(pro_matcher: ProMatcher = Depends(get_pro_matcher)):
    """List all pros in the reference database."""
    return await pro_matcher.list_pros()
