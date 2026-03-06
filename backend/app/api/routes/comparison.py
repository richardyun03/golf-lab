from fastapi import APIRouter, HTTPException, Depends
from app.schemas.comparison import ComparisonResult
from app.services.pro_matcher import ProMatcher
from app.core.dependencies import get_pro_matcher

router = APIRouter()


@router.post("/{session_id}", response_model=ComparisonResult)
async def compare_to_pros(
    session_id: str,
    pro_matcher: ProMatcher = Depends(get_pro_matcher),
):
    """
    Given a completed analysis session, match the user's swing against the
    pro database and return the top similar pros with detailed comparisons.

    Matching is done via cosine similarity on learned swing embeddings,
    with DTW alignment across swing phases for temporal robustness.
    """
    result = await pro_matcher.match(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return result


@router.get("/pros", response_model=list)
async def list_pros(pro_matcher: ProMatcher = Depends(get_pro_matcher)):
    """List all pros in the reference database."""
    return await pro_matcher.list_pros()
