from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.schemas.analysis import AnalysisResult
from app.services.video_processor import VideoProcessor
from app.services.swing_analyzer import SwingAnalyzer
from app.core.dependencies import get_video_processor, get_swing_analyzer

router = APIRouter()


@router.post("/", response_model=AnalysisResult)
async def analyze_swing(
    video: UploadFile = File(...),
    video_processor: VideoProcessor = Depends(get_video_processor),
    swing_analyzer: SwingAnalyzer = Depends(get_swing_analyzer),
):
    """
    Upload a swing video and receive AI-powered analysis including:
    - Pose keypoints for every frame
    - Swing phase segmentation
    - Biomechanical metrics (hip/shoulder rotation, spine angle, tempo, etc.)
    - Fault detection with correction cues
    - Overall swing score
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = video.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"mp4", "mov", "avi"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    video_data = await video_processor.load(video)
    result = await swing_analyzer.analyze(video_data)
    return result


@router.get("/{session_id}", response_model=AnalysisResult)
async def get_analysis(session_id: str):
    """Retrieve a previously computed analysis by session ID."""
    raise HTTPException(status_code=404, detail="Session not found (storage not yet implemented).")
