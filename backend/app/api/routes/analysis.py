from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.schemas.analysis import AnalysisResponse
from app.services.video_processor import VideoProcessor
from app.services.swing_analyzer import SwingAnalyzer
from app.services.storage import AnalysisStorage
from app.core.dependencies import get_video_processor, get_swing_analyzer, get_storage

router = APIRouter()


@router.post("/", response_model=AnalysisResponse)
async def analyze_swing(
    video: UploadFile = File(...),
    video_processor: VideoProcessor = Depends(get_video_processor),
    swing_analyzer: SwingAnalyzer = Depends(get_swing_analyzer),
    storage: AnalysisStorage = Depends(get_storage),
):
    """
    Upload a swing video and receive AI-powered analysis including:
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
    response = AnalysisResponse(
        session_id=result.session_id,
        video_duration_seconds=result.video_duration_seconds,
        fps=result.fps,
        swing_phases=result.swing_phases,
        metrics=result.metrics,
        faults=result.faults,
        overall_score=result.overall_score,
        summary=result.summary,
        frame_count=len(result.keypoints_by_frame),
    )
    storage.save(response)
    return response


@router.get("/sessions", response_model=list[dict])
async def list_sessions(storage: AnalysisStorage = Depends(get_storage)):
    """List recent analysis sessions."""
    return storage.list_sessions()


@router.get("/{session_id}", response_model=AnalysisResponse)
async def get_analysis(
    session_id: str,
    storage: AnalysisStorage = Depends(get_storage),
):
    """Retrieve a previously computed analysis by session ID."""
    result = storage.get(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return result
