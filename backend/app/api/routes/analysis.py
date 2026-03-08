import shutil
from pathlib import Path

import cv2
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from app.schemas.analysis import AnalysisResponse
from app.services.video_processor import VideoProcessor
from app.services.swing_analyzer import SwingAnalyzer
from app.services.storage import AnalysisStorage
from app.core.dependencies import get_video_processor, get_swing_analyzer, get_storage
from app.core.config import get_settings
from ml.swing_analysis.skeleton_renderer import render_phase_frames

router = APIRouter()

settings = get_settings()
VIDEOS_DIR = settings.data_dir / "uploads"
FRAMES_DIR = settings.data_dir / "phase_frames"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/", response_model=AnalysisResponse)
async def analyze_swing(
    video: UploadFile = File(...),
    video_processor: VideoProcessor = Depends(get_video_processor),
    swing_analyzer: SwingAnalyzer = Depends(get_swing_analyzer),
    storage: AnalysisStorage = Depends(get_storage),
):
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

    # Persist video file for playback
    video_dest = VIDEOS_DIR / f"{result.session_id}.{ext}"
    shutil.copy2(str(video_data.file_path), str(video_dest))

    # Render and save skeleton-overlay frames for each phase
    session_frames_dir = FRAMES_DIR / result.session_id
    session_frames_dir.mkdir(parents=True, exist_ok=True)
    phase_frames = render_phase_frames(
        video_data.raw_frames,
        result.keypoints_by_frame,
        result.swing_phases,
    )
    for phase_name, frame_img in phase_frames.items():
        out_path = session_frames_dir / f"{phase_name}.jpg"
        cv2.imwrite(str(out_path), frame_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    storage.save(response)
    return response


@router.get("/sessions", response_model=list[dict])
async def list_sessions(storage: AnalysisStorage = Depends(get_storage)):
    return storage.list_sessions()


@router.get("/{session_id}/video")
async def get_video(session_id: str):
    """Serve the uploaded video for playback."""
    for ext in ("mp4", "mov", "avi"):
        path = VIDEOS_DIR / f"{session_id}.{ext}"
        if path.exists():
            media_types = {"mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo"}
            return FileResponse(path, media_type=media_types[ext])
    raise HTTPException(status_code=404, detail="Video not found.")


@router.get("/{session_id}/frames/{phase}")
async def get_phase_frame(session_id: str, phase: str):
    """Serve the skeleton-overlay frame for a specific swing phase."""
    frame_path = FRAMES_DIR / session_id / f"{phase}.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Phase frame not found.")
    return FileResponse(frame_path, media_type="image/jpeg")


@router.get("/{session_id}", response_model=AnalysisResponse)
async def get_analysis(
    session_id: str,
    storage: AnalysisStorage = Depends(get_storage),
):
    result = storage.get(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return result
