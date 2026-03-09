import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from app.schemas.analysis import AnalysisResponse
from app.services.video_processor import VideoProcessor
from app.services.swing_analyzer import SwingAnalyzer
from app.services.storage import AnalysisStorage
from app.core.dependencies import get_video_processor, get_swing_analyzer, get_storage
from app.core.config import get_settings
from ml.swing_analysis.skeleton_renderer import render_phase_frames
from ml.pro_comparison.comparison_renderer import draw_comparison_frame
from ml.pro_comparison.pro_database import PRO_PROFILES

router = APIRouter()

settings = get_settings()
VIDEOS_DIR = settings.data_dir / "uploads"
FRAMES_DIR = settings.data_dir / "phase_frames"
RAW_FRAMES_DIR = settings.data_dir / "raw_phase_frames"
KEYPOINTS_DIR = settings.data_dir / "phase_keypoints"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)


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

    # Render and save skeleton-overlay frames with guide annotations
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

    # Save raw phase frames and keypoints for on-demand comparison rendering
    raw_dir = RAW_FRAMES_DIR / result.session_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    kp_dir = KEYPOINTS_DIR / result.session_id
    kp_dir.mkdir(parents=True, exist_ok=True)
    for phase, frame_idx in result.swing_phases.items():
        phase_key = phase.value if hasattr(phase, "value") else str(phase)
        idx = min(frame_idx, len(video_data.raw_frames) - 1)
        cv2.imwrite(str(raw_dir / f"{phase_key}.jpg"), video_data.raw_frames[idx], [cv2.IMWRITE_JPEG_QUALITY, 95])
        kps = result.keypoints_by_frame[idx] if idx < len(result.keypoints_by_frame) else []
        kp_data = [kp.model_dump() for kp in kps]
        (kp_dir / f"{phase_key}.json").write_text(json.dumps(kp_data))

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


@router.get("/{session_id}/compare/{pro_id}/frames/{phase}")
async def get_comparison_frame(session_id: str, pro_id: str, phase: str):
    """Render a comparison frame showing user skeleton with pro angle annotations."""
    # Load raw frame
    raw_path = RAW_FRAMES_DIR / session_id / f"{phase}.jpg"
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Raw phase frame not found.")

    # Load keypoints
    kp_path = KEYPOINTS_DIR / session_id / f"{phase}.json"
    if not kp_path.exists():
        raise HTTPException(status_code=404, detail="Keypoints not found.")

    # Find pro
    pro_data = next((p for p in PRO_PROFILES if p["pro_id"] == pro_id), None)
    if pro_data is None:
        raise HTTPException(status_code=404, detail="Pro not found.")

    from app.schemas.analysis import PoseKeypoint
    frame = cv2.imread(str(raw_path))
    kp_list = json.loads(kp_path.read_text())
    keypoints = [PoseKeypoint(**kp) for kp in kp_list]

    rendered = draw_comparison_frame(
        frame, keypoints, phase,
        {**pro_data["metrics"], "_name": pro_data["name"]},
    )

    _, buf = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.get("/{session_id}", response_model=AnalysisResponse)
async def get_analysis(
    session_id: str,
    storage: AnalysisStorage = Depends(get_storage),
):
    result = storage.get(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return result
