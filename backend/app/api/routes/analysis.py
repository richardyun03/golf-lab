import json
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Form, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from app.schemas.analysis import AnalysisResponse, ClubType, PoseKeypoint, SessionTrendPoint
from app.services.video_processor import VideoProcessor
from app.services.swing_analyzer import SwingAnalyzer
from app.services.storage import AnalysisStorage
from app.core.dependencies import get_video_processor, get_swing_analyzer, get_storage
from app.core.config import get_settings
from ml.swing_analysis.skeleton_renderer import draw_skeleton
from ml.pro_comparison.comparison_renderer import draw_comparison_frame
from ml.pro_comparison.pro_database import PRO_PROFILES
from ml.swing_analysis.club_profiles import get_ideal_ranges

router = APIRouter()

settings = get_settings()
VIDEOS_DIR = settings.data_dir / "uploads"
KEYPOINTS_DIR = settings.data_dir / "phase_keypoints"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────

def _find_video_path(session_id: str) -> Path | None:
    for ext in ("mp4", "mov", "avi"):
        path = VIDEOS_DIR / f"{session_id}.{ext}"
        if path.exists():
            return path
    return None


def _extract_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    """Extract a single frame from a video file by index."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _load_keypoints(session_id: str, phase: str) -> list[PoseKeypoint] | None:
    kp_path = KEYPOINTS_DIR / session_id / f"{phase}.json"
    if not kp_path.exists():
        return None
    kp_list = json.loads(kp_path.read_text())
    return [PoseKeypoint(**kp) for kp in kp_list]


def _render_to_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ── Routes ───────────────────────────────────────────────────────────

@router.post("/", response_model=AnalysisResponse)
async def analyze_swing(
    video: UploadFile = File(...),
    club_type: str | None = Form(None),
    video_processor: VideoProcessor = Depends(get_video_processor),
    swing_analyzer: SwingAnalyzer = Depends(get_swing_analyzer),
    storage: AnalysisStorage = Depends(get_storage),
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = video.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"mp4", "mov", "avi"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    # Validate club_type if provided
    resolved_club: ClubType | None = None
    if club_type:
        try:
            resolved_club = ClubType(club_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid club type: {club_type}")

    video_data = await video_processor.load(video)
    video_data.club_type = club_type
    result = await swing_analyzer.analyze(video_data)
    response = AnalysisResponse(
        session_id=result.session_id,
        club_type=resolved_club,
        video_duration_seconds=result.video_duration_seconds,
        fps=result.fps,
        swing_phases=result.swing_phases,
        metrics=result.metrics,
        ideal_ranges=get_ideal_ranges(club_type),
        faults=result.faults,
        overall_score=result.overall_score,
        phase_scores=result.phase_scores,
        summary=result.summary,
        frame_count=len(result.keypoints_by_frame),
    )

    # Persist video file
    video_dest = VIDEOS_DIR / f"{result.session_id}.{ext}"
    shutil.copy2(str(video_data.file_path), str(video_dest))

    # Persist keypoints per phase (tiny JSON files — only data we store per session)
    kp_dir = KEYPOINTS_DIR / result.session_id
    kp_dir.mkdir(parents=True, exist_ok=True)
    for phase, frame_idx in result.swing_phases.items():
        phase_key = phase.value if hasattr(phase, "value") else str(phase)
        idx = min(frame_idx, len(result.keypoints_by_frame) - 1)
        kps = result.keypoints_by_frame[idx] if idx < len(result.keypoints_by_frame) else []
        kp_data = [kp.model_dump() for kp in kps]
        (kp_dir / f"{phase_key}.json").write_text(json.dumps(kp_data))

    storage.save(response)
    return response


@router.get("/sessions", response_model=list[dict])
async def list_sessions(storage: AnalysisStorage = Depends(get_storage)):
    return storage.list_sessions()


@router.get("/sessions/trends", response_model=list[SessionTrendPoint])
async def get_session_trends(storage: AnalysisStorage = Depends(get_storage)):
    return storage.list_sessions_with_metrics()


@router.get("/{session_id}/video")
async def get_video(session_id: str):
    """Serve the uploaded video for playback."""
    video_path = _find_video_path(session_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video not found.")
    media_types = {"mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo"}
    ext = video_path.suffix.lstrip(".")
    return FileResponse(video_path, media_type=media_types.get(ext, "video/mp4"))


@router.get("/{session_id}/frames/{phase}")
async def get_phase_frame(
    session_id: str,
    phase: str,
    storage: AnalysisStorage = Depends(get_storage),
):
    """Render skeleton-overlay frame on-demand from video + keypoints."""
    session = storage.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    phase_idx = session.swing_phases.get(phase)
    if phase_idx is None:
        raise HTTPException(status_code=404, detail="Phase not found.")

    video_path = _find_video_path(session_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video not found.")

    frame = _extract_frame(video_path, phase_idx)
    if frame is None:
        raise HTTPException(status_code=500, detail="Could not extract frame.")

    keypoints = _load_keypoints(session_id, phase)
    if keypoints is None:
        raise HTTPException(status_code=404, detail="Keypoints not found.")

    rendered = draw_skeleton(frame, keypoints, phase=phase)
    return Response(content=_render_to_jpeg(rendered), media_type="image/jpeg")


@router.get("/{session_id}/compare/{pro_id}/frames/{phase}")
async def get_comparison_frame(
    session_id: str,
    pro_id: str,
    phase: str,
    storage: AnalysisStorage = Depends(get_storage),
):
    """Render comparison frame showing user skeleton with pro angle annotations."""
    session = storage.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    phase_idx = session.swing_phases.get(phase)
    if phase_idx is None:
        raise HTTPException(status_code=404, detail="Phase not found.")

    video_path = _find_video_path(session_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video not found.")

    pro_data = next((p for p in PRO_PROFILES if p["pro_id"] == pro_id), None)
    if pro_data is None:
        raise HTTPException(status_code=404, detail="Pro not found.")

    frame = _extract_frame(video_path, phase_idx)
    if frame is None:
        raise HTTPException(status_code=500, detail="Could not extract frame.")

    keypoints = _load_keypoints(session_id, phase)
    if keypoints is None:
        raise HTTPException(status_code=404, detail="Keypoints not found.")

    rendered = draw_comparison_frame(
        frame, keypoints, phase,
        {**pro_data["metrics"], "_name": pro_data["name"]},
    )
    return Response(content=_render_to_jpeg(rendered), media_type="image/jpeg")


@router.get("/{session_id}", response_model=AnalysisResponse)
async def get_analysis(
    session_id: str,
    storage: AnalysisStorage = Depends(get_storage),
):
    result = storage.get(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not result.ideal_ranges:
        club = result.club_type.value if result.club_type else None
        result.ideal_ranges = get_ideal_ranges(club)
    return result
