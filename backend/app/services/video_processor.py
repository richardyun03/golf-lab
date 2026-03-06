import uuid
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from fastapi import UploadFile, HTTPException
from app.core.config import Settings


@dataclass
class VideoData:
    session_id: str
    file_path: Path
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int
    raw_frames: list[np.ndarray] = field(default_factory=list)


class VideoProcessor:
    """Handles video ingestion, validation, and frame extraction."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def load(self, upload: UploadFile) -> VideoData:
        """
        Save uploaded video to a temp file, extract metadata, and load frames.
        Resamples to target_fps if the source FPS differs.
        """
        contents = await upload.read()

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > self.settings.max_video_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"Video too large ({size_mb:.1f}MB). Max is {self.settings.max_video_size_mb}MB.",
            )

        suffix = Path(upload.filename).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(contents)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if source_fps <= 0 or total_frames <= 0:
            cap.release()
            raise HTTPException(status_code=400, detail="Invalid video: could not read FPS or frame count.")

        duration = total_frames / source_fps
        target_fps = self.settings.target_fps

        # Determine which frames to keep for target FPS resampling
        if target_fps >= source_fps:
            frame_indices = set(range(total_frames))
        else:
            ratio = source_fps / target_fps
            frame_indices = {int(i * ratio) for i in range(int(total_frames / ratio))}

        frames: list[np.ndarray] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_indices:
                frames.append(frame)
            frame_idx += 1

        cap.release()

        session_id = str(uuid.uuid4())
        output_fps = min(target_fps, source_fps)

        return VideoData(
            session_id=session_id,
            file_path=Path(tmp.name),
            fps=output_fps,
            frame_count=len(frames),
            duration_seconds=duration,
            width=width,
            height=height,
            raw_frames=frames,
        )
