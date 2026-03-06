import uuid
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from fastapi import UploadFile
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
    raw_frames: list = field(default_factory=list)  # populated by load()


class VideoProcessor:
    """Handles video ingestion, validation, and frame extraction."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def load(self, upload: UploadFile) -> VideoData:
        """
        Save uploaded video to a temp file, extract metadata, and load frames.

        Steps:
        1. Write upload bytes to a temporary file
        2. Use OpenCV to read metadata (fps, dimensions, frame count)
        3. Extract and normalize frames at target_fps
        4. Return VideoData for downstream ML processing
        """
        session_id = str(uuid.uuid4())
        contents = await upload.read()

        suffix = Path(upload.filename).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(contents)
        tmp.flush()

        # TODO: Replace stub with real OpenCV frame extraction
        # import cv2
        # cap = cv2.VideoCapture(tmp.name)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return VideoData(
            session_id=session_id,
            file_path=Path(tmp.name),
            fps=30.0,
            frame_count=0,
            duration_seconds=0.0,
            width=0,
            height=0,
        )
