from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "Golf Lab API"
    debug: bool = False

    # Paths
    data_dir: Path = BASE_DIR / "data"
    model_weights_dir: Path = BASE_DIR / "data" / "model_weights"
    pro_swings_dir: Path = BASE_DIR / "data" / "pro_swings"

    # Video processing
    max_video_size_mb: int = 500
    supported_video_formats: list[str] = ["mp4", "mov", "avi"]
    target_fps: int = 30
    pose_model: str = "mediapipe"  # "mediapipe" | "openpose" | "vitpose"

    # ML
    swing_embedding_dim: int = 128
    similarity_threshold: float = 0.75
    device: str = "cpu"  # "cpu" | "cuda" | "mps"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
