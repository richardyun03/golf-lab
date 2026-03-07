from functools import lru_cache
from fastapi import Depends
from app.core.config import Settings, get_settings
from app.services.video_processor import VideoProcessor
from app.services.swing_analyzer import SwingAnalyzer
from app.services.pro_matcher import ProMatcher


def get_video_processor(settings: Settings = Depends(get_settings)) -> VideoProcessor:
    return VideoProcessor(settings)


@lru_cache
def _create_swing_analyzer() -> SwingAnalyzer:
    """Singleton — loads ML models once and reuses across requests."""
    return SwingAnalyzer(get_settings())


def get_swing_analyzer() -> SwingAnalyzer:
    return _create_swing_analyzer()


def get_pro_matcher(settings: Settings = Depends(get_settings)) -> ProMatcher:
    return ProMatcher(settings)
