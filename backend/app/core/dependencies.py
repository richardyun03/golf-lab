from fastapi import Depends
from app.core.config import Settings, get_settings
from app.services.video_processor import VideoProcessor
from app.services.swing_analyzer import SwingAnalyzer
from app.services.pro_matcher import ProMatcher


def get_video_processor(settings: Settings = Depends(get_settings)) -> VideoProcessor:
    return VideoProcessor(settings)


def get_swing_analyzer(settings: Settings = Depends(get_settings)) -> SwingAnalyzer:
    return SwingAnalyzer(settings)


def get_pro_matcher(settings: Settings = Depends(get_settings)) -> ProMatcher:
    return ProMatcher(settings)
