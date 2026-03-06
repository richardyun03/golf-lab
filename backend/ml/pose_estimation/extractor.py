import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from app.core.config import Settings
from app.services.video_processor import VideoData
from app.schemas.analysis import PoseKeypoint

# MediaPipe landmark index → name mapping for golf-relevant keypoints.
# Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# We keep 13 body landmarks that matter for swing analysis.
LANDMARK_MAP = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


class PoseExtractor:
    """
    Extracts 3D pose keypoints from every frame of a swing video
    using MediaPipe PoseLandmarker (Tasks API).

    Output: list of keypoints per frame, normalized to [0, 1] image coordinates.
    World landmarks (x, y, z in meters, hip-centered) are used for z-depth.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.landmarker = self._load_model()

    def _load_model(self) -> PoseLandmarker:
        if self.settings.pose_model != "mediapipe":
            raise ValueError(f"Unknown pose model: {self.settings.pose_model}")

        model_path = str(self.settings.model_weights_dir / "pose_landmarker_heavy.task")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return PoseLandmarker.create_from_options(options)

    async def extract(self, video: VideoData) -> list[list[PoseKeypoint]]:
        """
        Run pose estimation on all frames.

        Returns a list of length frame_count where each element is a list
        of PoseKeypoint objects for the golf-relevant landmarks in that frame.
        Frames where no pose is detected return an empty list.
        """
        results: list[list[PoseKeypoint]] = []
        frame_interval_ms = int(1000 / video.fps) if video.fps > 0 else 33

        for i, frame in enumerate(video.raw_frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = i * frame_interval_ms

            pose_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if not pose_result.pose_landmarks:
                results.append([])
                continue

            # Take first detected pose
            image_landmarks = pose_result.pose_landmarks[0]
            world_landmarks = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None

            keypoints = self._parse_landmarks(image_landmarks, world_landmarks)
            results.append(keypoints)

        return results

    def _parse_landmarks(self, image_landmarks, world_landmarks) -> list[PoseKeypoint]:
        """
        Convert MediaPipe landmarks to PoseKeypoint schema.

        Uses image-space (x, y) for 2D position and world-space z for depth.
        """
        keypoints: list[PoseKeypoint] = []

        for idx, name in LANDMARK_MAP.items():
            lm = image_landmarks[idx]
            z = world_landmarks[idx].z if world_landmarks else None

            keypoints.append(PoseKeypoint(
                name=name,
                x=lm.x,
                y=lm.y,
                z=z,
                confidence=lm.visibility,
            ))

        return keypoints

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
