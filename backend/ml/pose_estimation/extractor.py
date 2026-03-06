from app.core.config import Settings
from app.services.video_processor import VideoData
from app.schemas.analysis import PoseKeypoint

# MediaPipe keypoint indices relevant to golf swing analysis
GOLF_KEYPOINTS = [
    "nose", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


class PoseExtractor:
    """
    Extracts 3D pose keypoints from every frame of a swing video.

    Supported backends:
    - mediapipe: Google MediaPipe Pose (default, no GPU needed)
    - vitpose:   ViTPose transformer model (higher accuracy, GPU recommended)

    Output: list of keypoints per frame, normalized to [0, 1] image coordinates.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self._load_model()

    def _load_model(self):
        if self.settings.pose_model == "mediapipe":
            # import mediapipe as mp
            # return mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)
            return None  # TODO: initialize MediaPipe Pose
        raise ValueError(f"Unknown pose model: {self.settings.pose_model}")

    async def extract(self, video: VideoData) -> list[list[PoseKeypoint]]:
        """
        Run pose estimation on all frames.

        Returns a list of length frame_count where each element is a list
        of PoseKeypoint objects for that frame.
        """
        # TODO: Implement frame-by-frame MediaPipe inference
        # results = []
        # for frame in video.raw_frames:
        #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     pose_result = self.model.process(rgb)
        #     keypoints = self._parse_landmarks(pose_result.pose_world_landmarks)
        #     results.append(keypoints)
        # return results
        return []

    def _parse_landmarks(self, landmarks) -> list[PoseKeypoint]:
        """Convert MediaPipe landmark output to PoseKeypoint schema."""
        return []
