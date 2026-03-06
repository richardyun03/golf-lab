import torch
import torch.nn as nn
from app.core.config import Settings
from app.schemas.analysis import PoseKeypoint


class SwingEncoder(nn.Module):
    """
    Encodes a variable-length sequence of pose keypoints into a fixed-size
    swing embedding using a Transformer encoder.

    Input:  (batch, time_steps, keypoint_features)
    Output: (batch, embedding_dim)

    The embedding captures the overall shape and tempo of a swing in a way
    that is invariant to minor timing differences (addressed by DTW at retrieval).
    """

    def __init__(self, keypoint_dim: int = 39, embedding_dim: int = 128, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(keypoint_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return nn.functional.normalize(x, dim=-1)


class SwingEmbedder:
    """Wraps SwingEncoder for inference — loads weights and produces embeddings."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = SwingEncoder(embedding_dim=settings.swing_embedding_dim)
        self._load_weights()

    def _load_weights(self):
        weights_path = self.settings.model_weights_dir / "swing_encoder.pt"
        if weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.settings.device))
        self.model.eval()

    def embed(self, keypoints_by_frame: list[list[PoseKeypoint]]) -> torch.Tensor:
        """Convert keypoints to a normalized embedding vector."""
        # TODO: flatten keypoints to tensor, run inference
        # features = self._keypoints_to_tensor(keypoints_by_frame)
        # with torch.no_grad():
        #     return self.model(features.unsqueeze(0)).squeeze(0)
        return torch.zeros(self.settings.swing_embedding_dim)

    def _keypoints_to_tensor(self, keypoints_by_frame) -> torch.Tensor:
        """Flatten (frames, keypoints) into a (frames, keypoint_dim) tensor."""
        raise NotImplementedError
