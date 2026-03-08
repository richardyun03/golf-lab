"""
Compare SwingNet (GolfDB) ML-based phase detection vs our rule-based classifier.

Usage:
    cd backend
    python -m scripts.compare_swingnet --video-dir ../data/sample_videos

Requires:
    - SwingNet weights at data/golfdb/models/swingnet_1800.pth.tar
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ── SwingNet model (adapted from GolfDB) ──────────────────────────────

# Inline MobileNetV2 to avoid path issues with imports
def _conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

def _conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
            [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
        ]
        input_channel = int(input_channel * width_mult) if width_mult >= 1.0 else input_channel
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [_conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = max(int(c * width_mult), 16)
            for i in range(n):
                self.features.append(
                    InvertedResidual(input_channel, output_channel, s if i == 0 else 1, expand_ratio=t)
                )
                input_channel = output_channel
        self.features.append(_conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features(x)
        return x.mean(3).mean(2)

class EventDetector(nn.Module):
    def __init__(self, width_mult=1.0, lstm_layers=1, lstm_hidden=256,
                 bidirectional=True, dropout=False):
        super().__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        net = MobileNetV2(width_mult=width_mult)
        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.rnn = nn.LSTM(
            int(1280 * width_mult if width_mult > 1.0 else 1280),
            self.lstm_hidden, self.lstm_layers,
            batch_first=True, bidirectional=bidirectional,
        )
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size, device):
        d = 2 * self.lstm_layers if self.bidirectional else self.lstm_layers
        return (
            torch.zeros(d, batch_size, self.lstm_hidden, device=device, requires_grad=True),
            torch.zeros(d, batch_size, self.lstm_hidden, device=device, requires_grad=True),
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        device = x.device
        self.hidden = self.init_hidden(batch_size, device)
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)
        return out


# ── SwingNet event names (GolfDB) ─────────────────────────────────────

SWINGNET_EVENTS = {
    0: "Address",
    1: "Toe-up",
    2: "Mid-backswing",
    3: "Top",
    4: "Mid-downswing",
    5: "Impact",
    6: "Mid-follow-through",
    7: "Finish",
}

# ── Our phase names ───────────────────────────────────────────────────

OUR_PHASES = [
    "ADDRESS", "TAKEAWAY", "BACKSWING", "TOP",
    "DOWNSWING", "IMPACT", "FOLLOW_THROUGH", "FINISH",
]

# Mapping: SwingNet index -> our phase name
SWINGNET_TO_OURS = {
    0: "ADDRESS",
    1: "TAKEAWAY",       # Toe-up ≈ takeaway
    2: "BACKSWING",      # Mid-backswing ≈ backswing
    3: "TOP",
    4: "DOWNSWING",      # Mid-downswing ≈ downswing
    5: "IMPACT",
    6: "FOLLOW_THROUGH", # Mid-follow-through ≈ follow-through
    7: "FINISH",
}


# ── Video preprocessing (from GolfDB test_video.py) ──────────────────

class ToTensor:
    def __call__(self, sample):
        images, labels = sample["images"], sample["labels"]
        images = images.transpose((0, 3, 1, 2))
        return {
            "images": torch.from_numpy(images).float().div(255.0),
            "labels": torch.from_numpy(labels).long(),
        }

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample["images"], sample["labels"]
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {"images": images, "labels": labels}


def load_video_for_swingnet(path: str, input_size: int = 160):
    """Load and preprocess a video for SwingNet inference."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    ratio = input_size / max(frame_h, frame_w)
    new_h, new_w = int(frame_h * ratio), int(frame_w * ratio)
    delta_w = input_size - new_w
    delta_h = input_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    images = []
    for _ in range(frame_count):
        ret, img = cap.read()
        if not ret:
            break
        resized = cv2.resize(img, (new_w, new_h))
        bordered = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=[0.406 * 255, 0.456 * 255, 0.485 * 255],
        )
        images.append(cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB))
    cap.release()

    sample = {"images": np.asarray(images), "labels": np.zeros(len(images))}
    transform = transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    sample = transform(sample)
    return sample["images"].unsqueeze(0), fps, frame_count


# ── Run our rule-based classifier ────────────────────────────────────

def run_rule_based(video_path: str):
    """Run our pipeline: MediaPipe pose -> rule-based classifier."""
    # Add backend to path
    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from app.core.config import Settings
    from ml.pose_estimation.extractor import PoseExtractor
    from ml.swing_analysis.classifier import SwingPhaseClassifier
    from app.services.video_processor import VideoData

    settings = Settings()
    extractor = PoseExtractor(settings)
    classifier = SwingPhaseClassifier(settings)

    # Load video frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    import uuid
    video_data = VideoData(
        session_id=str(uuid.uuid4()),
        file_path=Path(video_path),
        fps=fps,
        frame_count=len(frames),
        duration_seconds=len(frames) / fps if fps > 0 else 0,
        width=width,
        height=height,
        raw_frames=frames,
    )

    import asyncio
    keypoints = asyncio.run(extractor.extract(video_data))
    phases = classifier.classify(keypoints)

    return {phase.name: frame_idx for phase, frame_idx in phases.items()}


# ── Run SwingNet ─────────────────────────────────────────────────────

def run_swingnet(video_path: str, model: EventDetector, device: torch.device, seq_length: int = 64):
    """Run SwingNet inference and return event frame predictions with confidences."""
    images, fps, frame_count = load_video_for_swingnet(video_path)

    with torch.no_grad():
        batch = 0
        probs = None
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.to(device))
            batch_probs = F.softmax(logits.data, dim=1).cpu().numpy()
            if probs is None:
                probs = batch_probs
            else:
                probs = np.append(probs, batch_probs, 0)
            batch += 1

    # Events: argmax across frames for each event class (columns 0-7, ignoring 8=no-event)
    events = np.argmax(probs, axis=0)[:-1]  # shape (8,)
    confidences = [probs[events[i], i] for i in range(8)]

    result = {}
    for i in range(8):
        phase_name = SWINGNET_TO_OURS[i]
        result[phase_name] = {
            "frame": int(events[i]),
            "confidence": float(confidences[i]),
            "swingnet_name": SWINGNET_EVENTS[i],
        }
    return result


# ── Visualization ────────────────────────────────────────────────────

def save_phase_grid(video_path: str, sn_result: dict, rb_result: dict | None,
                    output_path: Path):
    """Save a grid image showing detected phase frames from both methods."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return

    # Determine grid layout: 1 or 2 rows x 8 columns
    n_phases = len(OUR_PHASES)
    has_rb = rb_result is not None
    n_rows = 2 if has_rb else 1

    # Layout: 2 rows of 4 phases per method block
    sample = frames[0]
    aspect = sample.shape[1] / sample.shape[0]
    thumb_w = 400
    thumb_h = int(thumb_w / aspect)
    label_h = 70  # space for text above each thumbnail
    row_label_h = 50  # space for method name header
    cols_per_row = 4

    block_rows = 2  # 8 phases across 2 rows of 4
    canvas_w = cols_per_row * thumb_w
    block_h = row_label_h + block_rows * (thumb_h + label_h)
    canvas_h = n_rows * block_h + 20

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = 40  # dark gray background

    font = cv2.FONT_HERSHEY_SIMPLEX

    for row in range(n_rows):
        if row == 0:
            method_name = "SwingNet"
            result = sn_result
        else:
            method_name = "Rule-based"
            result = rb_result

        block_y = row * block_h

        # Method header
        cv2.putText(canvas, method_name, (15, block_y + 35),
                    font, 1.2, (100, 200, 255), 2)

        for idx, phase in enumerate(OUR_PHASES):
            grid_row = idx // cols_per_row
            grid_col = idx % cols_per_row

            x = grid_col * thumb_w
            y = block_y + row_label_h + grid_row * (thumb_h + label_h)

            if row == 0:
                frame_idx = result[phase]["frame"]
                conf = result[phase]["confidence"]
                label_text = f"{phase}  frame {frame_idx}"
                conf_text = f"confidence: {conf:.3f}"
            else:
                frame_idx = result.get(phase, 0)
                label_text = f"{phase}  frame {frame_idx}"
                conf_text = ""

            frame_idx = max(0, min(frame_idx, len(frames) - 1))

            # Phase label
            cv2.putText(canvas, label_text, (x + 8, y + 25),
                        font, 0.7, (255, 255, 255), 2)
            if conf_text:
                color = (0, 255, 0) if conf > 0.3 else (0, 200, 255) if conf > 0.1 else (0, 0, 255)
                cv2.putText(canvas, conf_text, (x + 8, y + 55),
                            font, 0.65, color, 2)

            # Thumbnail
            thumb = cv2.resize(frames[frame_idx], (thumb_w - 6, thumb_h - 6))
            ty = y + label_h
            canvas[ty:ty + thumb_h - 6, x + 3:x + thumb_w - 3] = thumb

    cv2.imwrite(str(output_path), canvas)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare SwingNet vs rule-based phase detection")
    parser.add_argument("--video-dir", type=str, default="../data/sample_videos",
                        help="Directory with sample videos")
    parser.add_argument("--video", type=str, default=None,
                        help="Single video to test (overrides --video-dir)")
    parser.add_argument("--weights", type=str,
                        default="../data/golfdb/models/swingnet_1800.pth.tar",
                        help="Path to SwingNet weights")
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--skip-rule-based", action="store_true",
                        help="Skip rule-based (much faster, SwingNet only)")
    parser.add_argument("--output-dir", type=str, default="output/swingnet_comparison",
                        help="Directory to save visual output")
    args = parser.parse_args()

    # Load SwingNet model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"SwingNet weights not found at {weights_path}")
        print("Download from: https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = EventDetector(
        width_mult=1.0, lstm_layers=1, lstm_hidden=256,
        bidirectional=True, dropout=False,
    )
    save_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(save_dict["model_state_dict"])
    model.to(device)
    model.eval()
    print("Loaded SwingNet model\n")

    # Collect videos
    if args.video:
        videos = [Path(args.video)]
    else:
        video_dir = Path(args.video_dir)
        videos = sorted([
            p for p in video_dir.iterdir()
            if p.suffix.lower() in {".mp4", ".mov", ".avi"}
            and not p.name.startswith("output")
        ])

    if not videos:
        print("No videos found.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'Video':<20} {'Phase':<18} {'SwingNet':>10} {'Rule-based':>12} {'Diff':>6}  {'SN Conf':>8}")
    print("-" * 80)

    for video_path in videos:
        print(f"\n{video_path.name}")
        print("-" * 80)

        # SwingNet
        sn_result = run_swingnet(str(video_path), model, device, args.seq_length)

        # Rule-based
        if not args.skip_rule_based:
            rb_result = run_rule_based(str(video_path))
        else:
            rb_result = None

        for phase in OUR_PHASES:
            sn = sn_result[phase]
            sn_frame = sn["frame"]
            sn_conf = sn["confidence"]
            sn_name = sn["swingnet_name"]

            if rb_result and phase in rb_result:
                rb_frame = rb_result[phase]
                diff = sn_frame - rb_frame
                diff_str = f"{diff:+d}"
            else:
                rb_frame = "-"
                diff_str = "-"

            print(f"  {phase:<18} {sn_frame:>10}   {str(rb_frame):>10} {diff_str:>6}  {sn_conf:>8.3f}  ({sn_name})")

        # Save visual grid
        stem = video_path.stem
        grid_path = output_dir / f"{stem}_phases.png"
        save_phase_grid(str(video_path), sn_result, rb_result, grid_path)
        print(f"  -> Saved: {grid_path}")

    print("\n" + "=" * 80)
    print(f"Done. Visual outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
