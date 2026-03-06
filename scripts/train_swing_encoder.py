"""
Train the SwingEncoder model on labeled swing data.

Usage:
    python scripts/train_swing_encoder.py --data_dir data/training --epochs 50

Training approach: contrastive learning (SimCLR / triplet loss)
- Positive pairs: augmented versions of the same swing
- Negative pairs: swings from different golfers
- Loss: NT-Xent or triplet margin loss

Once trained, the encoder generalizes to embed any swing into a space
where similar swings cluster together.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/training"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=Path("data/model_weights/swing_encoder.pt"))
    args = parser.parse_args()

    print("Training not yet implemented.")
    # TODO: implement training loop


if __name__ == "__main__":
    main()
