"""
Build the pro swing embedding index from raw pro swing videos.

Usage:
    python scripts/build_pro_index.py --pros_dir data/pro_swings/videos --out data/pro_swings/

Steps:
1. For each pro video in pros_dir, run the full pose extraction pipeline
2. Compute swing embedding via SwingEmbedder
3. Save embeddings.pt (tensor matrix) and profiles.json
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build pro swing embedding index")
    parser.add_argument("--pros_dir", type=Path, default=Path("data/pro_swings/videos"))
    parser.add_argument("--out", type=Path, default=Path("data/pro_swings"))
    args = parser.parse_args()

    print(f"Scanning {args.pros_dir} for pro swing videos...")
    # TODO:
    # 1. Load settings and initialize VideoProcessor + PoseExtractor + SwingEmbedder
    # 2. Iterate over videos, extract embeddings
    # 3. Save torch.stack(embeddings) to args.out / "embeddings.pt"
    # 4. Save pro profiles to args.out / "profiles.json"
    print("Not yet implemented — implement after pose extraction pipeline is complete.")


if __name__ == "__main__":
    main()
