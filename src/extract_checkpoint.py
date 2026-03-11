#!/usr/bin/env python3
"""Copy a checkpoint to a new file (e.g. extract 5k-episode weights as standalone model).

Use when you have trained with --save-checkpoint-every N, which creates files like
  models/dqn_03092026_0126_shooting_curriculum_ep5000.pt

Example:
  python src/extract_checkpoint.py models/dqn_xxx_ep5000.pt -o models/dqn_5k.pt

If you only have the final model (no _ep5000 file), you cannot extract 5k weights.
Re-train with: --save-checkpoint-every 5000
"""

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Copy a checkpoint to a new path (e.g. extract episode-N weights)"
    )
    ap.add_argument("input", help="Input checkpoint path (e.g. model_ep5000.pt)")
    ap.add_argument("-o", "--output", required=True, help="Output path for the copied model")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        print(f"Error: {src} not found")
        print("\nTo get episode-N weights, train with --save-checkpoint-every N")
        print("  e.g. --save-checkpoint-every 5000 creates model_ep5000.pt")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")
    return 0


if __name__ == "__main__":
    exit(main())
