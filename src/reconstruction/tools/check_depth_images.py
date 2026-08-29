#!/usr/bin/env python3
"""
Check depth image bit-depth and basic stats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check depth image properties")
    parser.add_argument("paths", nargs="+", help="Depth image paths")
    parser.add_argument("--sample-unique", type=int, default=20)
    return parser.parse_args()


def summarize(path: Path, sample_unique: int) -> None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"{path}: FAILED to read")
        return

    dtype = img.dtype
    shape = img.shape
    min_val = int(img.min()) if img.size else 0
    max_val = int(img.max()) if img.size else 0
    unique_vals = np.unique(img)
    unique_sample = unique_vals[:sample_unique]
    unique_count = unique_vals.size

    print(f"{path}")
    print(f"  dtype: {dtype}")
    print(f"  shape: {shape}")
    print(f"  min/max: {min_val}/{max_val}")
    print(f"  unique count: {unique_count}")
    print(f"  unique sample: {unique_sample}")


def main() -> None:
    args = parse_args()
    for p in args.paths:
        summarize(Path(p), args.sample_unique)


if __name__ == "__main__":
    main()
