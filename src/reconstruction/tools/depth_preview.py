#!/usr/bin/env python3
"""
Preview depth RAW + IR PNG and export a point cloud (single frame).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview IR + depth raw")
    parser.add_argument("--depth-raw", required=True)
    parser.add_argument("--ir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--out-dir", default="output/tests")
    parser.add_argument("--max-depth", type=float, default=8.0)
    return parser.parse_args()


def depth_to_points(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    h, w = depth_m.shape
    ys, xs = np.indices((h, w))
    z = depth_m
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_raw = np.fromfile(args.depth_raw, dtype=np.uint16)
    expected = args.width * args.height
    if depth_raw.size != expected:
        raise SystemExit(
            f"Depth size mismatch: {depth_raw.size} vs expected {expected}"
        )

    depth_raw = depth_raw.reshape(args.height, args.width)
    depth_m = depth_raw.astype(np.float32) * args.depth_scale
    valid = (depth_m > 0) & (depth_m < args.max_depth)

    depth_vis = depth_m.copy()
    depth_vis[~valid] = 0
    depth_vis = (depth_vis / args.max_depth * 255.0).clip(0, 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    ir = cv2.imread(args.ir, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        raise SystemExit(f"No se pudo leer IR: {args.ir}")
    ir_vis = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(out_dir / "depth_colormap.png"), depth_vis)
    cv2.imwrite(str(out_dir / "ir.png"), ir_vis)

    points = depth_to_points(depth_m, args.fx, args.fy, args.cx, args.cy)
    points = points[valid]

    ply_path = out_dir / "single_frame_cloud.ply"
    with open(ply_path, "w", encoding="utf-8") as ply:
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {points.shape[0]}\n")
        ply.write("property float x\nproperty float y\nproperty float z\n")
        ply.write("end_header\n")
        for x, y, z in points:
            ply.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    print("IR preview:", out_dir / "ir.png")
    print("Depth preview:", out_dir / "depth_colormap.png")
    print("Point cloud:", ply_path)


if __name__ == "__main__":
    main()
