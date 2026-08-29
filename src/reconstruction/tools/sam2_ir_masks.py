#!/usr/bin/env python3
"""
Generate solid masks using SAM2 on IR images and refine with depth.
Outputs one mask per image and overlay for review.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 masks from IR + depth")
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--model", default="FastSAM-s.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument("--max-component-ratio", type=float, default=0.7)
    parser.add_argument("--exclude-border", action="store_true")
    parser.add_argument("--border-margin", type=int, default=8)
    parser.add_argument("--morph-kernel", type=int, default=5)
    parser.add_argument("--morph-iter", type=int, default=2)
    return parser.parse_args()


def _load_pairs(ir_dir: Path, depth_dir: Path) -> List[Tuple[Path, Path]]:
    ir_files = sorted(ir_dir.glob("yolo_data_*.jpg")) + sorted(ir_dir.glob("yolo_data_*.png"))
    depth_files = {p.stem: p for p in depth_dir.glob("yolo_data_*.png")}
    pairs = []
    for ir in ir_files:
        depth = depth_files.get(ir.stem)
        if depth is not None:
            pairs.append((ir, depth))
    return pairs


def _refine_with_depth(mask: np.ndarray, depth_raw: np.ndarray, depth_scale: float, max_depth: float) -> np.ndarray:
    depth_raw = np.squeeze(depth_raw)
    depth_m = depth_raw.astype(np.float32) * depth_scale
    valid = (depth_m > 0) & (depth_m < max_depth)
    if mask.ndim > 2:
        mask = mask[..., 0]
    refined = np.zeros_like(mask, dtype=np.uint8)
    refined[mask > 0] = 255
    refined[~valid] = 0
    return refined


def _filter_components(
    mask: np.ndarray,
    min_area: int,
    max_component_ratio: float,
    exclude_border: bool,
    border_margin: int,
) -> np.ndarray:
    if mask.ndim > 2:
        mask = mask[..., 0]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    total_area = mask.shape[0] * mask.shape[1]
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if area / total_area > max_component_ratio:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        if exclude_border:
            if x <= border_margin or y <= border_margin:
                continue
            if x + w >= mask.shape[1] - border_margin:
                continue
            if y + h >= mask.shape[0] - border_margin:
                continue

        cleaned[labels == label] = 255
    return cleaned


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs(Path(args.ir_dir), Path(args.depth_dir))
    if not pairs:
        raise SystemExit("No se encontraron pares IR+Depth")

    if args.frame_step > 1:
        pairs = pairs[:: args.frame_step]

    from ultralytics import FastSAM

    model = FastSAM(args.model)

    for ir_path, depth_path in pairs:
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if ir is None or depth_raw is None:
            continue

        if depth_raw.dtype != np.uint16:
            raise SystemExit(f"Depth no es uint16: {depth_path}")

        ir_color = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        results = model(ir_color, conf=args.conf, verbose=False)
        if not results:
            continue

        result = results[0]
        depth_raw_sq = np.squeeze(depth_raw)
        depth_m = depth_raw_sq.astype(np.float32) * args.depth_scale
        valid = (depth_m > 0) & (depth_m < args.max_depth)

        accum = np.zeros(ir.shape[:2], dtype=np.uint8)
        h_img, w_img = ir.shape[:2]
        total_px = h_img * w_img
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.morph_kernel, args.morph_kernel)
        )

        if result.masks is not None and len(result.masks) > 0:
            raw_masks = result.masks.data.cpu().numpy()
            for idx in range(len(raw_masks)):
                raw = (raw_masks[idx] > 0.5).astype(np.uint8) * 255
                area = int(raw.sum() / 255)
                if area < args.min_area:
                    continue
                if area / total_px > args.max_component_ratio:
                    continue
                if args.exclude_border:
                    ys, xs = np.where(raw > 0)
                    if len(ys) == 0:
                        continue
                    x_min, x_max = int(xs.min()), int(xs.max())
                    y_min, y_max = int(ys.min()), int(ys.max())
                    if x_min <= args.border_margin or y_min <= args.border_margin:
                        continue
                    if x_max >= w_img - args.border_margin or y_max >= h_img - args.border_margin:
                        continue

                raw[~valid] = 0
                raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=args.morph_iter)
                raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=max(1, args.morph_iter // 2))

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
                for label in range(1, num_labels):
                    ca = int(stats[label, cv2.CC_STAT_AREA])
                    if ca < args.min_area:
                        continue
                    accum[labels == label] = 255

        mask_all = accum

        overlay = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        overlay[mask_all > 0] = (0, 200, 0)
        overlay = cv2.addWeighted(cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR), 0.6, overlay, 0.4, 0)

        mask_path = masks_dir / f"{ir_path.stem}_mask.png"
        overlay_path = overlays_dir / f"{ir_path.stem}_overlay.png"
        cv2.imwrite(str(mask_path), mask_all)
        cv2.imwrite(str(overlay_path), overlay)

    print("Masks:", masks_dir)
    print("Overlays:", overlays_dir)


if __name__ == "__main__":
    main()
