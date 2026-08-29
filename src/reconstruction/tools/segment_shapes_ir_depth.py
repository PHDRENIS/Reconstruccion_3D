#!/usr/bin/env python3
"""
Generate solid shape masks from IR + depth without labels.
Outputs one binary mask per image and an overlay preview.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment shapes from IR + depth")
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--depth-percentile", type=float, default=70.0)
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument("--max-component-ratio", type=float, default=0.7)
    parser.add_argument("--exclude-border", action="store_true")
    parser.add_argument("--border-margin", type=int, default=5)
    parser.add_argument("--frame-step", type=int, default=1)
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


def _auto_canny_thresholds(img: np.ndarray, mask: np.ndarray | None = None) -> Tuple[int, int]:
    if mask is not None and mask.any():
        med = np.median(img[mask])
    else:
        med = np.median(img)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    if lower == upper:
        upper = min(255, lower + 10)
    return lower, upper


def _solid_mask(
    ir: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    max_depth: float,
    depth_percentile: float,
    min_area: int,
    max_component_ratio: float,
    exclude_border: bool,
    border_margin: int,
) -> np.ndarray:
    def build_mask(
        depth_percentile_val: float,
        min_area_val: int,
        max_component_ratio_val: float,
        exclude_border_val: bool,
        border_margin_val: int,
    ) -> np.ndarray:
        depth_m = depth_raw.astype(np.float32) * depth_scale
        valid = (depth_m > 0) & (depth_m < max_depth)

        if not valid.any():
            return np.zeros_like(depth_raw, dtype=np.uint8)

        ir_blur = cv2.bilateralFilter(ir, d=5, sigmaColor=50, sigmaSpace=50)
        depth_norm = np.zeros_like(depth_m, dtype=np.float32)
        depth_norm[valid] = depth_m[valid] / max_depth * 255.0
        depth_u8 = depth_norm.astype(np.uint8)
        depth_blur = cv2.bilateralFilter(depth_u8, d=5, sigmaColor=50, sigmaSpace=50)

        t1, t2 = _auto_canny_thresholds(ir_blur, valid)
        d1, d2 = _auto_canny_thresholds(depth_blur, valid)

        depth_vals = depth_m[valid]
        depth_thresh = np.percentile(depth_vals, depth_percentile_val)
        foreground = valid & (depth_m <= depth_thresh)

        edges_ir = cv2.Canny(ir_blur, t1, t2)
        edges_depth = cv2.Canny(depth_blur, d1, d2)
        edges = cv2.bitwise_or(edges_ir, edges_depth)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        foreground_u8 = (foreground.astype(np.uint8) * 255)
        foreground_u8 = cv2.morphologyEx(
            foreground_u8, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        foreground_u8 = cv2.morphologyEx(
            foreground_u8, cv2.MORPH_OPEN, kernel, iterations=1
        )

        solid = cv2.bitwise_and(foreground_u8, cv2.bitwise_not(edges))
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
        mask_out = np.zeros_like(depth_raw, dtype=np.uint8)
        total_area = solid.shape[0] * solid.shape[1]

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area_val:
                continue
            if area / total_area > max_component_ratio_val:
                continue

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            if exclude_border_val:
                if x <= border_margin_val or y <= border_margin_val:
                    continue
                if x + w >= solid.shape[1] - border_margin_val:
                    continue
                if y + h >= solid.shape[0] - border_margin_val:
                    continue

            mask_out[labels == label] = 255

        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_OPEN, kernel, iterations=1)

        flood = mask_out.copy()
        h, w = flood.shape[:2]
        mask_flood = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask_flood, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        mask_out = cv2.bitwise_or(mask_out, flood_inv)

        return mask_out

    mask_out = build_mask(
        depth_percentile,
        min_area,
        max_component_ratio,
        exclude_border,
        border_margin,
    )

    if mask_out.sum() == 0:
        relaxed_percentile = min(95.0, depth_percentile + 15.0)
        relaxed_area = max(200, int(min_area * 0.6))
        mask_out = build_mask(
            relaxed_percentile,
            relaxed_area,
            max(0.9, max_component_ratio),
            False,
            border_margin,
        )

    return mask_out


def main() -> None:
    args = parse_args()
    ir_dir = Path(args.ir_dir)
    depth_dir = Path(args.depth_dir)
    out_dir = Path(args.output_dir)
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs(ir_dir, depth_dir)
    if not pairs:
        raise SystemExit("No se encontraron pares IR+Depth")

    if args.frame_step > 1:
        pairs = pairs[:: args.frame_step]

    summary = []

    for ir_path, depth_path in pairs:
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if ir is None or depth_raw is None:
            continue

        if depth_raw.dtype != np.uint16:
            raise SystemExit(f"Depth no es uint16: {depth_path}")

        if ir.shape[:2] != depth_raw.shape[:2]:
            ir = cv2.resize(ir, (depth_raw.shape[1], depth_raw.shape[0]), interpolation=cv2.INTER_AREA)

        mask = _solid_mask(
            ir=ir,
            depth_raw=depth_raw,
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            depth_percentile=args.depth_percentile,
            min_area=args.min_area,
            max_component_ratio=args.max_component_ratio,
            exclude_border=args.exclude_border,
            border_margin=args.border_margin,
        )

        overlay = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        overlay[mask > 0] = (0, 200, 0)
        overlay = cv2.addWeighted(cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR), 0.6, overlay, 0.4, 0)

        mask_path = masks_dir / f"{ir_path.stem}_mask.png"
        overlay_path = overlays_dir / f"{ir_path.stem}_overlay.png"
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(overlay_path), overlay)

        mask_ratio = float((mask > 0).sum()) / float(mask.size)
        summary.append((ir_path.name, mask_ratio))

    summary_path = out_dir / "segmentation_summary.txt"
    if summary:
        ratios = [r for _, r in summary]
        with open(summary_path, "w", encoding="utf-8") as file:
            file.write(f"frames: {len(summary)}\n")
            file.write(f"mask_ratio_mean: {float(np.mean(ratios))}\n")
            file.write(f"mask_ratio_min: {float(np.min(ratios))}\n")
            file.write(f"mask_ratio_max: {float(np.max(ratios))}\n")
    print("Masks:", masks_dir)
    print("Overlays:", overlays_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
