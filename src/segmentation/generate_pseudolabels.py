#!/usr/bin/env python3
"""
Genera pseudo-labels YOLO-seg a partir de IR usando FastSAM + Depth.
Refinamiento: filtra fondo por profundidad, elimina regiones invalidas,
aplica morfologia y convierte a poligonos normalizados YOLO.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


def generate_pseudolabels(
    ir_dir: str,
    depth_dir: str,
    output_dir: str,
    depth_scale: float = 0.001,
    max_depth: float = 8.0,
    depth_percentile: float = 80.0,
    min_area: int = 500,
    max_component_ratio: float = 0.7,
    conf: float = 0.3,
) -> int:
    ir_dir = Path(ir_dir)
    depth_dir = Path(depth_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_files = sorted(ir_dir.glob("*.jpg")) + sorted(ir_dir.glob("*.png"))
    depth_files = {p.stem: p for p in depth_dir.glob("*.png")}

    if not ir_files:
        logger.error(f"No se encontraron imagenes IR en {ir_dir}")
        return 0

    from ultralytics import FastSAM

    model = FastSAM("FastSAM-s.pt")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    total = len(ir_files)
    generated = 0

    for ir_path in tqdm(ir_files, desc="FastSAM pseudo-labels"):
        depth_path = depth_files.get(ir_path.stem)
        if depth_path is None:
            logger.warning(f"Sin depth para {ir_path.name}")
            continue

        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if ir is None or depth_raw is None:
            continue

        h, w = ir.shape[:2]
        depth_raw = np.squeeze(depth_raw)
        depth_m = depth_raw.astype(np.float32) * depth_scale
        valid = (depth_m > 0) & (depth_m < max_depth)

        if not valid.any():
            continue

        depth_thresh = np.percentile(depth_m[valid], depth_percentile)

        ir3 = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        results = model(ir3, conf=conf, verbose=False)
        if not results or not results[0].masks:
            continue

        result = results[0]
        masks = result.masks.data.cpu().numpy()
        label_lines = []

        for idx in range(len(masks)):
            m = (masks[idx] > 0.5).astype(np.uint8) * 255
            area = int(m.sum() / 255)
            if area < min_area:
                continue
            if area / (h * w) > max_component_ratio:
                continue

            m[~valid] = 0
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_area = cv2.contourArea(cnt)
                if cnt_area < min_area or len(cnt) < 6:
                    continue

                mean_depth = depth_m[m > 0].mean() if m.sum() > 0 else max_depth
                if mean_depth > depth_thresh:
                    continue

                poly = []
                for pt in cnt:
                    x_s, y_s = pt[0]
                    poly.append(f"{x_s / w:.6f}")
                    poly.append(f"{y_s / h:.6f}")
                label_lines.append("0 " + " ".join(poly))

        if not label_lines:
            continue

        label_path = out_dir / f"{ir_path.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))
        generated += 1

    logger.info(f"Pseudo-labels generadas: {generated}/{total} en {out_dir}")
    return generated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FastSAM + Depth pseudo-labels")
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--depth-percentile", type=float, default=80.0)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")

    generate_pseudolabels(**vars(args))
