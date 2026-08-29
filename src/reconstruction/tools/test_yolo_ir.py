#!/usr/bin/env python3
"""
Run YOLO segmentation on an IR image and export overlays/masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test YOLO model on IR image")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLO segmentation model (.pt)",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to IR image",
    )
    parser.add_argument(
        "--out-dir",
        default="output/tests",
        help="Output directory for results",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"No se pudo leer la imagen: {args.image}")

    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model(
        image_bgr,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    if not results:
        raise SystemExit("No se obtuvieron resultados del modelo")

    result = results[0]
    overlay = result.plot()
    overlay_path = out_dir / "yolo_ir_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    summary = {
        "model": str(Path(args.model).resolve()),
        "image": str(Path(args.image).resolve()),
        "conf": args.conf,
        "iou": args.iou,
        "detections": [],
        "mask_count": 0,
        "overlay": str(overlay_path),
    }

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        names = result.names if hasattr(result, "names") else {}
        for idx in range(len(boxes)):
            class_id = int(class_ids[idx])
            summary["detections"].append(
                {
                    "class_id": class_id,
                    "class_name": str(names.get(class_id, class_id)),
                    "confidence": float(confidences[idx]),
                    "bbox": [float(v) for v in boxes[idx].tolist()],
                }
            )

    if result.masks is not None and len(result.masks) > 0:
        mask_data = result.masks.data.cpu().numpy()
        summary["mask_count"] = int(mask_data.shape[0])

        combined = (mask_data.max(axis=0) > 0.5).astype(np.uint8) * 255
        combined_path = out_dir / "yolo_ir_mask_combined.png"
        cv2.imwrite(str(combined_path), combined)
        summary["mask_combined"] = str(combined_path)

    summary_path = out_dir / "yolo_ir_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Overlay:", overlay_path)
    print("Summary:", summary_path)
    print("Detections:", len(summary["detections"]))


if __name__ == "__main__":
    main()
