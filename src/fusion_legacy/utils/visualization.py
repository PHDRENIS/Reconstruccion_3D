from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict
import json
from loguru import logger

from src.models.yolo_detector import Detection
from src.processing.semantic_pipeline import SemanticResult


class Visualizer:
    """Utilidades de visualización y guardado de resultados."""

    CLASS_COLORS = {
        "person": (255, 0, 0),
        "chair": (0, 255, 0),
        "table": (0, 0, 255),
        "couch": (255, 255, 0),
        "bed": (255, 0, 255),
        "tv": (0, 255, 255),
    }

    def __init__(
        self,
        output_dir: str,
        save_overlays: bool = True,
        save_masks: bool = True,
        save_json: bool = True,
        save_depth: bool = True,
        overlay_alpha: float = 0.4,
        overlay_thickness: int = 2,
    ):
        self.output_dir = Path(output_dir)
        self.save_overlays = save_overlays
        self.save_masks = save_masks
        self.save_json = save_json
        self.save_depth = save_depth
        self.overlay_alpha = overlay_alpha
        self.overlay_thickness = overlay_thickness

        self._create_directories()
        logger.info(f"Visualizer inicializado: {output_dir}")

    def _create_directories(self) -> None:
        dirs = [
            "overlays/train",
            "overlays/validation",
            "masks/train",
            "masks/validation",
            "ignore_masks/train",
            "ignore_masks/validation",
            "detections/train",
            "detections/validation",
            "depth/train",
            "depth/validation",
        ]
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)

    def _get_color(self, class_name: str) -> tuple:
        if class_name in self.CLASS_COLORS:
            return self.CLASS_COLORS[class_name]
        hash_val = hash(class_name)
        return ((hash_val >> 16) % 255, (hash_val >> 8) % 255, hash_val % 255)

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        masks: List[np.ndarray] = None,
    ) -> np.ndarray:
        result = image.copy()
        h, w = result.shape[:2]
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        if masks is not None:
            for i, mask in enumerate(masks):
                if i < len(detections):
                    color = self._get_color(detections[i].class_name)
                    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                    mask_overlay[mask] = color
                    result_bgr = cv2.addWeighted(
                        result_bgr, 1, mask_overlay, self.overlay_alpha, 0
                    )

        for detection in detections:
            bbox = [int(x) for x in detection.bbox]
            color = self._get_color(detection.class_name)

            cv2.rectangle(
                result_bgr,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                self.overlay_thickness,
            )

            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(
                result_bgr,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    def save_overlay(self, result: SemanticResult, split: str = "train") -> str:
        if not self.save_overlays:
            return ""

        overlay = self.draw_detections(result.image, result.detections, result.masks)
        output_path = (
            self.output_dir / "overlays" / split / f"{result.image_id}_overlay.jpg"
        )
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return str(output_path)

    def save_masks(self, result: SemanticResult, split: str = "train") -> List[str]:
        if not self.save_masks or not result.masks:
            return []

        saved_paths = []
        for i, (detection, mask) in enumerate(zip(result.detections, result.masks)):
            class_name = detection.class_name
            class_dir = self.output_dir / "masks" / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            mask_path = class_dir / f"{result.image_id}_{i:02d}.png"
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_img)
            saved_paths.append(str(mask_path))

        return saved_paths

    def save_ignore_mask(self, result: SemanticResult, split: str = "train") -> str:
        if not self.save_masks:
            return ""

        output_path = (
            self.output_dir / "ignore_masks" / split / f"{result.image_id}_ignore.png"
        )
        cv2.imwrite(str(output_path), result.ignore_mask * 255)
        return str(output_path)

    def save_json(self, result: SemanticResult, split: str = "train") -> str:
        if not self.save_json:
            return ""

        output_path = self.output_dir / "detections" / split / f"{result.image_id}.json"

        data = {
            "image_name": result.image_path.name,
            "image_path": str(result.image_path),
            "depth_gt_path": str(result.image_path).replace("rgb", "depth_gt")
            if result.depth_gt is not None
            else None,
            "depth_input_path": str(result.image_path).replace("rgb", "depth_input")
            if result.depth_input is not None
            else None,
            "width": result.image.shape[1],
            "height": result.image.shape[0],
            "detectors_used": result.detectors_used,
            "detections": [],
            "static_count": len(result.static_masks),
            "dynamic_count": len(result.dynamic_masks),
            "ignore_mask_path": str(
                self.output_dir
                / "ignore_masks"
                / split
                / f"{result.image_id}_ignore.png"
            ),
            "processing_time": result.processing_time,
        }

        for i, detection in enumerate(result.detections):
            detection_data = {
                "id": i,
                "class_name": detection.class_name,
                "class_id": detection.class_id,
                "confidence": detection.confidence,
                "detector": detection.detector,
                "bbox": detection.bbox,
                "is_dynamic": detection.class_name
                in [
                    "person",
                    "dog",
                    "cat",
                    "bird",
                    "horse",
                    "car",
                    "bicycle",
                    "motorcycle",
                    "airplane",
                    "bus",
                    "train",
                    "truck",
                ],
            }

            if i < len(result.masks):
                detection_data["mask_path"] = str(
                    self.output_dir
                    / "masks"
                    / split
                    / detection.class_name
                    / f"{result.image_id}_{i:02d}.png"
                )

            data["detections"].append(detection_data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_path)

    def save_depth(
        self, result: SemanticResult, split: str = "train"
    ) -> Dict[str, str]:
        saved = {}

        if result.depth_gt is not None and self.save_depth:
            output_path = (
                self.output_dir / "depth" / split / f"{result.image_id}_gt.png"
            )
            cv2.imwrite(str(output_path), result.depth_gt)
            saved["depth_gt"] = str(output_path)

        if result.depth_input is not None and self.save_depth:
            output_path = (
                self.output_dir / "depth" / split / f"{result.image_id}_input.png"
            )
            cv2.imwrite(str(output_path), result.depth_input)
            saved["depth_input"] = str(output_path)

        return saved

    def save_result(
        self, result: SemanticResult, split: str = "train"
    ) -> Dict[str, str]:
        saved = {}

        overlay_path = self.save_overlay(result, split)
        if overlay_path:
            saved["overlay"] = overlay_path

        mask_paths = self.save_masks(result, split)
        if mask_paths:
            saved["masks"] = mask_paths

        ignore_path = self.save_ignore_mask(result, split)
        if ignore_path:
            saved["ignore_mask"] = ignore_path

        json_path = self.save_json(result, split)
        if json_path:
            saved["json"] = json_path

        depth_paths = self.save_depth(result, split)
        saved.update(depth_paths)

        return saved
