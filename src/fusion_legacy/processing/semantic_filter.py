from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import cv2
from loguru import logger

from src.models.yolo_detector import Detection


class SemanticFilter:
    """Filtro de objetos dinámicos."""

    DEFAULT_DYNAMIC_CLASSES = {
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
    }

    def __init__(self, dynamic_classes: Optional[List[str]] = None):
        if dynamic_classes:
            self.dynamic_classes = set(c.lower() for c in dynamic_classes)
        else:
            self.dynamic_classes = self.DEFAULT_DYNAMIC_CLASSES.copy()

        logger.info(f"SemanticFilter: {len(self.dynamic_classes)} clases dinámicas")

    def is_dynamic(self, class_name: str) -> bool:
        return class_name.lower() in self.dynamic_classes

    def filter_dynamic(self, detections: List[Detection]) -> tuple:
        static, dynamic = [], []
        for detection in detections:
            if self.is_dynamic(detection.class_name):
                dynamic.append(detection)
            else:
                static.append(detection)
        return static, dynamic

    def create_ignore_mask(
        self,
        image_shape: tuple,
        detections: List[Detection],
        masks: List[np.ndarray] = None,
    ) -> np.ndarray:
        h, w = image_shape
        ignore_mask = np.zeros((h, w), dtype=np.uint8)

        for i, detection in enumerate(detections):
            if self.is_dynamic(detection.class_name):
                if masks is not None and i < len(masks):
                    ignore_mask = np.logical_or(ignore_mask, masks[i])
                else:
                    bbox = [int(x) for x in detection.bbox]
                    x1, y1, x2, y2 = bbox
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    ignore_mask[y1:y2, x1:x2] = 1

        return ignore_mask.astype(np.uint8)

    def get_static_mask(
        self,
        image_shape: tuple,
        detections: List[Detection],
        masks: List[np.ndarray] = None,
    ) -> np.ndarray:
        h, w = image_shape
        static_mask = np.zeros((h, w), dtype=np.uint8)

        for i, detection in enumerate(detections):
            if not self.is_dynamic(detection.class_name):
                if masks is not None and i < len(masks):
                    static_mask = np.logical_or(static_mask, masks[i])
                else:
                    bbox = [int(x) for x in detection.bbox]
                    x1, y1, x2, y2 = bbox
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    static_mask[y1:y2, x1:x2] = 1

        return static_mask.astype(np.uint8)

    def get_class_summary(self, detections: List[Detection]) -> Dict:
        classes = {}
        static_count = dynamic_count = 0

        for detection in detections:
            classes[detection.class_name] = classes.get(detection.class_name, 0) + 1
            if self.is_dynamic(detection.class_name):
                dynamic_count += 1
            else:
                static_count += 1

        return {
            "total": len(detections),
            "static": static_count,
            "dynamic": dynamic_count,
            "classes": classes,
        }
