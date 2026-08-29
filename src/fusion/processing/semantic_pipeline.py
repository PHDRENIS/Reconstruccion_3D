from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
import time
from pathlib import Path
from loguru import logger

from src.models.yolo_segmentor import YOLOSegmentor
from src.processing.semantic_filter import SemanticFilter


class SemanticResult:
    def __init__(
        self,
        image: np.ndarray,
        image_id: str,
        image_path: Path,
        detections: List,
        masks: List[np.ndarray],
        ignore_mask: np.ndarray,
        static_mask: np.ndarray,
        depth_gt: Optional[np.ndarray] = None,
        depth_input: Optional[np.ndarray] = None,
        processing_time: float = 0.0,
        detectors_used: List[str] = None,
    ):
        self.image = image
        self.image_id = image_id
        self.image_path = image_path
        self.detections = detections
        self.masks = masks
        self.ignore_mask = ignore_mask
        self.static_mask = static_mask
        self.depth_gt = depth_gt
        self.depth_input = depth_input
        self.processing_time = processing_time
        self.detectors_used = detectors_used or []

        self.static_masks = []
        self.dynamic_masks = []
        filter_inst = SemanticFilter()
        for i, d in enumerate(detections):
            if i < len(masks):
                if filter_inst.is_dynamic(d.class_name):
                    self.dynamic_masks.append(masks[i])
                else:
                    self.static_masks.append(masks[i])

    def to_dict(self) -> Dict:
        return {
            "image_name": self.image_path.name,
            "image_path": str(self.image_path),
            "image_id": self.image_id,
            "width": self.image.shape[1],
            "height": self.image.shape[0],
            "detectors_used": self.detectors_used,
            "detections": [
                {
                    "id": i,
                    "class_name": d.class_name,
                    "class_id": d.class_id,
                    "confidence": d.confidence,
                    "detector": d.detector,
                    "bbox": d.bbox,
                }
                for i, d in enumerate(self.detections)
            ],
            "static_count": len(self.static_masks),
            "dynamic_count": len(self.dynamic_masks),
            "processing_time": self.processing_time,
        }


class SemanticPipeline:
    def __init__(self, config: Dict, device_yolo: str = "cuda"):
        self.config = config
        yolo_cfg = config.get("models", {}).get("yolo", {})
        model_name = yolo_cfg.get("model_name", "yolo11x-sunrgbd-seg.pt")

        logger.info("Inicializando YOLO11 Segmentación...")
        self.yolo_seg = YOLOSegmentor(
            model_name=model_name,
            confidence=yolo_cfg.get("confidence_threshold", 0.18),
            iou=yolo_cfg.get("iou_threshold", 0.6),
            device=device_yolo,
            use_sunrgbd=True,
        )

        semantic_config = config.get("semantic", {})
        self.filter = SemanticFilter(
            dynamic_classes=semantic_config.get("dynamic_classes")
        )

        logger.info("Pipeline semántico inicializado")

    def process(
        self,
        image: np.ndarray,
        image_id: str,
        image_path: Path,
        depth_gt: Optional[np.ndarray] = None,
        depth_input: Optional[np.ndarray] = None,
    ) -> SemanticResult:
        start_time = time.time()
        detectors_used = ["yolo11x-sunrgbd-seg"]

        detections, masks = self.yolo_seg.detect_and_segment(image)

        for i, d in enumerate(detections):
            d.id = i

        h, w = image.shape[:2]
        ignore_mask = self.filter.create_ignore_mask((h, w), detections, masks)
        static_mask = self.filter.get_static_mask((h, w), detections, masks)

        processing_time = time.time() - start_time

        return SemanticResult(
            image=image,
            image_id=image_id,
            image_path=image_path,
            detections=detections,
            masks=masks,
            ignore_mask=ignore_mask,
            static_mask=static_mask,
            depth_gt=depth_gt,
            depth_input=depth_input,
            processing_time=processing_time,
            detectors_used=detectors_used,
        )

    def __repr__(self) -> str:
        return f"SemanticPipeline(yolo_seg={self.yolo_seg})"
