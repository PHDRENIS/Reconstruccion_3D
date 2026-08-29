from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger


class TeacherSegmentor:
    def __init__(
        self,
        world_model: str = "yolov8x-worldv2.pt",
        sam_model: str = "sam2_b.pt",
        confidence: float = 0.28,
        iou: float = 0.6,
        device: str = "cuda",
        use_box_fallback: bool = False,
    ):
        self.world_model_name = world_model
        self.sam_model_name = sam_model
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.use_box_fallback = use_box_fallback

        self.world_model = None
        self.sam_model = None
        self.available = False
        self._load_models()

    def _load_models(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            logger.warning(f"No se pudo importar ultralytics para teacher: {exc}")
            self.available = False
            return

        try:
            logger.info(f"Cargando teacher YOLO-World: {self.world_model_name}")
            self.world_model = YOLO(self.world_model_name)
            self.world_model.to(self.device)
        except Exception as exc:
            logger.warning(f"No se pudo cargar YOLO-World teacher: {exc}")
            self.world_model = None

        try:
            logger.info(f"Cargando teacher SAM: {self.sam_model_name}")
            self.sam_model = YOLO(self.sam_model_name)
            self.sam_model.to(self.device)
        except Exception as exc:
            logger.warning(f"No se pudo cargar SAM teacher: {exc}")
            self.sam_model = None

        self.available = self.world_model is not None and (
            self.sam_model is not None or self.use_box_fallback
        )
        if self.available:
            logger.info("Teacher auxiliar listo para pseudo-labels")
        else:
            logger.warning("Teacher auxiliar no disponible; se omitirá")

    @staticmethod
    def _xyxy_to_mask(image_shape: tuple, bbox: List[float]) -> np.ndarray:
        h, w = image_shape
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        mask = np.zeros((h, w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask

    def _run_sam_with_boxes(
        self, image: np.ndarray, bboxes: List[List[float]]
    ) -> Optional[List[np.ndarray]]:
        if self.sam_model is None or not bboxes:
            return None

        candidates = [
            lambda: self.sam_model(
                image,
                bboxes=bboxes,
                verbose=False,
                device=self.device,
            ),
            lambda: self.sam_model.predict(
                source=image,
                bboxes=bboxes,
                verbose=False,
                device=self.device,
            ),
            lambda: self.sam_model(image, bboxes=bboxes, verbose=False),
            lambda: self.sam_model.predict(source=image, bboxes=bboxes, verbose=False),
        ]

        sam_results = None
        for run in candidates:
            try:
                sam_results = run()
                if sam_results:
                    break
            except Exception:
                continue

        if not sam_results:
            return None

        result = sam_results[0]
        if result.masks is None or len(result.masks) == 0:
            return None

        h, w = image.shape[:2]
        masks = []
        mask_data = result.masks.data.cpu().numpy()
        for mask in mask_data:
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            masks.append((mask_resized > 0.5).astype(np.uint8))
        return masks

    def predict(
        self,
        image: np.ndarray,
        target_classes: List[str],
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Dict]:
        if not self.available or self.world_model is None:
            return []

        if not target_classes:
            return []

        confidence = self.confidence if confidence is None else confidence
        iou = self.iou if iou is None else iou

        try:
            if hasattr(self.world_model, "set_classes"):
                self.world_model.set_classes(target_classes)

            world_results = self.world_model(
                image,
                conf=confidence,
                iou=iou,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            logger.warning(f"Teacher YOLO-World falló: {exc}")
            return []

        if not world_results:
            return []

        result = world_results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        names = result.names if hasattr(result, "names") else {}
        bboxes = [box.tolist() for box in boxes]

        masks = self._run_sam_with_boxes(image, bboxes)

        detections = []
        h, w = image.shape[:2]

        for idx in range(len(bboxes)):
            class_id = int(class_ids[idx])
            class_name = str(names.get(class_id, f"class_{class_id}")).strip().lower()
            confidence_val = float(confidences[idx])

            if class_name not in target_classes:
                continue

            mask = None
            if masks is not None and idx < len(masks):
                mask = masks[idx]
            elif self.use_box_fallback:
                mask = self._xyxy_to_mask((h, w), bboxes[idx])

            if mask is None:
                continue

            detections.append(
                {
                    "class_name": class_name,
                    "confidence": confidence_val,
                    "bbox": bboxes[idx],
                    "mask": mask,
                    "source": "teacher",
                }
            )

        return detections
