from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import torch
from loguru import logger


class Detection:
    """Clase que representa una detección de objeto."""

    def __init__(
        self,
        bbox: List[float],  # [x1, y1, x2, y2]
        class_id: int,
        class_name: str,
        confidence: float,
        detector: str = "yolo11x",
    ):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.detector = detector

    def to_dict(self) -> Dict:
        """Convierte la detección a diccionario."""
        return {
            "bbox": self.bbox,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "detector": self.detector,
        }

    def __repr__(self) -> str:
        return f"Detection({self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


class YOLODetector:
    """
    Wrapper para YOLO11x - Detección rápida de objetos.

    Configuración:
        - Device: cuda (GPU) para velocidad
        - Confidence: 0.25 por defecto
        - IOU: 0.45 por defecto
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "cuda",
    ):
        """
        Args:
            model_path: Ruta al modelo YOLO (si es None, descarga el modelo yolo11x)
            confidence: Umbral de confianza
            iou: Umbral IOU para NMS
            device: Dispositivo para inferencia ("cuda" o "cpu")
        """
        self.confidence = confidence
        self.iou = iou
        self.device = device

        # Importar ultralytics lazily
        from ultralytics import YOLO

        # Cargar modelo
        if model_path and Path(model_path).exists():
            logger.info(f"Cargando modelo YOLO desde: {model_path}")
            self.model = YOLO(model_path)
        else:
            logger.info("Descargando modelo YOLO11x...")
            self.model = YOLO("yolo11x.pt")

        # Mover a dispositivo
        self.model.to(self.device)

        logger.info(f"YOLO11x inicializado en {device}")

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detecta objetos en una imagen.

        Args:
            image: Imagen RGB (H, W, 3)

        Returns:
            Lista de objetos Detection
        """
        # Ejecutar inferencia
        results = self.model(
            image, conf=self.confidence, iou=self.iou, verbose=False, device=self.device
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                # Obtener nombres de clases
                class_names = result.names

                for i in range(len(boxes)):
                    detection = Detection(
                        bbox=boxes[i].tolist(),
                        class_id=int(class_ids[i]),
                        class_name=class_names[int(class_ids[i])],
                        confidence=float(confidences[i]),
                        detector="yolo11x",
                    )
                    detections.append(detection)

        return detections

    def filter_by_classes(
        self, detections: List[Detection], classes: List[str]
    ) -> List[Detection]:
        """Filtra detecciones por nombres de clase."""
        classes_set = set(c.lower() for c in classes)
        return [d for d in detections if d.class_name.lower() in classes_set]

    def filter_by_confidence(
        self, detections: List[Detection], min_confidence: float
    ) -> List[Detection]:
        """Filtra detecciones por confianza mínima."""
        return [d for d in detections if d.confidence >= min_confidence]

    def get_detection_summary(self, detections: List[Detection]) -> Dict:
        """Genera un resumen de las detecciones."""
        if not detections:
            return {"count": 0, "classes": {}}

        classes_count = {}
        for d in detections:
            classes_count[d.class_name] = classes_count.get(d.class_name, 0) + 1

        return {
            "count": len(detections),
            "classes": classes_count,
            "avg_confidence": np.mean([d.confidence for d in detections]),
        }

    def __repr__(self) -> str:
        return f"YOLODetector(confidence={self.confidence}, iou={self.iou}, device={self.device})"
