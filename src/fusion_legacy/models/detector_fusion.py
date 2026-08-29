from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
from loguru import logger


class DetectorFusion:
    """
    Fusión de detecciones de múltiples detectores.

    Combina detecciones de YOLO11x (rápido) y RT-DETRv2 (preciso)
    para obtener el mejor resultado de ambos.

    Estrategia:
        1. Usar RT-DET - MásR cuando:
           de min_objects objetos detectados
           - Alguna detección con confianza < low_confidence
           - Objetos pequeños detectados
        2. Fusionar resultados usando NMS
    """

    def __init__(
        self,
        nms_threshold: float = 0.5,
        min_objects_for_rtdetr: int = 10,
        low_confidence_threshold: float = 0.5,
    ):
        """
        Args:
            nms_threshold: Umbral NMS para fusionar detecciones
            min_objects_for_rtdetr: Usar RT-DETR si hay más de N objetos
            low_confidence_threshold: Usar RT-DETR si hay detecciones con conf < N
        """
        self.nms_threshold = nms_threshold
        self.min_objects_for_rtdetr = min_objects_for_rtdetr
        self.low_confidence_threshold = low_confidence_threshold

        logger.info(f"DetectorFusion inicializado: NMS={nms_threshold}")

    def should_use_rt_detr(
        self, yolo_detections: List, has_small_objects: bool = False
    ) -> bool:
        """
        Determina si se debe usar RT-DETR basándose en las detecciones de YOLO.

        Args:
            yolo_detections: Lista de detecciones de YOLO
            has_small_objects: Si hay objetos pequeños

        Returns:
            True si se debe usar RT-DETR
        """
        if not yolo_detections:
            return True

        num_objects = len(yolo_detections)
        min_conf = min(d.confidence for d in yolo_detections)

        # Condiciones para usar RT-DETR
        conditions = [
            num_objects >= self.min_objects_for_rtdetr,
            min_conf < self.low_confidence_threshold,
            has_small_objects,
        ]

        return any(conditions)

    def fuse_detections(self, detections_list: List[List]) -> List:
        """
        Fusiona detecciones de múltiples detectores usando NMS.

        Args:
            detections_list: Lista de listas de detecciones

        Returns:
            Lista fusionada de detecciones
        """
        # Combinar todas las detecciones
        all_detections = []
        for detections in detections_list:
            all_detections.extend(detections)

        if not all_detections:
            return []

        # Ordenar por confianza
        all_detections.sort(key=lambda d: d.confidence, reverse=True)

        # Aplicar NMS
        keep = []
        processed_classes = set()

        for detection in all_detections:
            # Para cada clase, aplicar NMS
            class_key = (detection.class_name, tuple(detection.bbox))

            if class_key in processed_classes:
                continue

            should_keep = True

            for kept_detection in keep:
                # Misma clase
                if detection.class_name == kept_detection.class_name:
                    # Calcular IoU
                    iou = self._box_iou(detection.bbox, kept_detection.bbox)

                    if iou > self.nms_threshold:
                        # Si el nuevo tiene mayor confianza, reemplazar
                        if detection.confidence > kept_detection.confidence:
                            keep.remove(kept_detection)
                        else:
                            should_keep = False
                        break

            if should_keep:
                keep.append(detection)
                processed_classes.add(class_key)

        # Ordenar por confianza final
        keep.sort(key=lambda d: d.confidence, reverse=True)

        # Reasignar IDs
        for i, detection in enumerate(keep):
            detection.id = i

        return keep

    def _box_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calcula IoU entre dos bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def merge_by_confidence(
        self, primary_detections: List, secondary_detections: List
    ) -> List:
        """
        Combina detecciones priorizando las de mayor confianza.

        Args:
            primary_detections: Detecciones principales (ej: YOLO)
            secondary_detections: Detecciones secundarias (ej: RT-DETR)

        Returns:
            Lista combinada
        """
        if not primary_detections:
            return secondary_detections

        if not secondary_detections:
            return primary_detections

        # Combinar y ordenar por confianza
        all_detections = primary_detections + secondary_detections
        all_detections.sort(key=lambda d: d.confidence, reverse=True)

        # Eliminar duplicados
        result = []
        seen = {}

        for detection in all_detections:
            class_name = detection.class_name
            bbox_key = tuple(int(x) for x in detection.bbox)
            key = (class_name, bbox_key)

            if key not in seen:
                seen[key] = detection
                result.append(detection)
            else:
                # Mantener el de mayor confianza
                if detection.confidence > seen[key].confidence:
                    seen[key] = detection

        result.sort(key=lambda d: d.confidence, reverse=True)

        # Reasignar IDs
        for i, detection in enumerate(result):
            detection.id = i

        return result

    def add_detection_id(self, detections: List) -> List:
        """Añade IDs secuenciales a las detecciones."""
        for i, detection in enumerate(detections):
            detection.id = i
        return detections

    def __repr__(self) -> str:
        return (
            f"DetectorFusion(NMS={self.nms_threshold}, "
            f"min_objects={self.min_objects_for_rtdetr})"
        )


# Re-exportar Detection
from .yolo_detector import Detection
