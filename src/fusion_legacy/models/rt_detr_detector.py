from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import torch
from loguru import logger


class RTDETRDetector:
    """
    Wrapper para RT-DETRv2 - Detección de precisión con transformadores.

    Configuración:
        - Device: cpu (para RTX 3050 con 4GB VRAM)
        - Confidence: 0.3 por defecto
        - IOU: 0.5 por defecto

    Nota: RT-DETR es más preciso que YOLO para escenas complejas
    con múltiples objetos pequeños, pero más lento.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.3,
        iou: float = 0.5,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: Ruta al modelo RT-DETR (si es None, descarga el modelo)
            confidence: Umbral de confianza
            iou: Umbral IOU para NMS
            device: Dispositivo para inferencia
        """
        self.confidence = confidence
        self.iou = iou
        self.device = device

        # Importar transformers lazily
        from transformers import RTDetrv2ForObjectDetection, RTDetrv2ImageProcessor

        # Cargar modelo
        if model_path and Path(model_path).exists():
            logger.info(f"Cargando modelo RT-DETR desde: {model_path}")
            # Cargar desde archivo local
            self.processor = RTDetrv2ImageProcessor.from_pretrained(".")
            self.model = RTDetrv2ForObjectDetection.from_pretrained(".")
        else:
            logger.info("Descargando modelo RT-DETRv2...")
            # Usar el modelo base de HuggingFace
            self.processor = RTDetrv2ImageProcessor.from_pretrained(
                "PekingU/rtdetr_v2_base"
            )
            self.model = RTDetrv2ForObjectDetection.from_pretrained(
                "PekingU/rtdetr_v2_base"
            )

        # Mover a dispositivo
        self.model.to(self.device)
        self.model.eval()

        # Obtener lista de clases (COCO)
        self.class_labels = self._get_coco_labels()

        logger.info(f"RT-DETRv2 inicializado en {device}")

    def _get_coco_labels(self) -> Dict[int, str]:
        """Obtiene las etiquetas de clases COCO."""
        return {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }

    def detect(self, image: np.ndarray) -> List:
        """
        Detecta objetos en una imagen.

        Args:
            image: Imagen RGB (H, W, 3)

        Returns:
            Lista de objetos Detection
        """
        # Preparar imagen para el modelo
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inferencia
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-procesamiento
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[(image.shape[0], image.shape[1])],
            threshold=self.confidence,
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]

            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            boxes = result["boxes"].cpu().numpy()

            # Aplicar NMS manualmente si es necesario
            keep_indices = self._nms(boxes, scores, self.iou)

            for idx in keep_indices:
                label_id = int(labels[idx])
                class_name = self.class_labels.get(label_id, f"class_{label_id}")

                detection = Detection(
                    bbox=boxes[idx].tolist(),
                    class_id=label_id,
                    class_name=class_name,
                    confidence=float(scores[idx]),
                    detector="rt_detr",
                )
                detections.append(detection)

        return detections

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> List[int]:
        """Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []

        # Ordenar por score
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Calcular IOU
            iou = self._box_iou(boxes[i], boxes[order[1:]])

            # Mantener solo boxes con IOU < threshold
            mask = iou < iou_threshold
            order = order[1:][mask]

        return keep

    def _box_iou(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calcula IOU entre una caja y un array de cajas."""
        x1 = np.maximum(box1[0], boxes2[:, 0])
        y1 = np.maximum(box1[1], boxes2[:, 1])
        x2 = np.minimum(box1[2], boxes2[:, 2])
        y2 = np.minimum(box1[3], boxes2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def filter_by_classes(self, detections: List, classes: List[str]) -> List:
        """Filtra detecciones por nombres de clase."""
        classes_set = set(c.lower() for c in classes)
        return [d for d in detections if d.class_name.lower() in classes_set]

    def __repr__(self) -> str:
        return f"RTDETRDetector(confidence={self.confidence}, iou={self.iou}, device={self.device})"


# Re-exportar Detection
from .yolo_detector import Detection
