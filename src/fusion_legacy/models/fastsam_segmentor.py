from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import torch
import cv2
from loguru import logger


class Mask:
    """Clase que representa una máscara de segmentación."""

    def __init__(
        self,
        segmentation: np.ndarray,  # (H, W) bool
        area: int,
        bbox: List[float],  # [x1, y1, x2, y2]
        confidence: float = 1.0,
    ):
        self.segmentation = segmentation
        self.area = area
        self.bbox = bbox
        self.confidence = confidence

    def to_dict(self) -> Dict:
        """Convierte la máscara a diccionario."""
        return {"area": self.area, "bbox": self.bbox, "confidence": self.confidence}

    def __repr__(self) -> str:
        return f"Mask(area={self.area}, bbox={self.bbox})"


class FastSAMSegmentor:
    """
    Wrapper para FastSAM - Segmentación de instancias.

    Configuración:
        - Modelo: FastSAM-x (más preciso, más pesado)
        - Device: cuda (GPU) o cpu

    Uso:
        - Segmenta objetos usando bounding boxes de YOLO/RT-DETR como prompt
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        iou_threshold: float = 0.7,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            model_path: Ruta al modelo FastSAM (si es None, descarga el modelo)
            device: Dispositivo para inferencia
            iou_threshold: Umbral IOU para NMS de máscaras
            confidence_threshold: Umbral de confianza
        """
        self.device = device
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        # Importar sam o segment-anything
        try:
            from sam import Sam

            self.use_new_sam = True
            logger.info("Usando paquete 'sam'")
        except ImportError:
            try:
                from segment_anything import sam_model_registry, SamPredictor

                self.use_new_sam = False
                logger.info("Usando paquete 'segment-anything'")
            except ImportError:
                raise ImportError(
                    "No se pudo importar 'sam' ni 'segment-anything'. "
                    "Por favor instale uno de ellos: pip install sam"
                )

        # Cargar modelo
        if model_path and Path(model_path).exists():
            logger.info(f"Cargando modelo FastSAM desde: {model_path}")
            if self.use_new_sam:
                self.sam = Sam(model_path)
            else:
                self.sam = sam_model_registry["x"](checkpoint=model_path)
        else:
            logger.info("Descargando modelo FastSAM-x...")
            if self.use_new_sam:
                # Para el nuevo sam, descargar desde URL
                import urllib.request
                import os

                model_url = (
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_x.pth"
                )
                cache_dir = Path.home() / ".cache" / "sam"
                cache_dir.mkdir(parents=True, exist_ok=True)
                model_file = cache_dir / "sam_vit_x.pth"

                if not model_file.exists():
                    logger.info(
                        "Descargando modelo FastSAM-x (esto puede tomar un tiempo)..."
                    )
                    urllib.request.urlretrieve(model_url, model_file)

                self.sam = Sam(str(model_file))
            else:
                self.sam = sam_model_registry["x"](checkpoint=None)

        # Mover a dispositivo
        self.sam.to(self.device)
        self.sam.eval()

        # Crear predictor
        if self.use_new_sam:
            self.predictor = None  # El nuevo sam usa diferente API
        else:
            self.predictor = SamPredictor(self.sam)

        logger.info(f"FastSAM-x inicializado en {device}")

    def segment_from_boxes(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
        labels: Optional[List[str]] = None,
    ) -> List[Mask]:
        """
        Segmenta objetos a partir de bounding boxes.

        Args:
            image: Imagen RGB (H, W, 3)
            boxes: Lista de bounding boxes [[x1, y1, x2, y2], ...]
            labels: Lista opcional de etiquetas para cada box

        Returns:
            Lista de objetos Mask
        """
        if not boxes:
            return []

        # Convertir boxes al formato de SAM
        input_boxes = torch.tensor(boxes, device=self.device)

        # Transformar coordenadas
        h, w = image.shape[:2]
        input_boxes = input_boxes.clone()
        input_boxes[:, [0, 2]] = input_boxes[:, [0, 2]].clamp(0, w)
        input_boxes[:, [1, 3]] = input_boxes[:, [1, 3]].clamp(0, h)

        # Inferencia en modo batch
        with torch.no_grad():
            # Obtener embeddings de imagen
            self.predictor.set_image(image)

            # Segmentar cada box
            masks = []
            for i in range(len(input_boxes)):
                box = input_boxes[i]

                # Convertir box a formato [x, y, w, h]
                xyxy = box.cpu().numpy()
                x, y, x2, y2 = xyxy
                box_xywh = [x, y, x2 - x, y2 - y]

                # Obtener máscara
                mask, score, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_xywh,
                    multimask_output=False,
                )

                # Procesar máscara
                mask = mask[0]  # (H, W)
                area = int(mask.sum())

                # Calcular bbox de la máscara
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    mask_bbox = [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max()),
                        float(ys.max()),
                    ]
                else:
                    mask_bbox = box_xywh[:2] + [
                        box_xywh[0] + box_xywh[2],
                        box_xywh[1] + box_xywh[3],
                    ]

                masks.append(
                    Mask(
                        segmentation=mask,
                        area=area,
                        bbox=mask_bbox,
                        confidence=float(score[0])
                        if isinstance(score, np.ndarray)
                        else float(score),
                    )
                )

        # Aplicar NMS a las máscaras
        masks = self._filter_masks(masks)

        return masks

    def segment_from_points(
        self, image: np.ndarray, points: List[Tuple[int, int]], point_labels: List[int]
    ) -> List[Mask]:
        """
        Segmenta objetos a partir de puntos.

        Args:
            image: Imagen RGB (H, W, 3)
            points: Lista de puntos [(x, y), ...]
            point_labels: Lista de etiquetas (1=foreground, 0=background)

        Returns:
            Lista de objetos Mask
        """
        if not points:
            return []

        # Convertir a tensores
        input_points = np.array(points)
        input_labels = np.array(point_labels)

        with torch.no_grad():
            self.predictor.set_image(image)

            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

        result_masks = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            ys, xs = np.where(mask)
            area = int(mask.sum())

            if len(ys) > 0:
                bbox = [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ]
            else:
                bbox = [0, 0, image.shape[1], image.shape[0]]

            result_masks.append(
                Mask(segmentation=mask, area=area, bbox=bbox, confidence=float(score))
            )

        return result_masks

    def _filter_masks(self, masks: List[Mask]) -> List[Mask]:
        """Filtra máscaras usando NMS."""
        if len(masks) <= 1:
            return masks

        # Ordenar por área (de mayor a menor)
        sorted_masks = sorted(masks, key=lambda m: m.area, reverse=True)

        keep = []
        for i, mask_i in enumerate(sorted_masks):
            should_keep = True
            for mask_j in keep:
                # Calcular IoU
                iou = self._mask_iou(mask_i.segmentation, mask_j.segmentation)
                if iou > self.iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                keep.append(mask_i)

        return keep

    def _mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calcula IoU entre dos máscaras."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return float(intersection) / float(union)

    def segment_everything(self, image: np.ndarray) -> List[Mask]:
        """
        Segmenta todo en la imagen (modo automático).

        Args:
            image: Imagen RGB (H, W, 3)

        Returns:
            Lista de máscaras
        """
        from segment_anything import SamAutomaticMaskGenerator

        # Crear generador automático
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            pred_iou_thresh=self.confidence_threshold,
            stability_score_thresh=0.5,
        )

        # Generar máscaras
        with torch.no_grad():
            masks_data = mask_generator.generate(image)

        masks = []
        for mask_data in masks_data:
            masks.append(
                Mask(
                    segmentation=mask_data["segmentation"],
                    area=int(mask_data["area"]),
                    bbox=mask_data["bbox"],
                    confidence=mask_data["predicted_iou"],
                )
            )

        return masks

    def __repr__(self) -> str:
        return f"FastSAMSegmentor(device={self.device})"
