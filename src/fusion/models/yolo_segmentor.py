from __future__ import annotations
from typing import List, Dict
import numpy as np
import cv2
from pathlib import Path
from loguru import logger


class Detection:
    def __init__(
        self,
        bbox: List[float],
        class_id: int,
        class_name: str,
        confidence: float,
        detector: str = "yolo11x-sunrgbd",
    ):
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.detector = detector

    def to_dict(self) -> Dict:
        return {
            "bbox": self.bbox,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "detector": self.detector,
        }

    def __repr__(self) -> str:
        return f"Detection({self.class_name}, conf={self.confidence:.2f})"


class YOLOSegmentor:
    SUNRGBD_CLASSES = {
        0: "bed",
        1: "chair",
        2: "table",
        3: "sofa",
        4: "desk",
        5: "dresser",
        6: "bookshelf",
        7: "lamp",
        8: "pillow",
        9: "sink",
        10: "bathtub",
        11: "toilet",
        12: "box",
        13: "counter",
        14: "refrigerator",
        15: "tv",
        16: "curtain",
    }

    COCO_CLASSES = {
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

    def __init__(
        self,
        model_name: str = "yolo11x-sunrgbd-seg.pt",
        confidence: float = 0.18,
        iou: float = 0.6,
        device: str = "cuda",
        use_sunrgbd: bool = True,
    ):
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.use_sunrgbd = use_sunrgbd

        from ultralytics import YOLO

        if use_sunrgbd:
            base_path = Path(__file__).parent.parent.parent
            model_path = Path(model_name)
            if not model_path.is_absolute():
                model_path = base_path / "models" / model_name

            if model_path.exists():
                logger.info(f"Cargando modelo SUN-RGB-D: {model_path}")
                self.model = YOLO(str(model_path))
                self.class_labels = self.SUNRGBD_CLASSES
                self.model_type = "sunrgbd"
            else:
                logger.warning(f"Modelo SUN-RGB-D no encontrado, usando pre-entrenado")
                self.model = YOLO("yolo11x-seg.pt")
                self.class_labels = self.COCO_CLASSES
                self.model_type = "coco"
        else:
            logger.info(f"Cargando modelo: {model_name}")
            self.model = YOLO(model_name)
            self.class_labels = self.COCO_CLASSES
            self.model_type = "coco"

        self.model.to(self.device)
        logger.info(f"YOLO Segment inicializado en {device} (tipo: {self.model_type})")

    def detect_and_segment(self, image: np.ndarray) -> tuple:
        h, w = image.shape[:2]
        results = self.model(
            image, conf=self.confidence, iou=self.iou, verbose=False, device=self.device
        )

        detections = []
        masks = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                model_names = result.names if hasattr(result, "names") else {}

                for i in range(len(boxes)):
                    class_id = int(class_ids[i])
                    class_name = model_names.get(
                        class_id, self.class_labels.get(class_id, f"class_{class_id}")
                    )

                    detection = Detection(
                        bbox=boxes[i].tolist(),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confidences[i]),
                        detector=f"yolo11x-{self.model_type}",
                    )
                    detections.append(detection)

            if result.masks is not None and len(result.masks) > 0:
                mask_data = result.masks.data.cpu().numpy()
                for mask in mask_data:
                    mask_resized = cv2.resize(
                        mask.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    masks.append(mask_resized > 0.5)

        return detections, masks

    def filter_by_classes(
        self, detections: List[Detection], classes: List[str]
    ) -> List[Detection]:
        classes_set = set(c.lower() for c in classes)
        return [d for d in detections if d.class_name.lower() in classes_set]

    def __repr__(self) -> str:
        return f"YOLOSegmentor(type={self.model_type}, conf={self.confidence}, device={self.device})"
