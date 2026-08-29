#!/usr/bin/env python3
"""
Script para preparar datos de SUN-RGB-D usando pseudo-labels:
- YOLO11x-seg para clases mapeables COCO->SUNRGBD
- Teacher auxiliar (YOLO-World + SAM) para clases sin cobertura COCO
"""

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm
import shutil

from src.models.teacher_segmentor import TeacherSegmentor


class PseudoLabelGenerator:
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

    SUNRGBD_NAME_TO_ID = {name: class_id for class_id, name in SUNRGBD_CLASSES.items()}

    # Mapeo correcto COCO(80) -> SUNRGBD
    COCO_TO_SUNRGBD = {
        59: 0,  # bed
        56: 1,  # chair
        60: 2,  # dining table -> table
        57: 3,  # couch -> sofa
        71: 9,  # sink
        61: 11,  # toilet
        72: 14,  # refrigerator
        62: 15,  # tv
    }

    COCO_COVERED_CLASSES = {
        SUNRGBD_CLASSES[sunrgbd_id] for sunrgbd_id in COCO_TO_SUNRGBD.values()
    }

    TEACHER_CLASS_ALIASES = {
        "bed": ["bed"],
        "chair": ["chair"],
        "table": ["table", "dining table"],
        "sofa": ["sofa", "couch"],
        "desk": ["desk", "office desk", "writing desk"],
        "dresser": ["dresser", "chest of drawers", "drawer cabinet"],
        "bookshelf": ["bookshelf", "book shelf", "shelf"],
        "lamp": ["lamp", "table lamp", "floor lamp"],
        "pillow": ["pillow", "cushion"],
        "sink": ["sink"],
        "bathtub": ["bathtub", "bath tub"],
        "toilet": ["toilet"],
        "box": ["box", "cardboard box", "storage box"],
        "counter": ["counter", "countertop", "kitchen counter"],
        "refrigerator": ["refrigerator", "fridge"],
        "tv": ["tv", "television", "monitor"],
        "curtain": ["curtain", "drape", "window curtain"],
    }

    def __init__(self, data_root: str, output_root: str, config: Dict):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.config = config

        self.train_images = self.output_root / "data" / "sunrgbd" / "images" / "train"
        self.val_images = self.output_root / "data" / "sunrgbd" / "images" / "val"
        self.train_labels = self.output_root / "data" / "sunrgbd" / "labels" / "train"
        self.val_labels = self.output_root / "data" / "sunrgbd" / "labels" / "val"

        for directory in [
            self.train_images,
            self.val_images,
            self.train_labels,
            self.val_labels,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.yolo = None
        self.teacher: Optional[TeacherSegmentor] = None

        pseudo_cfg = config.get("PSEUDO_LABELS", {})
        quality_cfg = pseudo_cfg.get("quality", {})
        teacher_cfg = pseudo_cfg.get("teacher", {})

        self.yolo_confidence = float(pseudo_cfg.get("yolo_confidence", 0.35))
        self.yolo_iou = float(pseudo_cfg.get("yolo_iou", 0.6))
        self.val_stride = int(pseudo_cfg.get("val_stride", 10))

        self.min_mask_pixels = int(quality_cfg.get("min_mask_pixels", 140))
        self.min_polygon_points = int(quality_cfg.get("min_polygon_points", 6))
        self.min_polygon_area = float(quality_cfg.get("min_polygon_area", 90.0))
        self.dedupe_mask_iou = float(quality_cfg.get("dedupe_mask_iou", 0.85))

        self.teacher_enabled = bool(teacher_cfg.get("enabled", True))
        self.teacher_world_model = str(
            teacher_cfg.get("world_model", "yolov8x-worldv2.pt")
        )
        self.teacher_sam_model = str(teacher_cfg.get("sam_model", "sam2_b.pt"))
        self.teacher_confidence = float(teacher_cfg.get("confidence", 0.28))
        self.teacher_iou = float(teacher_cfg.get("iou", 0.6))
        self.teacher_device = str(teacher_cfg.get("device", "cuda"))
        self.teacher_use_box_fallback = bool(teacher_cfg.get("use_box_fallback", False))
        self.teacher_target_classes = [
            str(c).strip().lower()
            for c in teacher_cfg.get("target_classes", [])
            if str(c).strip()
        ]

        all_class_names = set(self.SUNRGBD_NAME_TO_ID.keys())
        if self.teacher_target_classes:
            self.teacher_target_classes = [
                c for c in self.teacher_target_classes if c in all_class_names
            ]
        else:
            self.teacher_target_classes = sorted(
                list(all_class_names - self.COCO_COVERED_CLASSES)
            )

        self.teacher_queries = self._build_teacher_queries(self.teacher_target_classes)

        self.class_counts = Counter()
        self.class_counts_source = Counter()
        self.skipped_reasons = Counter()

    def _build_teacher_queries(self, target_classes: List[str]) -> List[str]:
        query_terms = []
        for class_name in target_classes:
            aliases = self.TEACHER_CLASS_ALIASES.get(class_name, [class_name])
            query_terms.extend(
                alias.strip().lower() for alias in aliases if alias.strip()
            )
        return sorted(set(query_terms))

    def _normalize_teacher_class_name(self, raw_name: str) -> Optional[str]:
        label = raw_name.strip().lower()
        for canonical, aliases in self.TEACHER_CLASS_ALIASES.items():
            alias_set = {a.strip().lower() for a in aliases}
            if label == canonical or label in alias_set:
                return canonical
        if label in self.SUNRGBD_NAME_TO_ID:
            return label
        return None

    def load_yolo(self, device: str = "cuda") -> None:
        from ultralytics import YOLO

        logger.info("Cargando YOLO11x-seg pre-entrenado para pseudo-labeling...")
        self.yolo = YOLO("yolo11x-seg.pt")
        self.yolo.to(device)

    def load_teacher(self) -> None:
        if not self.teacher_enabled:
            logger.info("Teacher auxiliar deshabilitado por configuración")
            return

        if not self.teacher_queries:
            logger.info("No hay clases objetivo para teacher auxiliar")
            return

        self.teacher = TeacherSegmentor(
            world_model=self.teacher_world_model,
            sam_model=self.teacher_sam_model,
            confidence=self.teacher_confidence,
            iou=self.teacher_iou,
            device=self.teacher_device,
            use_box_fallback=self.teacher_use_box_fallback,
        )
        logger.info(f"Teacher queries activas: {', '.join(self.teacher_queries)}")

    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(inter) / float(union)

    def _is_mask_valid(self, mask: Optional[np.ndarray]) -> bool:
        if mask is None:
            self.skipped_reasons["missing_mask"] += 1
            return False

        if int(mask.sum()) < self.min_mask_pixels:
            self.skipped_reasons["small_mask"] += 1
            return False

        return True

    def _extract_polygons(
        self,
        mask: np.ndarray,
        width: int,
        height: int,
    ) -> List[str]:
        polygons = []
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < self.min_polygon_points:
                self.skipped_reasons["short_contour"] += 1
                continue

            area = cv2.contourArea(contour)
            if area < self.min_polygon_area:
                self.skipped_reasons["small_contour_area"] += 1
                continue

            coords = []
            for point in contour:
                x, y = point[0]
                coords.append(f"{x / width:.6f}")
                coords.append(f"{y / height:.6f}")

            if len(coords) >= self.min_polygon_points * 2:
                polygons.append(" ".join(coords))

        return polygons

    def _merge_detections(self, detections: List[Dict]) -> List[Dict]:
        merged = []

        for det in sorted(
            detections, key=lambda item: item["confidence"], reverse=True
        ):
            class_id = det["class_id"]
            mask = det["mask"]
            keep = True

            for kept in merged:
                if kept["class_id"] != class_id:
                    continue
                iou = self._mask_iou(mask, kept["mask"])
                if iou >= self.dedupe_mask_iou:
                    keep = False
                    self.skipped_reasons["dedupe_iou"] += 1
                    break

            if keep:
                merged.append(det)

        return merged

    def _collect_yolo_detections(self, image: np.ndarray) -> List[Dict]:
        if self.yolo is None:
            return []

        h, w = image.shape[:2]
        results = self.yolo(
            image,
            conf=self.yolo_confidence,
            iou=self.yolo_iou,
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []

        for idx in range(len(boxes)):
            coco_id = int(class_ids[idx])
            if coco_id not in self.COCO_TO_SUNRGBD:
                self.skipped_reasons["unmapped_coco_class"] += 1
                continue

            sunrgbd_id = self.COCO_TO_SUNRGBD[coco_id]
            confidence = float(confidences[idx])

            mask = None
            if idx < len(masks):
                mask_resized = cv2.resize(
                    masks[idx].astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = (mask_resized > 0.5).astype(np.uint8)

            if not self._is_mask_valid(mask):
                continue

            detections.append(
                {
                    "class_id": sunrgbd_id,
                    "class_name": self.SUNRGBD_CLASSES[sunrgbd_id],
                    "confidence": confidence,
                    "bbox": boxes[idx].tolist(),
                    "mask": mask,
                    "source": "yolo",
                }
            )

        return detections

    def _collect_teacher_detections(self, image: np.ndarray) -> List[Dict]:
        if self.teacher is None or not self.teacher.available:
            return []

        teacher_raw = self.teacher.predict(
            image=image,
            target_classes=self.teacher_queries,
            confidence=self.teacher_confidence,
            iou=self.teacher_iou,
        )

        detections = []
        for det in teacher_raw:
            class_name = self._normalize_teacher_class_name(
                str(det.get("class_name", ""))
            )
            if class_name is None or class_name not in self.SUNRGBD_NAME_TO_ID:
                self.skipped_reasons["teacher_unknown_class"] += 1
                continue

            mask = det.get("mask")
            if not self._is_mask_valid(mask):
                continue

            class_id = self.SUNRGBD_NAME_TO_ID[class_name]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(det.get("confidence", 0.0)),
                    "bbox": det.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                    "mask": mask.astype(np.uint8),
                    "source": "teacher",
                }
            )

        return detections

    def create_yolo_labels(
        self, image_shape: tuple, detections: List[Dict]
    ) -> List[str]:
        h, w = image_shape
        lines = []

        for det in detections:
            class_id = det["class_id"]
            class_name = det["class_name"]
            polygons = self._extract_polygons(det["mask"], w, h)

            if not polygons:
                self.skipped_reasons["no_valid_polygons"] += 1
                continue

            for poly in polygons:
                lines.append(f"{class_id} {poly}")
                self.class_counts[class_name] += 1
                self.class_counts_source[f"{det['source']}:{class_name}"] += 1

        return lines

    def process_split(self, split: str, device: str = "cuda") -> int:
        logger.info(f"Procesando split: {split}")

        split_dir = self.data_root / split
        if not split_dir.exists():
            split_dir = self.data_root / split.capitalize()

        rgb_dir = split_dir / "rgb"
        image_files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
        logger.info(f"Imágenes encontradas: {len(image_files)}")

        if self.yolo is None:
            self.load_yolo(device)

        if self.teacher_enabled and self.teacher is None:
            self.load_teacher()

        processed = 0
        skipped = 0

        for idx, img_path in enumerate(
            tqdm(image_files, desc=f"Pseudo-labeling {split}")
        ):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    self.skipped_reasons["image_read_error"] += 1
                    skipped += 1
                    continue

                h, w = image.shape[:2]

                detections = self._collect_yolo_detections(image)
                detections.extend(self._collect_teacher_detections(image))
                detections = self._merge_detections(detections)

                if not detections:
                    self.skipped_reasons["no_valid_detections"] += 1
                    skipped += 1
                    continue

                label_lines = self.create_yolo_labels((h, w), detections)
                if not label_lines:
                    skipped += 1
                    continue

                is_val = idx % self.val_stride == 0
                dest_img = (
                    self.val_images / img_path.name
                    if is_val
                    else self.train_images / img_path.name
                )
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)

                label_path = str(dest_img).replace("images", "labels")
                dest_label = Path(label_path).with_suffix(".txt")
                with open(dest_label, "w", encoding="utf-8") as file:
                    file.write("\n".join(label_lines))

                processed += 1

            except Exception as exc:
                logger.error(f"Error procesando {img_path.name}: {exc}")
                skipped += 1

        logger.info(f"Split {split}: procesadas={processed}, omitidas={skipped}")
        return processed

    def create_yaml_config(self) -> Path:
        yaml_path = self.output_root / "data" / "sunrgbd.yaml"
        with open(yaml_path, "w", encoding="utf-8") as file:
            file.write(f"path: {self.output_root / 'data' / 'sunrgbd'}\n")
            file.write("train: images/train\n")
            file.write("val: images/val\n")
            file.write(f"nc: {len(self.SUNRGBD_CLASSES)}\n")
            file.write("names:\n")
            for class_id, class_name in sorted(
                self.SUNRGBD_CLASSES.items(), key=lambda item: item[0]
            ):
                file.write(f"  {class_id}: {class_name}\n")

        logger.info(f"Configuración dataset guardada: {yaml_path}")
        return yaml_path

    def log_label_quality_report(self) -> None:
        logger.info("=" * 60)
        logger.info("REPORTE DE CALIDAD DE PSEUDO-LABELS")
        logger.info("Cobertura por clase SUN-RGB-D:")
        for class_id, class_name in sorted(
            self.SUNRGBD_CLASSES.items(), key=lambda item: item[0]
        ):
            logger.info(
                f"  {class_id:02d} {class_name}: {self.class_counts.get(class_name, 0)}"
            )

        if self.class_counts_source:
            logger.info("Cobertura por fuente:")
            for key, value in sorted(self.class_counts_source.items()):
                logger.info(f"  {key}: {value}")

        if self.skipped_reasons:
            logger.info("Descartes por motivo:")
            for reason, count in self.skipped_reasons.most_common():
                logger.info(f"  {reason}: {count}")
        logger.info("=" * 60)

    def prepare(self, device: str = "cuda") -> None:
        logger.info("Generando pseudo-labels de alta precisión...")
        total = self.process_split("train", device)
        total += self.process_split("validation", device)

        self.create_yaml_config()
        self.log_label_quality_report()

        train_imgs = len(list(self.train_images.glob("*")))
        val_imgs = len(list(self.val_images.glob("*")))
        logger.info("=" * 60)
        logger.info("PREPARACIÓN COMPLETADA")
        logger.info(f"Total procesadas: {total}")
        logger.info(f"Imágenes train: {train_imgs}")
        logger.info(f"Imágenes val: {val_imgs}")
        logger.info("=" * 60)


def _load_train_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    import torch

    config_path = Path(__file__).parent.parent / "train_config.yaml"
    config = _load_train_config(config_path)

    training_cfg = config.get("TRAINING", {})
    device = training_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    logger.info(f"Usando dispositivo: {device}")

    generator = PseudoLabelGenerator(
        data_root=config["DATA_ROOT"],
        output_root=config["OUTPUT_ROOT"],
        config=config,
    )
    generator.prepare(device=device)


if __name__ == "__main__":
    main()
