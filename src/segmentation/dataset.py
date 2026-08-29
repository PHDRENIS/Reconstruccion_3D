#!/usr/bin/env python3
"""
Prepara la estructura de dataset YOLO para fine-tune:
  data_dir/
    images/train/   <- copia de IR
    labels/train/   <- copia de pseudo-labels

Genera dataset.yaml automatico.
"""

import shutil
from pathlib import Path

from loguru import logger


def prepare_yolo_dataset(
    ir_dir: str,
    pseudolabels_dir: str,
    data_dir: str,
) -> str:
    ir_dir = Path(ir_dir)
    labels_dir = Path(pseudolabels_dir)
    data_dir = Path(data_dir)

    images_train = data_dir / "images" / "train"
    labels_train = data_dir / "labels" / "train"
    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)

    ir_files = sorted(ir_dir.glob("*.jpg")) + sorted(ir_dir.glob("*.png"))
    copied = 0

    for ir_path in ir_files:
        label_path = labels_dir / f"{ir_path.stem}.txt"
        if not label_path.exists():
            continue

        ext = ".jpg" if ir_path.suffix.lower() in (".jpg", ".jpeg") else ".png"
        dest_img = images_train / f"{ir_path.stem}{ext}"
        if not dest_img.exists():
            shutil.copy2(ir_path, dest_img)

        dest_label = labels_train / f"{ir_path.stem}.txt"
        if not dest_label.exists() or dest_label.stat().st_mtime < label_path.stat().st_mtime:
            shutil.copy2(label_path, dest_label)

        copied += 1

    if copied == 0:
        logger.error("No se encontraron pares IR + pseudo-label")
        return ""

    yaml_path = data_dir / "dataset_finetune.yaml"
    yaml_path.write_text(f"""# Dataset fine-tune YOLO IR (single-class)
path: {data_dir.resolve()}
train: images/train
val: images/train
nc: 1
names:
  0: object
""", encoding="utf-8")

    logger.info(f"Dataset preparado: {copied} pares en {data_dir}")
    logger.info(f"YAML: {yaml_path}")
    return str(yaml_path)
