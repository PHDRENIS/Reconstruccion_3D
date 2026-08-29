#!/usr/bin/env python3
"""
Script para entrenar YOLO11x-seg en SUN-RGB-D con perfil de máxima precisión.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
import yaml
from loguru import logger


def check_gpu() -> bool:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detectada: {gpu_name}")
        logger.info(f"Memoria: {gpu_memory:.1f} GB")
        return True

    logger.warning("No se detectó GPU, se usará CPU")
    return False


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_train_kwargs(train_cfg: Dict, has_gpu: bool, device: str) -> Dict:
    kwargs = {
        "epochs": train_cfg.get("epochs", 500),
        "imgsz": train_cfg.get("imgsz", 1536),
        "batch": train_cfg.get("batch", 24),
        "device": device,
        "optimizer": train_cfg.get("optimizer", "AdamW"),
        "lr0": train_cfg.get("lr0", 0.001),
        "lrf": train_cfg.get("lrf", 0.01),
        "momentum": train_cfg.get("momentum", 0.937),
        "weight_decay": train_cfg.get("weight_decay", 0.0005),
        "warmup_epochs": train_cfg.get("warmup_epochs", 5.0),
        "warmup_momentum": train_cfg.get("warmup_momentum", 0.8),
        "warmup_bias_lr": train_cfg.get("warmup_bias_lr", 0.1),
        "cos_lr": train_cfg.get("cos_lr", True),
        "close_mosaic": train_cfg.get("close_mosaic", 20),
        "overlap_mask": train_cfg.get("overlap_mask", True),
        "mask_ratio": train_cfg.get("mask_ratio", 1),
        "workers": train_cfg.get("workers", 16),
        "cache": train_cfg.get("cache", "ram"),
        "patience": train_cfg.get("patience", 100),
        "val": train_cfg.get("val", True),
        "plots": train_cfg.get("plots", True),
        "save": train_cfg.get("save", True),
        "amp": has_gpu and train_cfg.get("amp", True),
        "hsv_h": train_cfg.get("hsv_h", 0.015),
        "hsv_s": train_cfg.get("hsv_s", 0.7),
        "hsv_v": train_cfg.get("hsv_v", 0.4),
        "degrees": train_cfg.get("degrees", 5.0),
        "translate": train_cfg.get("translate", 0.1),
        "scale": train_cfg.get("scale", 0.25),
        "shear": train_cfg.get("shear", 2.0),
        "perspective": train_cfg.get("perspective", 0.0),
        "flipud": train_cfg.get("flipud", 0.0),
        "fliplr": train_cfg.get("fliplr", 0.5),
        "mosaic": train_cfg.get("mosaic", 1.0),
        "mixup": train_cfg.get("mixup", 0.1),
        "copy_paste": train_cfg.get("copy_paste", 0.0),
    }
    return kwargs


def train(config_path: str = None) -> None:
    from ultralytics import YOLO

    if config_path is None:
        config_path = Path(__file__).parent.parent / "train_config.yaml"
    else:
        config_path = Path(config_path)

    config = load_config(config_path)

    has_gpu = check_gpu()
    train_cfg = config.get("TRAINING", {})
    device = train_cfg.get("device", "0") if has_gpu else "cpu"

    base_path = Path(__file__).parent.parent
    models_dir = base_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / config.get("MODEL_BASE", "yolo11x-seg.pt")

    if model_path.exists():
        logger.info(f"Cargando modelo base local: {model_path}")
        model = YOLO(str(model_path))
    else:
        logger.info(
            f"Descargando modelo base: {config.get('MODEL_BASE', 'yolo11x-seg.pt')}"
        )
        model = YOLO(config.get("MODEL_BASE", "yolo11x-seg.pt"))

    data_yaml = base_path / "data" / "sunrgbd.yaml"
    if not data_yaml.exists():
        logger.error(f"No se encontró archivo de datos: {data_yaml}")
        logger.info("Ejecuta primero: python train/prepare_data.py")
        return

    run_project = base_path / "runs" / "segment"
    run_name = "sunrgbd_train_high_precision"
    train_kwargs = _build_train_kwargs(train_cfg, has_gpu, device)

    logger.info("=" * 70)
    logger.info("INICIANDO ENTRENAMIENTO YOLO11X-SEG (MÁXIMA PRECISIÓN)")
    logger.info(f"Modelo base: {config.get('MODEL_BASE', 'yolo11x-seg.pt')}")
    logger.info(f"Épocas: {train_kwargs['epochs']}")
    logger.info(f"Batch: {train_kwargs['batch']}")
    logger.info(f"Image size: {train_kwargs['imgsz']}")
    logger.info(f"Optimizer: {train_kwargs['optimizer']}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)

    model.train(
        data=str(data_yaml),
        project=str(run_project),
        name=run_name,
        exist_ok=True,
        **train_kwargs,
    )

    best_weights = run_project / run_name / "weights" / "best.pt"
    last_weights = run_project / run_name / "weights" / "last.pt"
    final_model_path = models_dir / "yolo11x-sunrgbd-seg.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, final_model_path)
        logger.info(f"Modelo final guardado en: {final_model_path}")
    elif last_weights.exists():
        shutil.copy2(last_weights, final_model_path)
        logger.warning("No se encontró best.pt, se usó last.pt como respaldo")
        logger.info(f"Modelo final guardado en: {final_model_path}")
    else:
        logger.error("No se encontraron pesos resultantes del entrenamiento")

    logger.info("=" * 70)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 70)


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    train()


if __name__ == "__main__":
    main()
