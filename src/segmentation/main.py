#!/usr/bin/env python3
"""
main.py — Fine-Tune YOLO11x-seg para IR (single-class)

Pipeline completo:
  1) Genera pseudo-labels con FastSAM + Depth
  2) Prepara estructura de dataset YOLO
  3) Fine-tunea modelo pre-entrenado (SUNRGBD -> IR)
  4) Evalua y guarda modelo final

Uso:
    python main.py                          # Pipeline completo
    python main.py --pseudolabels-only      # Solo generar pseudo-labels
    python main.py --train-only             # Solo entrenar (ya hay labels)
    python main.py --eval-only              # Solo evaluar modelo existente
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger

from generate_pseudolabels import generate_pseudolabels
from dataset import prepare_yolo_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-Tune YOLO IR")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pseudolabels-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda:0)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config no encontrado: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = Path(cfg["pseudolabels"]["output_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(cfg["model"]["base"])
    model_final = Path(cfg["model"]["output"])
    model_final.parent.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["train"]
    pl_cfg = cfg["pseudolabels"]
    path_cfg = cfg["paths"]
    device = args.device or train_cfg.get("device", "cuda:0")

    # ------------------- FASE 1: Pseudo-labels -------------------
    if not args.train_only and not args.eval_only:
        logger.info("=" * 60)
        logger.info("FASE 1: GENERANDO PSEUDO-LABELS (FastSAM + Depth)")
        logger.info("=" * 60)

        n_labels = generate_pseudolabels(
            ir_dir=path_cfg["ir_images"],
            depth_dir=path_cfg["depth_maps"],
            output_dir=str(labels_dir),
            min_area=pl_cfg["min_area"],
            conf=pl_cfg["conf"],
            depth_percentile=pl_cfg.get("depth_percentile", 80.0),
        )

        if n_labels == 0:
            logger.error("No se generaron pseudo-labels. Abortando.")
            return

    if args.pseudolabels_only:
        logger.info("Pseudo-labels generadas. Fin.")
        return

    # ------------------- FASE 2: Preparar dataset -------------------
    logger.info("=" * 60)
    logger.info("FASE 2: PREPARANDO DATASET YOLO")
    logger.info("=" * 60)

    yaml_path = prepare_yolo_dataset(
        ir_dir=path_cfg["ir_images"],
        pseudolabels_dir=str(labels_dir),
        data_dir=path_cfg["data_dir"],
    )

    if not yaml_path:
        logger.error("Dataset no pudo prepararse. Abortando.")
        return

    if args.eval_only:
        # Solo evaluar
        logger.info("=== EVALUANDO MODELO ===")
        from ultralytics import YOLO

        m = YOLO(str(model_final)) if model_final.exists() else YOLO(str(model_path))
        metrics = m.val(data=yaml_path, imgsz=train_cfg["imgsz"], batch=train_cfg["batch"],
                        device=device, split="train", plots=True, project=str(out_dir.parent),
                        name="yolo_ir_eval", exist_ok=True)
        logger.info(f"Box mAP50: {getattr(metrics.box, 'map50', 0):.4f}")
        logger.info(f"Box mAP50-95: {getattr(metrics.box, 'map', 0):.4f}")
        if hasattr(metrics, 'seg'):
            logger.info(f"Mask mAP50: {getattr(metrics.seg, 'map50', 0):.4f}")
        return

    # ------------------- FASE 3: Fine-tune -------------------
    logger.info("=" * 60)
    logger.info("FASE 3: FINE-TUNE YOLO EN IR")
    logger.info(f"Modelo base: {model_path}")
    logger.info(f"Epocas: {train_cfg['epochs']}, Batch: {train_cfg['batch']}")
    logger.info(f"LR: {train_cfg['lr0']}, Device: {device}")
    logger.info("=" * 60)

    from ultralytics import YOLO

    if not model_path.exists():
        fb = cfg["model"].get("base_fallback", "yolo11x-seg.pt")
        logger.warning(f"Modelo base no encontrado: {model_path}")
        logger.info(f"Usando fallback: {fb}")
        yolo_model = YOLO(fb)
    else:
        logger.info(f"Cargando modelo base: {model_path}")
        yolo_model = YOLO(str(model_path))

    run_dir = out_dir.parent / "runs" / "segment" / "yolo_ir_finetune"

    yolo_model.train(
        data=yaml_path,
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch"],
        imgsz=train_cfg["imgsz"],
        device=device,
        workers=train_cfg["workers"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=train_cfg["warmup_epochs"],
        cos_lr=train_cfg["cos_lr"],
        close_mosaic=train_cfg["close_mosaic"],
        patience=train_cfg["patience"],
        optimizer="AdamW",
        amp=True,
        hsv_h=train_cfg["hsv_h"],
        hsv_s=train_cfg["hsv_s"],
        hsv_v=train_cfg["hsv_v"],
        degrees=train_cfg["degrees"],
        translate=train_cfg["translate"],
        scale=train_cfg["scale"],
        fliplr=train_cfg["fliplr"],
        mosaic=train_cfg["mosaic"],
        mixup=train_cfg["mixup"],
        project=str(out_dir.parent),
        name="yolo_ir_finetune",
        exist_ok=True,
        plots=True,
        save=True,
    )

    best_path = run_dir / "weights" / "best.pt"
    if best_path.exists():
        shutil.copy2(best_path, model_final)
        logger.info(f"Modelo fine-tuneado guardado: {model_final}")
    else:
        logger.error("No se encontro best.pt tras fine-tune")

    logger.info("=" * 60)
    logger.info("FINE-TUNE COMPLETADO")
    logger.info("=" * 60)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
               level="INFO")
    main()
