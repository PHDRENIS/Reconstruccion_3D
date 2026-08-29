#!/usr/bin/env python3
"""
Script para evaluar modelo YOLO11x-seg entrenado en SUN-RGB-D.
Incluye métricas de detección y segmentación globales y por clase.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from loguru import logger


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _name_at(names, class_idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_idx, class_idx))
    if isinstance(names, (list, tuple)) and class_idx < len(names):
        return str(names[class_idx])
    return str(class_idx)


def _load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _get_eval_metric(metrics_obj, names, prefix: str) -> Dict:
    if metrics_obj is None:
        return {
            "global": {
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            },
            "per_class": {},
        }

    prefix_lower = prefix.lower()

    global_values = {
        "map50": _safe_float(getattr(metrics_obj, "map50", 0.0)),
        "map50_95": _safe_float(getattr(metrics_obj, "map", 0.0)),
        "precision": _safe_float(getattr(metrics_obj, "mp", 0.0)),
        "recall": _safe_float(getattr(metrics_obj, "mr", 0.0)),
    }

    per_class = {}
    all_ap = getattr(metrics_obj, "all_ap", None)
    p = getattr(metrics_obj, "p", None)
    r = getattr(metrics_obj, "r", None)

    if all_ap is not None:
        try:
            for class_idx, ap_values in enumerate(all_ap):
                class_name = _name_at(names, class_idx)

                map50 = _safe_float(ap_values[0]) if len(ap_values) > 0 else 0.0
                map50_95 = (
                    _safe_float(sum(ap_values) / len(ap_values))
                    if len(ap_values) > 0
                    else 0.0
                )

                precision = (
                    _safe_float(p[class_idx])
                    if p is not None and len(p) > class_idx
                    else 0.0
                )
                recall = (
                    _safe_float(r[class_idx])
                    if r is not None and len(r) > class_idx
                    else 0.0
                )

                per_class[class_name] = {
                    "map50": map50,
                    "map50_95": map50_95,
                    "precision": precision,
                    "recall": recall,
                }
        except Exception as exc:
            logger.warning(f"No se pudo extraer per-class {prefix_lower}: {exc}")

    return {
        "global": global_values,
        "per_class": per_class,
    }


def evaluate(model_path: str = None, data_yaml: str = None) -> None:
    from ultralytics import YOLO

    base_path = Path(__file__).parent.parent
    cfg = _load_config(base_path / "train_config.yaml")
    eval_cfg = cfg.get("EVALUATION", {})

    if model_path is None:
        model_path = base_path / "models" / "yolo11x-sunrgbd-seg.pt"
    else:
        model_path = Path(model_path)

    if data_yaml is None:
        data_yaml = base_path / "data" / "sunrgbd.yaml"
    else:
        data_yaml = Path(data_yaml)

    if not model_path.exists():
        logger.error(f"No se encontró modelo: {model_path}")
        return

    if not data_yaml.exists():
        logger.error(f"No se encontró archivo de datos: {data_yaml}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Cargando modelo: {model_path}")
    logger.info(f"Usando dispositivo: {device}")

    model = YOLO(str(model_path))

    logger.info("=" * 70)
    logger.info("EVALUANDO MODELO YOLO11X-SEG (ALTA PRECISIÓN)")
    logger.info("=" * 70)

    run_project = base_path / "runs" / "segment"
    run_name = "sunrgbd_eval_high_precision"
    metrics = model.val(
        data=str(data_yaml),
        imgsz=eval_cfg.get("imgsz", 1536),
        batch=eval_cfg.get("batch", 12),
        device=device,
        split=eval_cfg.get("split", "val"),
        augment=eval_cfg.get("augment", True),
        plots=eval_cfg.get("plots", True),
        save_json=eval_cfg.get("save_json", True),
        project=str(run_project),
        name=run_name,
        exist_ok=True,
    )

    names = metrics.names if hasattr(metrics, "names") else {}

    box_stats = _get_eval_metric(getattr(metrics, "box", None), names, prefix="box")
    seg_stats = _get_eval_metric(getattr(metrics, "seg", None), names, prefix="seg")

    logger.info("=" * 70)
    logger.info("MÉTRICAS GLOBALES")
    logger.info(
        f"Box  -> mAP50: {box_stats['global']['map50']:.4f} | mAP50-95: {box_stats['global']['map50_95']:.4f} | P: {box_stats['global']['precision']:.4f} | R: {box_stats['global']['recall']:.4f}"
    )
    logger.info(
        f"Seg  -> mAP50: {seg_stats['global']['map50']:.4f} | mAP50-95: {seg_stats['global']['map50_95']:.4f} | P: {seg_stats['global']['precision']:.4f} | R: {seg_stats['global']['recall']:.4f}"
    )
    logger.info("=" * 70)

    output_dir = run_project / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": str(model_path),
        "evaluation": {
            "imgsz": eval_cfg.get("imgsz", 1536),
            "batch": eval_cfg.get("batch", 12),
            "split": eval_cfg.get("split", "val"),
            "augment": eval_cfg.get("augment", True),
            "device": device,
        },
        "box": box_stats,
        "seg": seg_stats,
    }

    output_metrics = output_dir / "metrics_high_precision.json"
    with open(output_metrics, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    logger.info(f"Métricas guardadas en: {output_metrics}")


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    evaluate()


if __name__ == "__main__":
    main()
