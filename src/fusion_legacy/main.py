from __future__ import annotations

import sys
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

import yaml
import numpy as np
from tqdm import tqdm
from loguru import logger
import time

from src.data.sunrgbd_loader import SUNRGBDLoader
from src.processing.semantic_pipeline import SemanticPipeline
from src.utils.visualization import Visualizer


def setup_logging(log_file: str = None):
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
        )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_split(
    pipeline, visualizer, loader, split: str, max_images: int = None
) -> dict:
    logger.info(f"Procesando {split}: {len(loader)} imágenes")

    stats = {
        "total_images": 0,
        "total_detections": 0,
        "total_static": 0,
        "total_dynamic": 0,
        "classes_detected": {},
        "processing_times": [],
        "errors": 0,
    }

    num_images = min(len(loader), max_images) if max_images else len(loader)

    for idx in tqdm(range(num_images), desc=f"Procesando {split}"):
        try:
            data = loader.load_item(idx)

            result = pipeline.process(
                image=data["image"],
                image_id=data["image_id"],
                image_path=data["image_path"],
                depth_gt=data.get("depth_gt"),
                depth_input=data.get("depth_input"),
            )

            visualizer.save_result(result, split=split)

            stats["total_images"] += 1
            stats["total_detections"] += len(result.detections)
            stats["total_static"] += len(result.static_masks)
            stats["total_dynamic"] += len(result.dynamic_masks)
            stats["processing_times"].append(result.processing_time)

            for detection in result.detections:
                class_name = detection.class_name
                stats["classes_detected"][class_name] = (
                    stats["classes_detected"].get(class_name, 0) + 1
                )

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                logger.error(f"Error imagen {idx}: {e}")
            continue

    if stats["processing_times"]:
        stats["avg_processing_time"] = np.mean(stats["processing_times"])
        stats["min_processing_time"] = np.min(stats["processing_times"])
        stats["max_processing_time"] = np.max(stats["processing_times"])

    return stats


def main():
    base_dir = Path("C:/Users/renea/Desktop/TT/F Vision")
    config_path = base_dir / "config.yaml"
    data_root = "C:/Users/renea/Desktop/TT/SUNRBG_IMAGES"
    output_dir = base_dir / "output"

    log_file = output_dir / "logs" / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_file))

    logger.info("=" * 60)
    logger.info("INICIO DEL PIPELINE DE PROCESAMIENTO SEMÁNTICO")
    logger.info("=" * 60)

    config = load_config(str(config_path))
    config["paths"]["data_root"] = data_root

    splits_to_process = config.get("dataset", {}).get("split", "train")
    if splits_to_process == "all":
        splits_to_process = ["train", "validation"]
    else:
        splits_to_process = [splits_to_process]

    max_images = config.get("dataset", {}).get("max_images")

    all_stats = {}

    for split in splits_to_process:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"PROCESANDO SPLIT: {split.upper()}")
        logger.info(f"{'=' * 60}")

        loader = SUNRGBDLoader(data_root, split=split)

        pipeline = SemanticPipeline(
            config=config,
            device_yolo=config["models"]["yolo"]["device"],
            device_rtdetr=config["models"]["rt_detr"]["device"],
        )

        visualizer = Visualizer(
            output_dir=str(output_dir),
            save_overlays=config["processing"]["save_overlays"],
            save_masks=config["processing"]["save_masks"],
            save_json=config["processing"]["save_json"],
            save_depth=config["processing"]["save_depth"],
            overlay_alpha=config["processing"]["overlay_alpha"],
            overlay_thickness=config["processing"]["overlay_thickness"],
        )

        start_time = time.time()
        stats = process_split(pipeline, visualizer, loader, split, max_images)
        elapsed_time = time.time() - start_time

        all_stats[split] = stats

        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESUMEN - {split.upper()}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Imágenes procesadas: {stats['total_images']}")
        logger.info(f"Errores: {stats['errors']}")
        logger.info(f"Detecciones totales: {stats['total_detections']}")
        logger.info(f"Objetos estáticos: {stats['total_static']}")
        logger.info(f"Objetos dinámicos: {stats['total_dynamic']}")
        logger.info(f"Tiempo promedio: {stats.get('avg_processing_time', 0):.3f}s")
        logger.info(f"Tiempo total: {elapsed_time:.1f}s")

        if stats["classes_detected"]:
            logger.info(f"\nTop 10 clases:")
            sorted_classes = sorted(
                stats["classes_detected"].items(), key=lambda x: x[1], reverse=True
            )[:10]
            for class_name, count in sorted_classes:
                logger.info(f"  {class_name}: {count}")

    logger.info(f"\n{'=' * 60}")
    logger.info("PIPELINE COMPLETADO")
    logger.info(f"{'=' * 60}")

    stats_file = output_dir / "statistics.json"
    import json

    with open(stats_file, "w") as f:
        json_stats = {}
        for split, stats in all_stats.items():
            json_stats[split] = {}
            for k, v in stats.items():
                if isinstance(v, (np.integer, np.floating)):
                    json_stats[split][k] = float(v)
                else:
                    json_stats[split][k] = v
        json.dump(json_stats, f, indent=2)

    logger.info(f"Estadísticas guardadas en: {stats_file}")


if __name__ == "__main__":
    main()
