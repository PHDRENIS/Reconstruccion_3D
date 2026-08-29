#!/usr/bin/env bash
set -e
# Fine-tune YOLO-seg IR: pseudo-labels + train
# Requiere configs/yolo_ir_config.yaml y data/SUNRGBD
python src/segmentation/main.py --config configs/yolo_ir_config.yaml
# Solo pseudo-labels: python src/segmentation/main.py --pseudolabels-only
# Solo train: python src/segmentation/main.py --train-only
