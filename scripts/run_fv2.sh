#!/usr/bin/env bash
set -e
# Pipeline Fusion Vision v2 completo
# Requiere configs/fv2_config.yaml y data/SUNRGBD
python src/fusion/main.py
# Por fases:
# python src/fusion/train/prepare_data.py
# python src/fusion/train/train.py
# python src/fusion/train/evaluate.py
