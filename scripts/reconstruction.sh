#!/usr/bin/env bash
set -e
# Reconstrucción 3D (RANSAC, Poisson, MLS)
python src/reconstruction/tools/run_final_reconstruction.py
python src/reconstruction/tools/reconstruction_max_quality.py
