# Reconstrucción 3D de Escenas Interiores con Cámara de Profundidad de Bajo Costo

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/vhtc-rasm/TT-Limpio/actions/workflows/ci.yml/badge.svg)](https://github.com/vhtc-rasm/TT-Limpio/actions/workflows/ci.yml)

> Trabajo Terminal — IPN UPIIT | VHTC / RASM

Pipeline completo para **completamiento de profundidad + segmentación semántica (YOLO) + reconstrucción 3D** usando SUN RGB-D.

##  Estructura

```
TT Limpio/
├── configs/                  # Configuraciones YAML (YOLO, FV2, SUNRGBD)
│   ├── yolo_ir_config.yaml
│   ├── sunrgbd.yaml
│   ├── fv2_config.yaml
│   ├── fv2_train_config.yaml
│   ├── yolo_ir_dataset.yaml
│   └── legacy/               # f_vision_* archivado (rutas /zfs-home absolutas)
├── data/
│   ├── README.md             # Cómo obtener SUN RGB-D (no se versiona)
│   └── samples/              # 3 imágenes de ejemplo (val/rgb)
├── docs/
│   ├── reporte/              # PDFs finales (Reconstruccion_3D_...pdf, Trabajo_Terminal_*.pdf)
│   ├── reporte_final/        # Fuente LaTeX (main.tex + Figuras/)
│   ├── diapositivas/         # Presentación
│   ├── ejemplos/             # PNG muestras versionadas (.raw ignorados via .gitignore)
│   └── bitacora/             # Bitácora TT
├── src/
│   ├── preprocessing/        # png_to_npy, Resizing_*, quitar_abs, to_binary (rutas relativas + argparse)
│   ├── depth_completion/     # EfficientNet UNet (TensorFlow) — data_loader, model_builder, train
│   ├── segmentation/         # YOLO IR fine-tune — main, dataset, generate_pseudolabels
│   ├── fusion/               # Pipeline Fusión Visión v2 (FV2) — loader, models, processing, train
│   ├── fusion_legacy/        # F Vision v1 (referencia)
│   └── reconstruction/tools/ # 25 scripts de reconstrucción 3D (RANSAC, Poisson, MLS, etc.)
├── experiments/
│   ├── ablacion/             # p70/p80/p90 (segmentation_summary.txt + masks/.gitkeep, 954 PNGs ignorados)
│   ├── curva_aprendizaje/
│   ├── metricas/
│   └── reconstruccion/       # JSONs y scripts de evaluación (sin .ply gigantes)
├── scripts/                  # Wrappers bash (preprocessing.sh, train_yolo_ir.sh, run_fv2.sh, reconstruction.sh)
├── outputs/                  # .gitignore — resultados generados
├── LICENSE                   # MIT
├── .gitattributes            # LFS opcional + normalización eol
├── environment.yml           # conda env tt (python 3.12 + requirements.txt)
└── requirements.txt
```

##  Instalación

```bash
# 1. Clonar (ajusta <user> o usa SSH)
git clone https://github.com/vhtc-rasm/TT-Limpio.git
cd "TT Limpio"

# 2. Entorno (elige uno)
python -m venv .venv && .venv\Scripts\activate        # Windows
# o
conda env create -f environment.yml && conda activate tt

# 3. Dependencias
pip install -r requirements.txt
```

> **Nota:** Los pesos grandes (`*.pt`, `*.pth`, `best_depth_model*.pth`) están ignorados por `.gitignore`. Descárgalos de Releases / Drive y colócalos en `models/` o usa `ultralytics` que descarga `yolo11n.pt` automáticamente.

##  Datos

El dataset **SUN RGB-D** NO se incluye en el repo por tamaño. Ver `data/README.md`:

```bash
# Estructura esperada (no versionada):
data/SUNRGBD/
├── Train/
│   ├── rgb/
│   ├── depth_input/   # .npy o .png (mm -> m)
│   ├── depth_gt/
│   └── mask/          # .png binaria
└── Validation/
    └── ...
```

Usa `src/preprocessing/png_to_npy.py` y `Resizing_*.py` para preparar el dataset. Solo `data/samples/` (3 jpgs) se versiona como ejemplo.

##  Uso

### 1. Preprocessing
```bash
python src/preprocessing/png_to_npy.py
python src/preprocessing/Resizing_masks.py
```

### 2. Depth Completion (EfficientNet UNet)
```bash
# Edita primero src/depth_completion/train.py -> TRAIN_RGB, VAL_RGB (ahora relativos)
python -m src.depth_completion.train
```

### 3. Segmentación YOLO IR (recomendado)
```bash
python src/segmentation/main.py --config configs/yolo_ir_config.yaml
# Solo pseudo-labels:
python src/segmentation/main.py --pseudolabels-only
# Solo entrenar:
python src/segmentation/main.py --train-only
```

### 4. Pipeline Fusión Visión v2 (completo)
```bash
python src/fusion/main.py   # requiere configs/fv2_config.yaml + SUNRGBD en data/
# O por fases:
python src/fusion/train/prepare_data.py
python src/fusion/train/train.py
python src/fusion/train/evaluate.py
```

### 5. Reconstrucción 3D
```bash
python src/reconstruction/tools/run_final_reconstruction.py
python src/reconstruction/tools/reconstruction_max_quality.py
```

##  Configuración

Todas las rutas hardcodeadas originales (`C:\Users\victo\...`, `/zfs-home/...`) fueron centralizadas en `configs/`. **Debes editar `configs/*.yaml` antes de ejecutar**:

- `configs/yolo_ir_config.yaml` → `paths.ir_images`, `paths.depth_maps`
- `configs/fv2_config.yaml` → `paths.data_root`
- `configs/sunrgbd.yaml` → `path`

##  Qué se limpió respecto al original

- Eliminados: `3x .venv` (~2GB), `TT (1).zip` (3.5GB), `Diapositivas.zip`, `TT Reporte final.zip`, `__pycache__/`, `.ruff_cache`, `.ply` gigantes (>5MB), `SUNRBG_IMAGES` duplicado, `TT/TT/` anidado.
- Renombrado sin espacios/acentos (`F Vision` → `fusion`, `Nueva reconstrucción` → `experiments/reconstruccion`; 3 archivos `Reconstrucción...` → `reconstruccion_*.jpeg/pdf`).
- Archivados `configs/f_vision_*` en `configs/legacy/` (rutas absolutas `/zfs-home` como referencia).
- Ablaciones: solo métricas + `.gitkeep`/`README.md` en `masks/` (954 PNGs ~4.6 MB ignorados via `.gitignore:52`, antes 1911 imágenes). `overlays/*.png` ignorados salvo `sample_*.png`.
- `docs/ejemplos/*.raw` (10× ~400KB) ignorados via `.gitignore:57`; solo `.png` + `README.md` versionados.
- LaTeX: `docs/reporte_final/main.bbl-SAVE-ERROR` eliminado, auxiliares `*.aux/*.log` ignorados.
- Inferencia: `Validation/*.npy` (3.3GB) removido, queda `data/samples/` de ejemplo.

##  Reproducibilidad

- `requirements.txt` unificado (torch, ultralytics, tensorflow, opencv, open3d)
- Semillas y `args.yaml` en `experiments/`
- Para GitHub: usar **Git LFS** si necesitas versionar `*.pt`/`*.ply`:
  ```bash
  git lfs track "*.pt" "*.ply" "*.pth"
  ```

##  Licencia

MIT — ver [LICENSE](LICENSE) (VHTC / RASM 2025-2026).

##  Citar

Si usas este trabajo, por favor cita `CITATION.cff` o:

```bibtex
@thesis{tt-limpio-2026,
  title  = {Reconstrucción 3D de Escenas Interiores mediante una Cámara de Profundidad de Bajo Costo},
  author = {Tecpa Cisneros, Victor Hugo and Santos Mora, René Alejandro},
  school = {IPN UPIIT},
  year   = {2026},
  url    = {https://github.com/vhtc-rasm/TT-Limpio}
}
```

##  Autores

VHTC / RASM — Trabajo Terminal 2025-2026. Ver `docs/reporte/Trabajo_Terminal_VHTC_RASM.pdf` y `docs/reporte/Reconstruccion_3D_Escenas_Interiores_Camara_Bajo_Costo_v1.pdf`.

##  Contribuir

Ver [CONTRIBUTING.md](CONTRIBUTING.md) y `scripts/README.md` para wrappers reproducibles.
