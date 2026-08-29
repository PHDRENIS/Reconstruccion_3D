# Datos — SUN RGB-D

Este repositorio **NO incluye** el dataset completo por tamaño (>10GB).

## Descarga

1. SUN RGB-D oficial: http://rgbd.cs.princeton.edu/
2. O usa el script de preparación desde `data/samples/` (3 imágenes de ejemplo incluidas).

## Estructura esperada

```
data/
├── SUNRGBD/                    # <- Coloca aquí el dataset descomprimido (ignorado por .gitignore)
│   ├── Train/
│   │   ├── rgb/                # *.jpg  (480x640)
│   │   ├── depth_input/        # *.npy  (metros) o *.png (mm)
│   │   ├── depth_gt/           # *.npy  (metros)
│   │   └── mask/               # *.png  binaria 0/255
│   └── Validation/
│       └── ... (misma estructura)
├── SUNRBG_IMAGES/              # Alias legacy (si tu código aún lo referencia)
└── samples/                    # 3 jpgs de ejemplo versionados (val/rgb)
```

## Preparación

```bash
# 1. Convertir PNG (mm) -> NPY (m)
python src/preprocessing/png_to_npy.py

# 2. Redimensionar a 640x480
python src/preprocessing/Resizing_depth.py
python src/preprocessing/Resizing_masks.py

# 3. Quitar sufijo _abs mal generado
python src/preprocessing/quitar_abs.py

# 4. Regenerar máscaras binarias
python src/preprocessing/to_binary.py
```

> **IMPORTANTE:** Edita `configs/*.yaml` y `src/depth_completion/train.py` para apuntar a `data/SUNRGBD` con rutas relativas, no `C:\Users\victo\...`.
