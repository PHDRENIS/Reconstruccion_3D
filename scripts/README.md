# Scripts

Wrappers bash para ejecución reproducible.

- `preprocessing.sh` — PNG→NPY, resize, quitar _abs, binarizar
- `train_yolo_ir.sh` — fine-tune YOLO IR (`configs/yolo_ir_config.yaml`)
- `run_fv2.sh` — pipeline FV2 completo
- `reconstruction.sh` — reconstrucción 3D final

Todos asumen `data/SUNRGBD` existente. Editar `configs/*.yaml` antes.
En Windows usar `python src/...` directo; los `.sh` son para Linux/HPC.
