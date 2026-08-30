# Contribuir a TT Limpio

Gracias por considerar contribuir.

## Flujo

1. Haz fork y crea una rama (`git checkout -b feat/nueva-funcion`).
2. Instala el entorno: `conda env create -f environment.yml && conda activate tt` o `python -m venv .venv`.
3. Ejecuta linters antes de commit:
   ```bash
   pip install ruff
   ruff check src/
   python -m py_compile src/preprocessing/*.py src/depth_completion/*.py
   ```
4. Asegura que `.gitignore` no sea violado: `git status --ignored` debe mostrar solo `outputs/`, `data/SUNRGBD/`, `*.pt`, `experiments/ablacion/*/masks/*.png`.
5. Commit con mensaje convencional (`feat:`, `fix:`, `chore:`) y abre PR contra `master`.

## Estructura

- `configs/*.yaml` — rutas relativas a `data/SUNRGBD` (no hardcodear `/zfs-home` ni `C:\Users\...`).
- `src/preprocessing/` — scripts con `argparse` y `REPO_ROOT`.
- `experiments/` — solo métricas + `README.md`; datos pesados ignorados.
- `scripts/*.sh` — wrappers reproducibles; en Windows usar `python -m src...`.

## Reporte de issues

Incluye: versión de `requirements.txt`, comando ejecutado, log y `configs/*.yaml` usado.

## Licencia

Al contribuir aceptas que tu código se distribuya bajo [LICENSE](LICENSE) MIT.
