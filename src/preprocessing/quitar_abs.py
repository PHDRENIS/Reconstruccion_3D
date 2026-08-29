# Renombra archivos quitando el sufijo '_abs' añadido por error en pasos previos.
# Uso: python -m src.preprocessing.quitar_abs [--root data/SUNRGBD]
import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TEXTO_A_ELIMINAR = "_abs"


def get_directorios(root: Path):
    return [
        root / "Train" / "mask",
        root / "Train" / "rgb",
        root / "Validation" / "mask",
        root / "Validation" / "rgb",
    ]


# Legacy (no usar): C:\Users\victo\OneDrive\Documents\TT\SUNRBG_IMAGES\...


def corregir_nombres(root=None):
    root = Path(root) if root else REPO_ROOT / "data" / "SUNRGBD"
    directorios = get_directorios(root)
    print(f"Buscando archivos con '{TEXTO_A_ELIMINAR}' para renombrar...\n")

    total_renombrados = 0

    for carpeta in directorios:
        if not carpeta.exists():
            print(f"[ADVERTENCIA] No existe la carpeta: {carpeta}")
            continue

        print(f" Analizando: {carpeta}")
        archivos = os.listdir(carpeta)
        contador_carpeta = 0

        for archivo in archivos:
            if TEXTO_A_ELIMINAR in archivo:
                ruta_vieja = carpeta / archivo
                nuevo_nombre = archivo.replace(TEXTO_A_ELIMINAR, "")
                ruta_nueva = carpeta / nuevo_nombre

                try:
                    os.rename(ruta_vieja, ruta_nueva)
                    contador_carpeta += 1
                except FileExistsError:
                    print(f"    [ERROR] No se pudo renombrar {archivo} porque {nuevo_nombre} YA EXISTE.")
                except Exception as e:
                    print(f"  [ERROR] Falló {archivo}: {e}")

        print(f"    Corregidos en esta carpeta: {contador_carpeta}")
        total_renombrados += contador_carpeta

    print(f"\n Se renombraron un total de {total_renombrados} archivos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quitar sufijo _abs de archivos")
    parser.add_argument("--root", type=str, default=None, help="Raiz SUNRGBD (default: data/SUNRGBD)")
    args = parser.parse_args()
    corregir_nombres(args.root)
