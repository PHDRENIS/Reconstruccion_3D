from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import cv2
from loguru import logger


class SUNRGBDLoader:
    """
    Cargador de imágenes y profundidades del dataset SUN-RGB-D.

    Soporta formatos: .jpg, .png, .npy para profundidades
    """

    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = Path(root_dir)
        self.split = split.lower()

        if self.split not in ["train", "validation"]:
            raise ValueError(
                f"Split debe ser 'train' o 'validation', recibido: {split}"
            )

        self.split_dir = self.root_dir / self.split

        self.rgb_dir = self.split_dir / "rgb"
        self.depth_gt_dir = self.split_dir / "depth_gt"
        self.depth_input_dir = self.split_dir / "depth_input"

        self._verify_directories()
        self.image_files = sorted(self._get_image_files())

        logger.info(
            f"Cargador SUN-RGB-D inicializado: {len(self.image_files)} imágenes en {split}"
        )

    def _verify_directories(self) -> None:
        required_dirs = [self.rgb_dir, self.depth_gt_dir, self.depth_input_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directorio no encontrado: {dir_path}")

    def _get_image_files(self) -> List[Path]:
        if not self.rgb_dir.exists():
            return []

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_files = []

        for ext in extensions:
            image_files.extend(list(self.rgb_dir.glob(ext)))

        return sorted(image_files)

    def _get_image_id(self, image_path: Path) -> str:
        return image_path.stem

    def _find_depth_file(self, depth_dir: Path, image_id: str) -> Optional[Path]:
        """Busca archivo de profundidad con múltiples extensiones."""
        extensions = [".png", ".jpg", ".jpeg", ".npy", ".PNG", ".JPG", ".JPEG", ".NPY"]

        for ext in extensions:
            depth_path = depth_dir / f"{image_id}{ext}"
            if depth_path.exists():
                return depth_path

        # Buscar con patrón glob
        for depth_file in depth_dir.glob(f"{image_id}*"):
            return depth_file

        return None

    def load_image(self, idx: int) -> np.ndarray:
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Error al cargar imagen: {image_path}")

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_depth(self, idx: int, depth_dir: Path) -> np.ndarray:
        """Carga profundidad soporte .npy y .png"""
        image_id = self._get_image_id(self.image_files[idx])
        depth_path = self._find_depth_file(depth_dir, image_id)

        if depth_path is None:
            raise ValueError(f"Depth no encontrado para: {image_id}")

        # Cargar según extensión
        if depth_path.suffix.lower() == ".npy":
            depth = np.load(str(depth_path))
        else:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        if depth is None:
            raise ValueError(f"Error al cargar depth: {depth_path}")

        return depth

    def load_depth_gt(self, idx: int) -> np.ndarray:
        return self.load_depth(idx, self.depth_gt_dir)

    def load_depth_input(self, idx: int) -> np.ndarray:
        return self.load_depth(idx, self.depth_input_dir)

    def load_item(self, idx: int) -> Dict:
        return {
            "image": self.load_image(idx),
            "depth_gt": self.load_depth_gt(idx),
            "depth_input": self.load_depth_input(idx),
            "image_path": self.image_files[idx],
            "image_id": self._get_image_id(self.image_files[idx]),
        }

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        return self.load_item(idx)
