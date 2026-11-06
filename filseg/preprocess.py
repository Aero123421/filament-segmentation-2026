"""Image input and crop geometry, shared by training and inference.

The competition images are 2048x2048 8-bit H-alpha full-disk JPEGs, read as
grayscale.  The solar disk fits inside the central 1888x1888 square, so the
network sees that square at native resolution: filaments are often only a few
pixels wide and would be lost by downscaling.  Probability maps predicted on
the crop are pasted back onto a zero 2048x2048 canvas, so the 80-pixel margin
outside the crop is always background.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FULL_SIZE = 2048
CROP_SIZE = 1888
CROP_OFFSET = (FULL_SIZE - CROP_SIZE) // 2  # 80 px on every side

__all__ = [
    "CROP_OFFSET",
    "CROP_SIZE",
    "FULL_SIZE",
    "crop_center",
    "paste_to_full",
    "prob_to_uint8",
    "read_image",
]


def read_image(path: str | Path) -> np.ndarray:
    """Read one competition image as a ``(2048, 2048)`` uint8 grayscale array.

    Raises:
        FileNotFoundError: If the file is missing or not a readable image.
        ValueError: If the image is not 2048x2048.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if image.shape != (FULL_SIZE, FULL_SIZE):
        raise ValueError(f"{path}: expected a {FULL_SIZE}x{FULL_SIZE} image, got {image.shape[1]}x{image.shape[0]}")
    return image


def crop_center(image: np.ndarray) -> np.ndarray:
    """Return the central 1888x1888 region of a 2048x2048 array (extra axes kept)."""
    if image.shape[:2] != (FULL_SIZE, FULL_SIZE):
        raise ValueError(f"expected a {FULL_SIZE}x{FULL_SIZE} array, got shape {image.shape}")
    window = slice(CROP_OFFSET, CROP_OFFSET + CROP_SIZE)
    return np.ascontiguousarray(image[window, window, ...])


def paste_to_full(prob_crop: np.ndarray) -> np.ndarray:
    """Paste a 1888x1888 probability map onto a zero 2048x2048 float32 canvas."""
    if prob_crop.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"expected a {CROP_SIZE}x{CROP_SIZE} map, got shape {prob_crop.shape}")
    canvas = np.zeros((FULL_SIZE, FULL_SIZE), dtype=np.float32)
    window = slice(CROP_OFFSET, CROP_OFFSET + CROP_SIZE)
    canvas[window, window] = prob_crop
    return canvas


def prob_to_uint8(prob: np.ndarray) -> np.ndarray:
    """Round a [0, 1] probability map to uint8 (``np.rint(p * 255)``).

    This is the storage format of the out-of-fold probability maps on which
    the post-processing parameters were tuned; decode with ``q / 255``.
    """
    return np.rint(np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8)
