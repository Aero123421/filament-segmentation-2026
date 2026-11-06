"""Rotation-normalised crops around one instance, for the stage-2 refiner.

Each stage-1 instance is cut out along its principal axis into a fixed
``CROP_W x CROP_H`` (512 x 160) canvas: native resolution when the padded
instance fits, otherwise scaled down (never up).  The four uint8 channels are

0. the grayscale image,
1. the stage-1 probability (x255),
2. this instance's mask (0/255),
3. the masks of all other instances (0/255).

The idea of a rotation-normalised second stage follows a public competition
notebook (see README); this implementation is original.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CROP_W = 512  # along the principal axis
CROP_H = 160  # across it

__all__ = ["CROP_H", "CROP_W", "CropGeom", "crop_geometry", "extract_crop", "extract_target", "paste_crop"]


@dataclass
class CropGeom:
    """Placement of a crop in the full image.

    Attributes:
        cx, cy: Instance centroid (pixels).
        theta: Principal-axis angle in image ``(x, y)`` coordinates (radians).
        scale: Crop pixels per image pixel (at most 1).
    """

    cx: float
    cy: float
    theta: float
    scale: float

    def affine_src_to_crop(self) -> np.ndarray:
        """2x3 affine matrix mapping image coordinates to crop coordinates."""
        along = np.array([np.cos(self.theta), np.sin(self.theta)])
        across = np.array([-np.sin(self.theta), np.cos(self.theta)])
        centre = np.array([self.cx, self.cy])
        s = self.scale
        matrix = np.zeros((2, 3), np.float64)
        matrix[0, :2] = s * along
        matrix[0, 2] = CROP_W / 2.0 - s * float(along @ centre)
        matrix[1, :2] = s * across
        matrix[1, 2] = CROP_H / 2.0 - s * float(across @ centre)
        return matrix.astype(np.float32)


def crop_geometry(seed_mask: np.ndarray) -> CropGeom | None:
    """Principal-axis crop geometry of one boolean instance mask (``None`` if < 10 px)."""
    ys, xs = np.nonzero(seed_mask)
    if len(xs) < 10:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
    _, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, -1]  # direction of largest variance, (x, y)
    theta = float(np.arctan2(axis[1], axis[0]))
    along = np.array([np.cos(theta), np.sin(theta)])
    across = np.array([-np.sin(theta), np.cos(theta)])
    u = dx * along[0] + dy * along[1]
    v = dx * across[0] + dy * across[1]
    length = float(u.max() - u.min()) if len(u) else 0.0
    width = float(v.max() - v.min()) if len(v) else 0.0
    # The crop must show the instance plus context: +60 % along, 3x across.
    need_w = length * 1.6 + 32.0
    need_h = max(width * 3.0, width + 48.0) + 16.0
    scale = min(1.0, CROP_W / max(need_w, 1.0), CROP_H / max(need_h, 1.0))
    return CropGeom(cx=cx, cy=cy, theta=theta, scale=float(scale))


def _warp(src: np.ndarray, matrix: np.ndarray, interpolation: int) -> np.ndarray:
    return cv2.warpAffine(
        src, matrix, (CROP_W, CROP_H), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def extract_crop(
    image: np.ndarray, prob: np.ndarray, seed: np.ndarray, others: np.ndarray, geom: CropGeom
) -> np.ndarray:
    """Return the ``(4, CROP_H, CROP_W)`` uint8 refiner input for one instance.

    Args:
        image: uint8 grayscale image.
        prob: float32 stage-1 probability in [0, 1].
        seed: Boolean mask of this instance.
        others: Boolean mask of all other instances.
        geom: Crop placement from :func:`crop_geometry`.
    """
    matrix = geom.affine_src_to_crop()
    channels = [
        _warp(image, matrix, cv2.INTER_LINEAR),
        _warp((np.clip(prob, 0, 1) * 255).astype(np.uint8), matrix, cv2.INTER_LINEAR),
        _warp(seed.astype(np.uint8) * 255, matrix, cv2.INTER_NEAREST),
        _warp(others.astype(np.uint8) * 255, matrix, cv2.INTER_NEAREST),
    ]
    return np.stack(channels).astype(np.uint8)


def extract_target(target_soft: np.ndarray, geom: CropGeom) -> np.ndarray:
    """Warp a float [0, 1] full-size training target into crop coordinates (uint8)."""
    matrix = geom.affine_src_to_crop()
    return _warp((np.clip(target_soft, 0, 1) * 255).astype(np.uint8), matrix, cv2.INTER_LINEAR).astype(np.uint8)


def paste_crop(crop_prob: np.ndarray, geom: CropGeom, full_shape: tuple[int, int] = (2048, 2048)) -> np.ndarray:
    """Warp a ``(CROP_H, CROP_W)`` float probability back to image coordinates."""
    matrix = geom.affine_src_to_crop()
    out = cv2.warpAffine(
        crop_prob.astype(np.float32),
        matrix,
        (full_shape[1], full_shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out.astype(np.float32)
