"""Optional stage-2 refiner: re-segments each instance in a rotation-normalised crop.

Input: the stage-1 instance labels and probability map; output: refined,
still disjoint labels at 2048x2048.  Enabled by ``refiner_enabled`` in
``pipeline_config.json`` (this package is only shipped when it is enabled).
"""

from .crops import CROP_H, CROP_W, CropGeom, crop_geometry, extract_crop, paste_crop
from .infer import RefinerEnsemble, load_refiners, refine_labels

__all__ = [
    "CROP_H",
    "CROP_W",
    "CropGeom",
    "RefinerEnsemble",
    "crop_geometry",
    "extract_crop",
    "load_refiners",
    "paste_crop",
    "refine_labels",
]
