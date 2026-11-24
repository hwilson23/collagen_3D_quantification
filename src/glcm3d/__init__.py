"""Windowed 3D GLCM texture feature extraction (box-filter, memory-safe).

Python port of ``src/calculate_glcm3D/glcm3D_attempt.m`` with cross-platform
RAM/disk guards and corrected Haralick variance (``sosvh``) and exact
``0 * log(0) = 0`` entropy handling.
"""

from .chunked import compute_chunked
from .config import ALL_FEATURES, MARGINAL_ONLY_FEATURES, GLCMConfig
from .core import (
    box_filter_3d,
    compute_offsets,
    compute_windowed_glcm_features,
    finalize_texture,
    quantize_volume,
    shift_volume,
)
from .guards import GLCMMemoryError
from .pipeline import run_pipeline, run_single_file

__all__ = [
    "ALL_FEATURES",
    "MARGINAL_ONLY_FEATURES",
    "GLCMConfig",
    "GLCMMemoryError",
    "box_filter_3d",
    "compute_chunked",
    "compute_offsets",
    "compute_windowed_glcm_features",
    "finalize_texture",
    "quantize_volume",
    "run_pipeline",
    "run_single_file",
    "shift_volume",
]
