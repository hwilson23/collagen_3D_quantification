"""Memory-aware slab-chunked driver for 3D GLCM feature extraction."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .config import ALL_FEATURES, GLCMConfig, ProgressCallback
from .core import compute_offsets, compute_windowed_glcm_features
from .guards import (
    GLCMMemoryError,
    ensure_memory,
    estimate_bytes_per_voxel,
    get_available_mem_bytes,
)


def compute_chunked(
    si: np.ndarray,
    mask: np.ndarray,
    nl: int,
    d: int,
    radius: int,
    features: Sequence[str] = ALL_FEATURES,
    config: GLCMConfig | None = None,
    *,
    max_slab_override: int | None = None,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Compute windowed 3D GLCM features in memory-safe slabs along the longest axis.

    Each slab is padded with a halo of ``radius + d`` voxels on both sides so
    that every voxel in the core sees its full neighborhood (identical to
    whole-volume processing).
    """
    if si.ndim != 3:
        raise ValueError(f"si must be 3D, got shape {si.shape}")
    if si.size == 0:
        raise ValueError(f"si must not be empty, got shape {si.shape}")

    cfg = config or GLCMConfig()
    feat_list = list(features)
    sz = si.shape
    n_feat = len(feat_list)
    n_off = compute_offsets(d).shape[0]
    halo = radius + d
    bpv = estimate_bytes_per_voxel(nl, n_off, n_feat) * 1.25  # 25% safety margin
    min_free_bytes = cfg.min_free_gb * (2**30)

    avail = get_available_mem_bytes()
    if math.isfinite(avail):
        budget = min(cfg.mem_fraction * avail, avail - min_free_bytes)
    else:
        budget = float("inf")

    tex_bytes = 4 * n_feat * si.size
    if tex_bytes > budget:
        raise GLCMMemoryError(
            f"Result array alone needs {tex_bytes / 2**30:.1f} GB but budget is "
            f"{budget / 2**30:.1f} GB; request fewer features or crop the volume."
        )

    texture = np.zeros((*sz, n_feat), dtype=np.float32)
    budget = budget - tex_bytes

    cdim = int(np.argmax(sz))
    n = sz[cdim]
    plane_vox = si.size // n

    if max_slab_override is not None:
        core = max(1, min(int(max_slab_override), n))
    elif math.isfinite(budget):
        max_slab = int(math.floor(budget / (bpv * plane_vox)))
        core = min(max_slab - 2 * halo, n)
        if core < 1:
            min_slab_gb = bpv * plane_vox * (2 * halo + 1) / (2**30)
            raise GLCMMemoryError(
                f"Even a minimal slab ({2 * halo + 1} planes) needs ~{min_slab_gb:.1f} GB "
                f"but budget is {budget / 2**30:.1f} GB; lower quant_level or R."
            )
    else:
        core = n

    if verbose:
        print(
            f"  memory: avail {avail / 2**30:.1f} GB, budget {budget / 2**30:.1f} GB "
            f"-> slabs of {core} planes (+/-{halo} halo) along axis {cdim}",
            flush=True,
        )

    mask_bool = np.asarray(mask, dtype=bool)
    n_slabs = math.ceil(n / core)
    for s_idx, start in enumerate(range(0, n, core)):
        end = min(start + core, n)
        lo = max(0, start - halo)
        hi = min(n, end + halo)

        idx_core = [slice(None)] * 3
        idx_core[cdim] = slice(start, end)
        if not np.any(mask_bool[tuple(idx_core)]):
            if progress_callback is not None:
                progress_callback(
                    float(s_idx + 1),
                    float(n_slabs),
                    f"Slab {s_idx + 1}/{n_slabs} (empty, skipped)",
                )
            continue

        idx_slab = [slice(None)] * 3
        idx_slab[cdim] = slice(lo, hi)
        sub_si = si[tuple(idx_slab)]
        sub_mask = mask_bool[tuple(idx_slab)]

        if max_slab_override is None:
            ensure_memory(bpv * sub_si.size, min_free_bytes, cfg.max_wait_sec)

        if progress_callback is not None:

            def _slab_cb(
                cur: float,
                tot: float,
                msg: str,
                _s_idx: int = s_idx,
            ) -> None:
                frac = (cur / tot) if tot > 0 else 1.0
                progress_callback(
                    float(_s_idx) + frac,
                    float(n_slabs),
                    f"Slab {_s_idx + 1}/{n_slabs}: {msg}",
                )

        else:
            _slab_cb = None

        sub_tex = compute_windowed_glcm_features(
            sub_si,
            sub_mask,
            nl,
            d,
            radius,
            feat_list,
            progress_callback=_slab_cb,
        )

        idx_src = [slice(None)] * 4
        idx_src[cdim] = slice(start - lo, end - lo)
        idx_dst = [slice(None)] * 4
        idx_dst[cdim] = slice(start, end)
        texture[tuple(idx_dst)] = sub_tex[tuple(idx_src)]

    return texture
