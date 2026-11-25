"""Core mathematical routines for two-pass box-filter 3D GLCM features."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import ALL_FEATURES, MARGINAL_ONLY_FEATURES, ProgressCallback


def compute_offsets(d: int) -> np.ndarray:
    """Return all non-zero 3D integer offsets within rounded Euclidean radius ``d``.

    Matches MATLAB's ``round(sqrt(d1^2 + d2^2 + d3^2)) <= d`` (half-away-from-zero).
    For ``d = 1`` this yields 18 offsets (6 face + 12 edge neighbors).
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    grid = np.arange(-d, d + 1, dtype=np.int32)
    d1, d2, d3 = np.meshgrid(grid, grid, grid, indexing="ij")
    dist = np.sqrt(
        d1.astype(np.float64) ** 2
        + d2.astype(np.float64) ** 2
        + d3.astype(np.float64) ** 2
    )
    # Half-away-from-zero rounding to match MATLAB's round()
    inside = np.floor(dist + 0.5) <= d
    offsets = np.column_stack((d1[inside], d2[inside], d3[inside]))
    nonzero = np.any(offsets != 0, axis=1)
    return offsets[nonzero].astype(np.int32, copy=False)


def quantize_volume(
    vol: np.ndarray,
    nl: int,
    gl: tuple[float, float],
) -> np.ndarray:
    """Linearly quantize ``vol`` in float64 to integer levels ``1..nl``.

    Returns ``uint8`` when ``nl <= 255`` and ``uint16`` otherwise.
    """
    vol_f64 = np.asarray(vol, dtype=np.float64)
    gmin, gmax = float(gl[0]), float(gl[1])
    if gmax == gmin:
        si = np.ones(vol_f64.shape, dtype=np.float64)
    else:
        slope = (nl - 1) / (gmax - gmin)
        intercept = 1.0 - slope * gmin
        scaled = vol_f64 * slope + intercept
        # Half-away-from-zero rounding for non-negative mapped values (matches MATLAB)
        si = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))

    np.clip(si, 1.0, float(nl), out=si)
    dtype = np.uint8 if nl <= 255 else np.uint16
    return si.astype(dtype)


def _box_filter_1d(arr: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """1D zero-padded running sum over a window of width ``2*radius + 1``."""
    if radius == 0:
        return arr
    n = arr.shape[axis]
    win = 2 * radius + 1

    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(arr, pad_width, mode="constant", constant_values=0.0)
    cs = np.cumsum(padded, axis=axis, dtype=np.float64)

    pre_pad = [(0, 0)] * arr.ndim
    pre_pad[axis] = (1, 0)
    cs = np.pad(cs, pre_pad, mode="constant", constant_values=0.0)

    hi = [slice(None)] * arr.ndim
    lo = [slice(None)] * arr.ndim
    hi[axis] = slice(win, win + n)
    lo[axis] = slice(0, n)
    return cs[tuple(hi)] - cs[tuple(lo)]


def box_filter_3d(arr: np.ndarray, radius: int) -> np.ndarray:
    """Sum ``arr`` over a cubic window of radius ``radius`` (zero-padded).

    Computed via three separable 1D cumulative-sum passes in float64 (exact for
    integer counts) and cast to float32 (callers assert counts < 2^24).
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    out = np.asarray(arr, dtype=np.float64)
    if radius > 0:
        for axis in range(3):
            out = _box_filter_1d(out, radius, axis)
    return out.astype(np.float32)


def shift_volume(
    vol: np.ndarray,
    offset: Sequence[int],
    fill_val: int | bool = 0,
) -> np.ndarray:
    """Sample ``vol`` at ``pos + offset``, filling out-of-bounds entries with ``fill_val``."""
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {vol.shape}")
    out = np.full(vol.shape, fill_val, dtype=vol.dtype)
    dst_slices: list[slice] = []
    src_slices: list[slice] = []
    for size, delta in zip(vol.shape, offset):
        delta = int(delta)
        if abs(delta) >= size:
            return out
        if delta >= 0:
            dst_slices.append(slice(0, size - delta))
            src_slices.append(slice(delta, size))
        else:
            dst_slices.append(slice(-delta, size))
            src_slices.append(slice(0, size + delta))
    out[tuple(dst_slices)] = vol[tuple(src_slices)]
    return out


def _safe_xlogx(p: np.ndarray) -> np.ndarray:
    """Compute ``p * log(p)`` with exact convention ``0 * log(0) = 0``."""
    out = np.zeros_like(p, dtype=np.float64)
    pos = p > 0.0
    if np.any(pos):
        out[pos] = p[pos] * np.log(p[pos])
    return out


def _safe_xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute ``x * log(y)`` where ``x > 0`` and ``y > 0``, else ``0``."""
    out = np.zeros_like(x, dtype=np.float64)
    pos = (x > 0.0) & (y > 0.0)
    if np.any(pos):
        out[pos] = x[pos] * np.log(y[pos])
    return out


def _build_shifted(
    si: np.ndarray,
    mask: np.ndarray,
    offsets: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    si_shift: list[np.ndarray] = []
    valid_d: list[np.ndarray] = []
    for offset in offsets:
        si_shift.append(shift_volume(si, offset, 0))
        valid_d.append(mask & shift_volume(mask, offset, False))
    return si_shift, valid_d


def finalize_texture(texture: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace non-finite values with 0 and zero every voxel outside ``mask``."""
    inv_mask = ~mask
    for f in range(texture.shape[3]):
        sl = texture[..., f]
        sl[~np.isfinite(sl)] = 0.0
        sl[inv_mask] = 0.0
    return texture


def compute_windowed_glcm_features(
    si: np.ndarray,
    mask: np.ndarray,
    nl: int,
    d: int,
    radius: int,
    features: Sequence[str] = ALL_FEATURES,
    *,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """Compute windowed 3D GLCM features on an already-quantized volume ``si``.

    Parameters
    ----------
    si:
        3D integer array with quantized gray levels in ``1..nl``.
    mask:
        3D boolean foreground mask of the same shape as ``si``.
    nl:
        Number of quantization gray levels.
    d:
        Offset distance radius for :func:`compute_offsets`.
    radius:
        Cubic window half-width (window side length is ``2 * radius + 1``).
    features:
        Sequence of feature keys from :data:`ALL_FEATURES`.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(*si.shape, len(features))``.
    """
    if si.ndim != 3:
        raise ValueError(f"si must be 3D, got shape {si.shape}")
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != si.shape:
        raise ValueError(f"mask shape {mask_bool.shape} != si shape {si.shape}")

    feat_list = list(features)
    unknown = set(feat_list) - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")

    sz = si.shape
    offsets = compute_offsets(d)
    n_offsets = offsets.shape[0]
    win = 2 * radius + 1
    if (win**3) * n_offsets >= 2**24:
        raise ValueError(
            f"Window count {(win**3) * n_offsets} exceeds exact float32 integer range (2^24)"
        )

    si_shift, valid_d = _build_shifted(si, mask_bool, offsets)

    # ---- Total valid pair count per voxel ----
    valid_sum = np.zeros(sz, dtype=np.float32)
    for vd in valid_d:
        valid_sum += vd
    total_safe = box_filter_3d(valid_sum, radius).astype(np.float64)
    total_safe[total_safe == 0.0] = np.nan
    del valid_sum

    # ---- PASS 1: marginals (exact float32 counts) ----
    is_marginal_only = set(feat_list).issubset(MARGINAL_ONLY_FEATURES)
    total_steps = nl if is_marginal_only else 2 * nl

    p_x_counts = [np.zeros(sz, dtype=np.float32) for _ in range(nl)]
    p_y_counts = [np.zeros(sz, dtype=np.float32) for _ in range(nl)]
    p_xplusy_counts = [np.zeros(sz, dtype=np.float32) for _ in range(2 * nl - 1)]
    p_xminusy_counts = [np.zeros(sz, dtype=np.float32) for _ in range(nl)]

    for gi in range(1, nl + 1):
        mi = si == gi
        if np.any(mi):
            for gj in range(1, nl + 1):
                joint_sum = np.zeros(sz, dtype=np.float32)
                for idx_d in range(n_offsets):
                    joint_sum += valid_d[idx_d] & mi & (si_shift[idx_d] == gj)
                if not np.any(joint_sum):
                    continue
                local_count = box_filter_3d(joint_sum, radius)
                p_x_counts[gi - 1] += local_count
                p_y_counts[gj - 1] += local_count
                p_xplusy_counts[gi + gj - 2] += local_count
                p_xminusy_counts[abs(gi - gj)] += local_count
        if progress_callback is not None:
            progress_callback(
                float(gi),
                float(total_steps),
                f"Pass 1 (marginals) {gi}/{nl}",
            )

    del si_shift, valid_d

    # ---- Convert marginals to float64 probabilities ----
    p_x = [c.astype(np.float64) / total_safe for c in p_x_counts]
    p_y = [c.astype(np.float64) / total_safe for c in p_y_counts]
    del p_x_counts, p_y_counts

    u_x = np.zeros(sz, dtype=np.float64)
    u_y = np.zeros(sz, dtype=np.float64)
    for k in range(1, nl + 1):
        u_x += k * p_x[k - 1]
        u_y += k * p_y[k - 1]

    s_x2 = np.zeros(sz, dtype=np.float64)
    s_y2 = np.zeros(sz, dtype=np.float64)
    for k in range(1, nl + 1):
        s_x2 += ((k - u_x) ** 2) * p_x[k - 1]
        s_y2 += ((k - u_y) ** 2) * p_y[k - 1]
    s_x = np.sqrt(np.maximum(0.0, s_x2))
    s_y = np.sqrt(np.maximum(0.0, s_y2))
    del s_x2, s_y2

    hx = np.zeros(sz, dtype=np.float64)
    hy = np.zeros(sz, dtype=np.float64)
    for k in range(nl):
        hx -= _safe_xlogx(p_x[k])
        hy -= _safe_xlogx(p_y[k])

    hxy2 = np.zeros(sz, dtype=np.float64)
    for gi in range(nl):
        for gj in range(nl):
            pxpy = p_x[gi] * p_y[gj]
            hxy2 -= _safe_xlogx(pxpy)

    # Sum / difference statistics
    savgh = np.zeros(sz, dtype=np.float64)
    senth = np.zeros(sz, dtype=np.float64)
    p_xplusy = [c.astype(np.float64) / total_safe for c in p_xplusy_counts]
    del p_xplusy_counts
    for k_idx, pxy in enumerate(p_xplusy):
        s_val = k_idx + 2  # i + j ranges from 2 to 2*nl
        savgh += s_val * pxy
        senth -= _safe_xlogx(pxy)

    svarh = np.zeros(sz, dtype=np.float64)
    svarc = np.zeros(sz, dtype=np.float64)
    for k_idx, pxy in enumerate(p_xplusy):
        s_val = k_idx + 2
        svarh += ((s_val - senth) ** 2) * pxy
        svarc += ((s_val - savgh) ** 2) * pxy
    del p_xplusy

    denth = np.zeros(sz, dtype=np.float64)
    dvarh = np.zeros(sz, dtype=np.float64)
    for k_idx, cnt in enumerate(p_xminusy_counts):
        pxmy = cnt.astype(np.float64) / total_safe
        denth -= _safe_xlogx(pxmy)
        dvarh += (k_idx**2) * pxmy
    del p_xminusy_counts

    # ---- Assign marginal-only features ----
    n_feat = len(feat_list)
    texture = np.zeros((*sz, n_feat), dtype=np.float32)
    feat_idx = {name: i for i, name in enumerate(feat_list)}

    marginal_map = {
        "savgh": savgh,
        "senth": senth,
        "svarh": svarh,
        "svarc": svarc,
        "denth": denth,
        "dvarh": dvarh,
    }
    for name, arr in marginal_map.items():
        if name in feat_idx:
            texture[..., feat_idx[name]] = arr.astype(np.float32)
    del marginal_map, savgh, senth, svarh, svarc, denth, dvarh

    if is_marginal_only:
        return finalize_texture(texture, mask_bool)

    # ---- Allocate only needed joint accumulators ----
    has = feat_idx.__contains__
    need = {
        "autoc": has("autoc") or has("corrp"),
        "contr": has("contr"),
        "dissi": has("dissi"),
        "energ": has("energ"),
        "entro": has("entro") or has("inf1h") or has("inf2h"),
        "homom": has("homom"),
        "homop": has("homop"),
        "sosvh": has("sosvh"),
        "indnc": has("indnc"),
        "idmnc": has("idmnc"),
        "maxpr": has("maxpr"),
        "corrm": has("corrm"),
        "cprom": has("cprom"),
        "cshad": has("cshad"),
        "hxy1": has("inf1h"),
    }
    acc = {name: np.zeros(sz, dtype=np.float64) for name, flag in need.items() if flag}

    # ---- PASS 2: joint-dependent features ----
    si_shift, valid_d = _build_shifted(si, mask_bool, offsets)

    for gi in range(1, nl + 1):
        mi = si == gi
        if np.any(mi):
            for gj in range(1, nl + 1):
                joint_sum = np.zeros(sz, dtype=np.float32)
                for idx_d in range(n_offsets):
                    joint_sum += valid_d[idx_d] & mi & (si_shift[idx_d] == gj)
                if not np.any(joint_sum):
                    continue

                pij = box_filter_3d(joint_sum, radius).astype(np.float64) / total_safe
                pij = np.where(np.isnan(pij), 0.0, pij)
                dij = gi - gj
                abs_dij = abs(dij)

                if need["autoc"]:
                    acc["autoc"] += pij * (gi * gj)
                if need["contr"]:
                    acc["contr"] += pij * (dij * dij)
                if need["dissi"]:
                    acc["dissi"] += pij * abs_dij
                if need["energ"]:
                    acc["energ"] += pij * pij
                if need["entro"]:
                    acc["entro"] -= _safe_xlogx(pij)
                if need["homom"]:
                    acc["homom"] += pij / (1.0 + abs_dij)
                if need["homop"]:
                    acc["homop"] += pij / (1.0 + dij * dij)
                if need["sosvh"]:
                    # Haralick Sum of Squares Variance using gray-level mean u_y
                    acc["sosvh"] += pij * ((gj - u_y) ** 2)
                if need["indnc"]:
                    acc["indnc"] += pij / (1.0 + abs_dij / nl)
                if need["idmnc"]:
                    acc["idmnc"] += pij / (1.0 + (dij / nl) ** 2)
                if need["maxpr"]:
                    np.maximum(acc["maxpr"], pij, out=acc["maxpr"])
                if need["corrm"]:
                    acc["corrm"] += pij * (gi - u_x) * (gj - u_y)
                if need["cprom"]:
                    acc["cprom"] += pij * ((gi + gj - u_x - u_y) ** 4)
                if need["cshad"]:
                    acc["cshad"] += pij * ((gi + gj - u_x - u_y) ** 3)
                if need["hxy1"]:
                    acc["hxy1"] -= _safe_xlogy(pij, p_x[gi - 1] * p_y[gj - 1])
        if progress_callback is not None:
            progress_callback(
                float(nl + gi),
                float(total_steps),
                f"Pass 2 (joint features) {gi}/{nl}",
            )

    del si_shift, valid_d, p_x, p_y, total_safe

    simple = (
        "autoc",
        "contr",
        "dissi",
        "energ",
        "entro",
        "homom",
        "homop",
        "sosvh",
        "indnc",
        "idmnc",
        "maxpr",
        "cprom",
        "cshad",
    )
    for name in simple:
        if has(name):
            texture[..., feat_idx[name]] = acc[name].astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        if has("corrm"):
            texture[..., feat_idx["corrm"]] = (acc["corrm"] / (s_x * s_y)).astype(
                np.float32
            )
        if has("corrp"):
            texture[..., feat_idx["corrp"]] = (
                (acc["autoc"] - u_x * u_y) / (s_x * s_y)
            ).astype(np.float32)
        if has("inf1h"):
            denom = np.maximum(hx, hy)
            texture[..., feat_idx["inf1h"]] = (
                (acc["entro"] - acc["hxy1"]) / denom
            ).astype(np.float32)
        if has("inf2h"):
            val = np.maximum(0.0, 1.0 - np.exp(-2.0 * (hxy2 - acc["entro"])))
            texture[..., feat_idx["inf2h"]] = np.sqrt(val).astype(np.float32)

    return finalize_texture(texture, mask_bool)
