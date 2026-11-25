"""TIFF I/O, foreground cropping, global percentile range, and resume file list."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import tifffile


def make_out_name(stem: str, prop: str, d: int, radius: int, quant_level: int) -> str:
    """Build output filename matching ``glcm3D_attempt.m`` conventions."""
    return f"{stem}_{prop}_3D_D{d}_R{radius}_Q{quant_level}_raw.tif"


def load_file_list(list_file: str | Path) -> set[str]:
    """Load newline-delimited completed output names from ``list_file``."""
    p = Path(list_file)
    if not p.is_file():
        return set()
    lines = p.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip()}


def append_to_file_list(list_file: str | Path, name: str) -> None:
    """Append a completed output filename to ``list_file``."""
    p = Path(list_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(f"{name}\n")


def clean_stale_part_files(out_dir: str | Path) -> list[Path]:
    """Delete leftover ``*.part`` files from previous crashed runs."""
    removed: list[Path] = []
    out_path = Path(out_dir)
    if not out_path.is_dir():
        return removed
    for part in sorted(out_path.glob("*.part")):
        part.unlink()
        removed.append(part)
    return removed


def read_tiff_volume(
    path: str | Path,
    num_z_slices: int | None = None,
    time_index: int = 1,
) -> np.ndarray:
    """Read a 3D multi-page TIFF and return an array in ``(Y, X, Z)`` order.

    ``tifffile`` loads multi-page stacks as ``(Z, Y, X)``; this function moves
    the Z axis to the last dimension so internal coordinates match MATLAB's
    ``(row, col, z)``.
    """
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D TIFF stack at {path}, got shape {arr.shape}")

    if num_z_slices is not None:
        start = (time_index - 1) * num_z_slices
        end = start + num_z_slices
        if end > arr.shape[0]:
            raise ValueError(
                f"TIFF {path} has {arr.shape[0]} pages, cannot slice [{start}:{end}]"
            )
        arr = arr[start:end]

    # (Z, Y, X) -> (Y, X, Z)
    return np.moveaxis(arr, 0, -1)


def write_tiff_stack(
    path: str | Path,
    vol_yxz: np.ndarray,
    compress: bool = True,
) -> None:
    """Write a ``(Y, X, Z)`` float32 volume as a multi-page ``(Z, Y, X)`` TIFF."""
    vol_zyx = np.ascontiguousarray(np.moveaxis(vol_yxz, -1, 0), dtype=np.float32)
    compression = "adobe_deflate" if compress else None
    tifffile.imwrite(str(path), vol_zyx, compression=compression)


def write_tiff_atomic(
    final_path: str | Path,
    vol_yxz: np.ndarray,
    compress: bool = True,
) -> None:
    """Write a TIFF to ``<final_path>.part`` and atomically rename on success."""
    final = Path(final_path)
    tmp = Path(f"{final}.part")
    try:
        write_tiff_stack(tmp, vol_yxz, compress=compress)
        os.replace(tmp, final)
    except Exception:
        if tmp.is_file():
            tmp.unlink()
        raise


def crop_to_foreground(
    vol: np.ndarray,
    mask: np.ndarray,
) -> tuple[tuple[slice, slice, slice] | None, np.ndarray | None, np.ndarray | None]:
    """Crop ``vol`` and ``mask`` to the tight 3D bounding box of ``mask``."""
    rr = np.flatnonzero(np.any(mask, axis=(1, 2)))
    if rr.size == 0:
        return None, None, None
    cc = np.flatnonzero(np.any(mask, axis=(0, 2)))
    zz = np.flatnonzero(np.any(mask, axis=(0, 1)))
    slices = (
        slice(int(rr[0]), int(rr[-1]) + 1),
        slice(int(cc[0]), int(cc[-1]) + 1),
        slice(int(zz[0]), int(zz[-1]) + 1),
    )
    return slices, vol[slices], mask[slices]


def compute_global_percentile_range(
    tiff_paths: Sequence[str | Path],
    *,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
    num_z_slices: int | None = None,
    time_index: int = 1,
    verbose: bool = False,
    progress_callback: Callable[[float, float, str], None] | None = None,
) -> tuple[float, float]:
    """Compute foreground (``> 0``) intensity percentiles across all input TIFFs.

    Uses an exact 16-bit histogram (``0..65535``) for uint8/uint16 integer stacks
    (fast and O(1) memory, matching ``glcm3D_attempt.m``), and falls back to
    ``np.percentile`` for floating-point or >16-bit stacks.
    """
    if not tiff_paths:
        raise ValueError("tiff_paths must not be empty")

    hist_counts = np.zeros(65536, dtype=np.int64)
    fallback_chunks: list[np.ndarray] | None = None
    n_files = len(tiff_paths)

    for idx, p in enumerate(tiff_paths, start=1):
        name = Path(p).name
        if verbose:
            print(
                f"Scanning [{idx}/{n_files}] {name} for global intensity range...",
                flush=True,
            )
        if progress_callback is not None:
            progress_callback(
                float(idx - 1),
                float(n_files),
                f"Scanning [{idx}/{n_files}] {name} for intensity range",
            )
        vol = read_tiff_volume(p, num_z_slices=num_z_slices, time_index=time_index)
        fg = vol[vol > 0]
        if fg.size == 0:
            continue

        if fallback_chunks is None and fg.dtype in (np.uint8, np.uint16):
            hist_counts += np.bincount(fg.ravel(), minlength=65536)
        else:
            if fallback_chunks is None:
                # Convert any accumulated histogram counts if mixed dtypes occur
                fallback_chunks = []
                nz = np.flatnonzero(hist_counts)
                if nz.size > 0:
                    fallback_chunks.append(np.repeat(nz.astype(np.float64), hist_counts[nz]))
            fallback_chunks.append(fg.astype(np.float64, copy=False))

    if fallback_chunks is not None:
        if not fallback_chunks:
            return 0.0, 0.0
        all_fg = np.concatenate(fallback_chunks)
        lo, hi = np.percentile(all_fg, [low_pct, high_pct])
        return float(lo), float(hi)

    total = int(hist_counts.sum())
    if total == 0:
        return 0.0, 0.0

    cdf = np.cumsum(hist_counts, dtype=np.float64) / total
    low_val = int(np.searchsorted(cdf, low_pct / 100.0, side="left"))
    high_val = int(np.searchsorted(cdf, high_pct / 100.0, side="left"))
    return float(low_val), float(high_val)
