"""End-to-end batch pipeline for windowed 3D GLCM texture extraction."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .chunked import compute_chunked
from .config import GLCMConfig, ProgressCallback
from .core import quantize_volume
from .guards import (
    GLCMMemoryError,
    ensure_disk,
    get_available_mem_bytes,
    get_free_disk_bytes,
)
from .io import (
    append_to_file_list,
    clean_stale_part_files,
    compute_global_percentile_range,
    crop_to_foreground,
    load_file_list,
    make_out_name,
    read_tiff_volume,
    write_tiff_atomic,
)


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    list_file: str | Path | None = None,
    config: GLCMConfig | None = None,
    interactive: bool = False,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Process all ``*.tif`` stacks in ``input_dir`` and write 3D GLCM TIFFs.

    Returns a summary dictionary with ``processed``, ``skipped``,
    ``stopped_early``, and ``intensity_range``.
    """
    cfg = config or GLCMConfig()
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    list_path = Path(list_file) if list_file is not None else out_path.parent / "filelist.txt"

    tiff_paths = sorted(in_path.glob("*.tif"))
    if not tiff_paths:
        if verbose:
            print(f"No .tif files found in {in_path}")
        return {
            "processed": [],
            "skipped": [],
            "stopped_early": False,
            "intensity_range": None,
        }

    done_set = load_file_list(list_path)

    def is_done(name: str) -> bool:
        return (name in done_set) or (out_path / name).is_file()

    if verbose:
        print(f"{len(done_set)} outputs already listed in {list_path}", flush=True)

    stale = clean_stale_part_files(out_path)
    if verbose:
        for p in stale:
            print(f"Deleted stale partial file {p.name}", flush=True)

    # Global foreground percentile range over ALL input files
    n_files = len(tiff_paths)
    scan_cb = (
        (lambda cur, tot, msg: progress_callback(0.0, float(n_files), msg))
        if progress_callback is not None
        else None
    )
    setmin, setmax = compute_global_percentile_range(
        tiff_paths,
        low_pct=cfg.low_percentile,
        high_pct=cfg.high_percentile,
        num_z_slices=cfg.num_z_slices,
        time_index=cfg.time_index,
        verbose=verbose,
        progress_callback=scan_cb,
    )
    if verbose:
        print(f"Full dataset percentile range: {setmin:g} .. {setmax:g}", flush=True)

    if interactive:
        ans = input("continue? 1-yes/0-no: ").strip()
        if ans == "0":
            return {
                "processed": [],
                "skipped": [],
                "stopped_early": True,
                "intensity_range": (setmin, setmax),
            }

    if verbose:
        print(
            f"RAM available: {get_available_mem_bytes() / 2**30:.1f} GB | "
            f"Disk free on outpath: {get_free_disk_bytes(out_path) / 2**30:.1f} GB",
            flush=True,
        )

    processed: list[str] = []
    skipped: list[str] = []
    stopped_early = False
    reserve_disk_bytes = cfg.min_free_disk_gb * (2**30)
    n_files = len(tiff_paths)
    n_radii = len(cfg.neighbor_sizes)

    if progress_callback is not None:
        progress_callback(
            0.0,
            float(n_files),
            f"Intensity range: {setmin:g} .. {setmax:g}",
        )

    for f_idx, tif_path in enumerate(tiff_paths):
        stem = tif_path.stem

        pending = any(
            not is_done(make_out_name(stem, prop, cfg.d, r, cfg.quant_level))
            for r in cfg.neighbor_sizes
            for prop in cfg.features
        )
        if not pending:
            if verbose:
                print(f"Skipping {stem} (all outputs already done)")
            skipped.append(stem)
            if progress_callback is not None:
                progress_callback(
                    float(f_idx + 1),
                    float(n_files),
                    f"Skipped {stem} (already done)",
                )
            continue

        if verbose:
            print(f"Loading [{f_idx + 1}/{n_files}] {tif_path.name}...", flush=True)
        vol = read_tiff_volume(
            tif_path,
            num_z_slices=cfg.num_z_slices,
            time_index=cfg.time_index,
        )
        orig_shape = vol.shape
        mask = vol > 0
        slices, crop_vol, crop_mask = crop_to_foreground(vol, mask)
        del vol, mask

        if slices is None or crop_vol is None or crop_mask is None:
            if verbose:
                print(f"No foreground in {stem}, skipping")
            skipped.append(stem)
            if progress_callback is not None:
                progress_callback(
                    float(f_idx + 1),
                    float(n_files),
                    f"Skipped {stem} (no foreground)",
                )
            continue

        si = quantize_volume(crop_vol, cfg.quant_level, (setmin, setmax))
        del crop_vol

        for r_idx, r in enumerate(cfg.neighbor_sizes):
            names = [
                make_out_name(stem, prop, cfg.d, r, cfg.quant_level)
                for prop in cfg.features
            ]
            todo_idx = [idx for idx, nm in enumerate(names) if not is_done(nm)]
            if not todo_idx:
                if verbose:
                    print(f"Skipping {stem} | R={r} (done)")
                if progress_callback is not None:
                    progress_callback(
                        float(f_idx) + float(r_idx + 1) / n_radii,
                        float(n_files),
                        f"Skipped {stem} | R={r} (done)",
                    )
                continue

            feats_todo = [cfg.features[idx] for idx in todo_idx]
            uncompressed_needed = len(todo_idx) * int(np.prod(orig_shape)) * 4
            if not ensure_disk(
                out_path, uncompressed_needed, reserve_disk_bytes, cfg.disk_wait_sec
            ):
                stopped_early = True
                break

            t0 = time.monotonic()
            if verbose:
                print(
                    f"Processing {stem} | NL={cfg.quant_level} | R={r} | "
                    f"{len(feats_todo)} feature(s)"
                )

            if progress_callback is not None:
                r_base = float(f_idx) + float(r_idx) / n_radii
                r_compute_span = 0.9 / n_radii

                def _rad_cb(
                    cur: float,
                    tot: float,
                    msg: str,
                    _base: float = r_base,
                    _span: float = r_compute_span,
                    _stem: str = stem,
                    _r: int = r,
                ) -> None:
                    frac = (cur / tot) if tot > 0 else 1.0
                    progress_callback(
                        _base + _span * frac,
                        float(n_files),
                        f"{_stem} | R={_r}: {msg}",
                    )

            else:
                _rad_cb = None

            try:
                texture = compute_chunked(
                    si,
                    crop_mask,
                    cfg.quant_level,
                    cfg.d,
                    r,
                    feats_todo,
                    cfg,
                    verbose=verbose,
                    progress_callback=_rad_cb,
                )
            except GLCMMemoryError as exc:
                if verbose:
                    print(f"WARNING: Skipping {stem} R={r}: {exc}")
                continue

            if verbose:
                print(
                    f"Total time (NL={cfg.quant_level}, R={r}): {time.monotonic() - t0:.2f} s"
                )

            n_todo = len(todo_idx)
            for k, orig_feat_idx in enumerate(todo_idx):
                out_name = names[orig_feat_idx]
                full_img = np.zeros(orig_shape, dtype=np.float32)
                full_img[slices] = texture[..., k]

                if not ensure_disk(
                    out_path, full_img.nbytes, reserve_disk_bytes, cfg.disk_wait_sec
                ):
                    stopped_early = True
                    break

                write_tiff_atomic(out_path / out_name, full_img, compress=cfg.compress)
                append_to_file_list(list_path, out_name)
                done_set.add(out_name)
                processed.append(out_name)
                if progress_callback is not None:
                    w_pos = float(f_idx) + (float(r_idx) + 0.9 + 0.1 * ((k + 1) / n_todo)) / n_radii
                    progress_callback(
                        w_pos,
                        float(n_files),
                        f"Wrote {out_name}",
                    )

            if stopped_early:
                break

        if stopped_early:
            break

    if progress_callback is not None and not stopped_early:
        progress_callback(
            float(n_files),
            float(n_files),
            f"Complete ({len(processed)} written, {len(skipped)} skipped)",
        )

    return {
        "processed": processed,
        "skipped": skipped,
        "stopped_early": stopped_early,
        "intensity_range": (setmin, setmax),
    }


def run_single_file(
    tif_path: str | Path,
    output_dir: str | Path,
    *,
    config: GLCMConfig | None = None,
    overwrite: bool = True,
    progress_callback: ProgressCallback | None = None,
    verbose: bool = True,
) -> dict:
    """Process a single ``.tif``/``.tiff`` stack and write 3D GLCM TIFF outputs.

    Does not read or write ``filelist.txt``. Returns a summary dictionary with
    ``processed``, ``skipped``, ``stopped_early``, and ``intensity_range``.
    """
    in_file = Path(tif_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Input TIFF file not found: {in_file}")
    if not in_file.is_file():
        raise ValueError(f"Expected a .tif/.tiff file, got: {in_file}")
    if in_file.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(
            f"Unsupported file extension '{in_file.suffix}'; expected .tif or .tiff"
        )

    out_path = Path(output_dir)
    if out_path.exists() and not out_path.is_dir():
        raise NotADirectoryError(
            f"Output path exists and is not a directory: {out_path}"
        )
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = config or GLCMConfig()
    stem = in_file.stem
    total = 1.0

    def _emit(current: float, msg: str) -> None:
        if progress_callback is not None:
            clamped = min(total, max(0.0, float(current)))
            progress_callback(clamped, total, msg)

    stale = clean_stale_part_files(out_path)
    if verbose:
        for p in stale:
            print(f"Deleted stale partial file {p.name}")

    _emit(0.02, f"Computing intensity range for {in_file.name}...")
    setmin, setmax = compute_global_percentile_range(
        [in_file],
        low_pct=cfg.low_percentile,
        high_pct=cfg.high_percentile,
        num_z_slices=cfg.num_z_slices,
        time_index=cfg.time_index,
    )
    if verbose:
        print(f"Intensity range for {in_file.name}: {setmin:g} .. {setmax:g}")

    if not overwrite:
        pending = any(
            not (out_path / make_out_name(stem, prop, cfg.d, r, cfg.quant_level)).is_file()
            for r in cfg.neighbor_sizes
            for prop in cfg.features
        )
        if not pending:
            if verbose:
                print(f"Skipping {stem} (all outputs already exist)")
            _emit(1.0, f"Skipped {stem} (all outputs already exist)")
            return {
                "processed": [],
                "skipped": [stem],
                "stopped_early": False,
                "intensity_range": (setmin, setmax),
            }

    _emit(0.05, f"Loading {in_file.name}...")
    vol = read_tiff_volume(
        in_file,
        num_z_slices=cfg.num_z_slices,
        time_index=cfg.time_index,
    )
    orig_shape = vol.shape
    mask = vol > 0
    slices, crop_vol, crop_mask = crop_to_foreground(vol, mask)
    del vol, mask

    if slices is None or crop_vol is None or crop_mask is None:
        if verbose:
            print(f"No foreground in {stem}, skipping")
        _emit(1.0, f"No foreground (> 0) found in {in_file.name}; nothing written")
        return {
            "processed": [],
            "skipped": [stem],
            "stopped_early": False,
            "intensity_range": (setmin, setmax),
        }

    si = quantize_volume(crop_vol, cfg.quant_level, (setmin, setmax))
    del crop_vol

    processed: list[str] = []
    skipped: list[str] = []
    stopped_early = False
    reserve_disk_bytes = cfg.min_free_disk_gb * (2**30)
    n_radii = len(cfg.neighbor_sizes)

    for r_idx, r in enumerate(cfg.neighbor_sizes):
        r_start = 0.05 + 0.95 * (r_idx / n_radii)
        r_compute_end = 0.05 + 0.95 * ((r_idx + 0.85) / n_radii)
        r_end = 0.05 + 0.95 * ((r_idx + 1.0) / n_radii)

        names = [
            make_out_name(stem, prop, cfg.d, r, cfg.quant_level)
            for prop in cfg.features
        ]
        if overwrite:
            todo_idx = list(range(len(names)))
        else:
            todo_idx = [
                idx for idx, nm in enumerate(names) if not (out_path / nm).is_file()
            ]

        if not todo_idx:
            if verbose:
                print(f"Skipping {stem} | R={r} (all outputs exist)")
            _emit(r_end, f"Skipped {stem} | R={r} (outputs exist)")
            continue

        feats_todo = [cfg.features[idx] for idx in todo_idx]
        uncompressed_needed = len(todo_idx) * int(np.prod(orig_shape)) * 4
        if not ensure_disk(
            out_path, uncompressed_needed, reserve_disk_bytes, cfg.disk_wait_sec
        ):
            stopped_early = True
            _emit(
                r_start,
                f"Stopped early: insufficient free disk space (reserve {cfg.min_free_disk_gb:g} GB)",
            )
            break

        t0 = time.monotonic()
        if verbose:
            print(
                f"Processing {stem} | NL={cfg.quant_level} | R={r} | "
                f"{len(feats_todo)} feature(s)"
            )

        def _rad_cb(
            cur: float,
            tot: float,
            msg: str,
            _start: float = r_start,
            _end: float = r_compute_end,
            _r: int = r,
        ) -> None:
            frac = (cur / tot) if tot > 0 else 1.0
            _emit(_start + (_end - _start) * frac, f"{stem} | R={_r}: {msg}")

        texture = compute_chunked(
            si,
            crop_mask,
            cfg.quant_level,
            cfg.d,
            r,
            feats_todo,
            cfg,
            verbose=verbose,
            progress_callback=_rad_cb if progress_callback is not None else None,
        )

        if verbose:
            print(
                f"Total time (NL={cfg.quant_level}, R={r}): {time.monotonic() - t0:.2f} s"
            )

        n_todo = len(todo_idx)
        for k, orig_feat_idx in enumerate(todo_idx):
            out_name = names[orig_feat_idx]
            full_img = np.zeros(orig_shape, dtype=np.float32)
            full_img[slices] = texture[..., k]

            if not ensure_disk(
                out_path, full_img.nbytes, reserve_disk_bytes, cfg.disk_wait_sec
            ):
                stopped_early = True
                _emit(
                    r_compute_end + (r_end - r_compute_end) * (k / n_todo),
                    f"Stopped early: insufficient free disk space (reserve {cfg.min_free_disk_gb:g} GB)",
                )
                break

            write_tiff_atomic(out_path / out_name, full_img, compress=cfg.compress)
            processed.append(out_name)
            _emit(
                r_compute_end + (r_end - r_compute_end) * ((k + 1) / n_todo),
                f"Wrote {out_name}",
            )

        if stopped_early:
            break

    if not stopped_early:
        _emit(1.0, f"Done: wrote {len(processed)} file(s).")

    return {
        "processed": processed,
        "skipped": skipped,
        "stopped_early": stopped_early,
        "intensity_range": (setmin, setmax),
    }
