"""Command-line interface for the ``glcm3d`` package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import ALL_FEATURES, GLCMConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glcm3d",
        description="Compute windowed 3D GLCM texture maps from multi-page TIFF stacks.",
    )
    parser.add_argument("--input", "-i", required=True, help="Directory of input .tif stacks")
    parser.add_argument("--output", "-o", required=True, help="Directory for output .tif files")
    parser.add_argument(
        "--list-file",
        default=None,
        help="Path to resume filelist.txt (default: <output>/../filelist.txt)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to config.json (reads the 'glcm' section)",
    )
    parser.add_argument("--d", type=int, default=None, help="Offset radius D (default: 1)")
    parser.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=None,
        help="Window radii R (default: 1 3)",
    )
    parser.add_argument(
        "--quant-level",
        type=int,
        default=None,
        help="Number of gray levels (default: 8)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=ALL_FEATURES,
        default=None,
        help="Subset of GLCM features to compute (default: all 23)",
    )
    parser.add_argument(
        "--num-z",
        type=int,
        default=None,
        help="Number of Z slices per volume (default: all pages)",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=None,
        help="Minimum free disk space reserve in GB (default: 50)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable lossless Deflate TIFF compression",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for confirmation after printing the global intensity percentile range",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict = {}
    if args.d is not None:
        overrides["d"] = args.d
    if args.radii is not None:
        overrides["neighbor_sizes"] = tuple(args.radii)
    if args.quant_level is not None:
        overrides["quant_level"] = args.quant_level
    if args.features is not None:
        overrides["features"] = tuple(args.features)
    if args.num_z is not None:
        overrides["num_z_slices"] = args.num_z
    if args.min_free_disk_gb is not None:
        overrides["min_free_disk_gb"] = args.min_free_disk_gb
    if args.no_compress:
        overrides["compress"] = False

    if args.config:
        cfg = GLCMConfig.from_json(Path(args.config), overrides=overrides)
    else:
        cfg = GLCMConfig(**overrides)

    summary = run_pipeline(
        args.input,
        args.output,
        list_file=args.list_file,
        config=cfg,
        interactive=args.interactive,
    )
    return 1 if summary["stopped_early"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
