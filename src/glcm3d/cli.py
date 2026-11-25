"""Command-line interface for the ``glcm3d`` package."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Sequence

from .config import ALL_FEATURES, GLCMConfig
from .guards import get_available_mem_bytes, get_free_disk_bytes
from .pipeline import run_pipeline, run_single_file


def _make_plain_progress(width: int = 24):
    """Fallback plain-text terminal progress callback when ``rich`` is unavailable."""

    def _cb(current: float, total: float, message: str) -> None:
        pct = (100.0 * current / total) if total > 0 else 100.0
        pct = min(100.0, max(0.0, pct))
        filled = int(round(width * pct / 100.0))
        bar = "#" * filled + "-" * (width - filled)
        print(f"  [{bar}] {pct:5.1f}% | {message}", flush=True)

    return _cb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glcm3d",
        description="Compute windowed 3D GLCM texture maps from multi-page TIFF stacks.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input .tif file or directory of .tif stacks",
    )
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


def _run_with_rich(
    in_path: Path,
    out_path: Path,
    cfg: GLCMConfig,
    *,
    list_file: str | None,
    interactive: bool,
) -> dict:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    console = Console()
    mode = "Single file" if in_path.is_file() else "Directory batch"
    avail_gb = get_available_mem_bytes() / (2**30)
    disk_gb = get_free_disk_bytes(out_path) / (2**30)

    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan", justify="right")
    info.add_column()
    info.add_row("Mode:", mode)
    info.add_row("Input:", str(in_path))
    info.add_row("Output:", str(out_path))
    info.add_row(
        "Params:",
        f"D={cfg.d}  |  R={list(cfg.neighbor_sizes)}  |  Q={cfg.quant_level}  |  "
        f"{len(cfg.features)} feature(s)",
    )
    info.add_row("Resources:", f"RAM avail {avail_gb:.1f} GB  |  Disk free {disk_gb:.1f} GB")
    console.print(Panel(info, title="[bold]glcm3D[/bold]", border_style="cyan", expand=False))

    t0 = time.monotonic()

    if interactive:
        # Interactive prompt reads from stdin, so avoid holding a Live progress display open
        progress_cb = _make_plain_progress()
        summary = run_pipeline(
            in_path,
            out_path,
            list_file=list_file,
            config=cfg,
            interactive=True,
            verbose=True,
            progress_callback=progress_cb,
        )
    else:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=32),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        with progress:
            task_id = progress.add_task("Initializing...", total=100.0)

            def _rich_cb(current: float, total: float, message: str) -> None:
                pct = (100.0 * current / total) if total > 0 else 100.0
                pct = min(100.0, max(0.0, pct))
                progress.update(task_id, completed=pct, description=message)

            if in_path.is_file():
                summary = run_single_file(
                    in_path,
                    out_path,
                    config=cfg,
                    verbose=False,
                    progress_callback=_rich_cb,
                )
            else:
                summary = run_pipeline(
                    in_path,
                    out_path,
                    list_file=list_file,
                    config=cfg,
                    interactive=False,
                    verbose=False,
                    progress_callback=_rich_cb,
                )
            if not summary["stopped_early"]:
                progress.update(task_id, completed=100.0, description="Complete")

    elapsed = time.monotonic() - t0
    res_table = Table(title="Run Summary", show_header=False, border_style="green" if not summary["stopped_early"] else "yellow")
    res_table.add_column("Metric", style="bold")
    res_table.add_column("Value")
    status_str = (
        "[bold yellow]Stopped early (low disk or cancelled)[/bold yellow]"
        if summary["stopped_early"]
        else "[bold green]Complete[/bold green]"
    )
    res_table.add_row("Status", status_str)
    irange = summary.get("intensity_range")
    if irange is not None:
        res_table.add_row("Intensity range", f"{irange[0]:g} .. {irange[1]:g}")
    res_table.add_row("Outputs written", str(len(summary["processed"])))
    res_table.add_row("Skipped", str(len(summary["skipped"])))
    res_table.add_row("Elapsed time", f"{elapsed:.2f} s")
    res_table.add_row("Output folder", str(out_path))
    console.print(res_table)
    return summary


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

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        import rich  # noqa: F401

        use_rich = True
    except ImportError:
        use_rich = False

    if use_rich:
        summary = _run_with_rich(
            in_path,
            out_path,
            cfg,
            list_file=args.list_file,
            interactive=args.interactive,
        )
    else:
        progress_cb = _make_plain_progress()
        if in_path.is_file():
            summary = run_single_file(
                in_path,
                out_path,
                config=cfg,
                progress_callback=progress_cb,
            )
        else:
            summary = run_pipeline(
                in_path,
                out_path,
                list_file=args.list_file,
                config=cfg,
                interactive=args.interactive,
                progress_callback=progress_cb,
            )
    return 1 if summary["stopped_early"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
