"""
CurveAlign/ctFIRE Z-Stack Parser
=================================
Parses raw fiber measurement CSVs (HistANG, HistLEN, HistSTR, HistWID)
and overlay TIFs from ctFIRE output across multiple z-stacks and slices.

Each Hist*.csv contains a single column of raw fiber measurements (one row
per fiber). Slice numbers are mapped to physical z-depth using --z_step.

File naming convention:
    Hist<TYPE>_ctFIRE_<stack>_s<slice>.csv   e.g. HistANG_ctFIRE_bkwshg_s71.csv
    OL_ctFIRE_<stack>_s<slice>.tif            e.g. OL_ctFIRE_bkwshg_s17.tif


"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace
import tifffile
from collections import defaultdict


# ── Config ────────────────────────────────────────────────────────────────────

HIST_TYPES = ["HistANG", "HistLEN", "HistSTR", "HistWID"]

HIST_LABELS = {
    "HistANG": ("Fiber Angle",        "°"),
    "HistLEN": ("Fiber Length",       "µm"),
    "HistSTR": ("Fiber Straightness", ""),
    "HistWID": ("Fiber Width",        "µm"),
}

STACK_COLORS = {
    "bkwshg": "#2196F3",
    "flu":    "#E91E63",
    "fwdshg": "#4CAF50",
    "appliedMASKmean_bkwshg_8bit": "#2196F3",
    "appliedMASKmean_flu_8bit":    "#E91E63",
    "appliedMASKmean_fwdshg_8bit": "#4CAF50",
}
DEFAULT_COLOR = "#FF9800"


# ── File discovery ────────────────────────────────────────────────────────────

def discover_files(input_dir):
    """
    Returns:
        hist_files[stack][hist_type][slice] = path
        ol_files  [stack][slice]            = path
    """
    hist_pat = re.compile(r"(Hist\w+)_ctFIRE_(.+)_s(\d+)\.csv$", re.IGNORECASE)
    ol_pat   = re.compile(r"OL_ctFIRE_(.+)_s(\d+)\.tif$",         re.IGNORECASE)

    hist_files = defaultdict(lambda: defaultdict(dict))
    ol_files   = defaultdict(dict)

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)

        m = hist_pat.match(fname)
        if m:
            htype, stack, slc = m.group(1), m.group(2), int(m.group(3))
            hist_files[stack][htype][slc] = fpath
            continue

        m = ol_pat.match(fname)
        if m:
            stack, slc = m.group(1), int(m.group(2))
            ol_files[stack][slc] = fpath

    return hist_files, ol_files


# ── Load raw measurements ─────────────────────────────────────────────────────

def load_measurements(path):
    """Read a single-column CSV of raw fiber values. Returns a 1-D numpy array."""
    try:
        df = pd.read_csv(path, header=None, comment="#").dropna()
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return np.array([])


def load_all_measurements(hist_files):
    """Returns data[stack][hist_type][slice] = np.array of raw values."""
    data = defaultdict(lambda: defaultdict(dict))
    for stack, htypes in hist_files.items():
        for htype, slices in htypes.items():
            for slc, path in slices.items():
                vals = load_measurements(path)
                if len(vals) > 0:
                    data[stack][htype][slc] = vals
    return data


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(data, z_step):
    """
    Per-slice statistics for each (stack, hist_type, slice).
    z_depth = slice_number * z_step.

    Returns a long-form DataFrame with columns:
        stack, hist_type, slice, z_depth, n, mean, median, std, q25, q75
    """
    rows = []
    for stack, htypes in data.items():
        for htype, slices in htypes.items():
            for slc, vals in sorted(slices.items()):
                rows.append({
                    "stack":     stack,
                    "hist_type": htype,
                    "slice":     slc,
                    "z_depth":   slc * z_step,
                    "n":         len(vals),
                    "mean":      float(np.mean(vals)),
                    "median":    float(np.median(vals)),
                    "std":       float(np.std(vals)),
                    "q25":       float(np.percentile(vals, 25)),
                    "q75":       float(np.percentile(vals, 75)),
                })
    return pd.DataFrame(rows)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def stack_color(stack):
    return STACK_COLORS.get(stack, DEFAULT_COLOR)


def plot_mean_vs_z(stats_df, stacks, output_dir):
    """
    One figure per hist_type: mean +/- std vs z-depth, all stacks overlaid.
    X-axis is physical z-depth so slices map to real gel position.
    """
    for htype in stats_df["hist_type"].unique():
        sub = stats_df[stats_df["hist_type"] == htype]
        label, unit = HIST_LABELS.get(htype, (htype, ""))
        ylabel = f"{label} ({unit})" if unit else label

        fig, ax = plt.subplots(figsize=(8, 4))
        for stack in stacks:
            s = sub[sub["stack"] == stack].sort_values("z_depth")
            if s.empty:
                continue
            c = stack_color(stack)
            ax.plot(s["z_depth"], s["mean"], marker="o", label=stack, color=c, linewidth=2)
            ax.fill_between(s["z_depth"],
                            s["mean"] - s["std"],
                            s["mean"] + s["std"],
                            alpha=0.15, color=c)

        ax.set_xlabel("Z-depth (µm)", fontsize=11)
        ax.set_ylabel(f"Mean {ylabel}", fontsize=11)
        ax.set_title(f"{htype}  –  Mean ± Std vs Z-Depth", fontsize=12, fontweight="bold")
        ax.legend(title="Stack")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        out = os.path.join(output_dir, f"mean_z_{htype}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_violin_vs_z(data, stats_df, stacks, output_dir):
    """
    For each hist_type: violin plots at each z-depth slice showing full
    distribution shape across the gel. One subplot row per stack.
    """
    # Build a slice -> z_depth lookup from stats_df
    z_lookup = stats_df.set_index(["stack", "hist_type", "slice"])["z_depth"].to_dict()

    for htype in HIST_TYPES:
        active = [s for s in stacks if htype in data.get(s, {})]
        if not active:
            continue

        label, unit = HIST_LABELS.get(htype, (htype, ""))
        ylabel = f"{label} ({unit})" if unit else label

        n_rows  = len(active)
        n_slices = max(len(data[s][htype]) for s in active)
        fig_w   = max(8, n_slices * 0.7)
        fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, 4 * n_rows), squeeze=False)

        for ax, stack in zip(axes[:, 0], active):
            slices_dict = data[stack][htype]
            sorted_slcs = sorted(slices_dict.keys())
            positions   = [z_lookup.get((stack, htype, s), s) for s in sorted_slcs]

            # Violin width relative to z spacing
            if len(positions) > 1:
                width = (positions[-1] - positions[0]) / (len(positions) * 1.6)
            else:
                width = 1.0

            vp = ax.violinplot(
                [slices_dict[s] for s in sorted_slcs],
                positions=positions,
                widths=width,
                showmedians=True,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(stack_color(stack))
                body.set_alpha(0.5)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.5)

            ax.set_xlabel("Z-depth (µm)", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{stack}  –  {label} distribution across Z", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.3)

        fig.suptitle(f"{htype} – Full Distribution per Slice", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(output_dir, f"violin_z_{htype}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_fiber_count_vs_z(stats_df, stacks, output_dir):
    """Fiber detection count per slice — useful QC metric for z-stack quality."""
    for preferred in ["HistANG", "HistLEN", "HistSTR", "HistWID"]:
        sub = stats_df[stats_df["hist_type"] == preferred]
        if not sub.empty:
            break

    fig, ax = plt.subplots(figsize=(8, 4))
    for stack in stacks:
        s = sub[sub["stack"] == stack].sort_values("z_depth")
        if s.empty:
            continue
        ax.plot(s["z_depth"], s["n"], marker="s", label=stack,
                color=stack_color(stack), linewidth=2)

    ax.set_xlabel("Z-depth (µm)", fontsize=11)
    ax.set_ylabel("Fiber count per slice", fontsize=11)
    ax.set_title("Detected Fibers vs Z-Depth", fontsize=12, fontweight="bold")
    ax.legend(title="Stack")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(output_dir, "fiber_count_vs_z.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Overlay TIF stacking ──────────────────────────────────────────────────────

def build_overlay_zstacks(ol_files, stacks, output_dir):
    """
    For each stack, sort OL TIFs by slice number and write a single
    ImageJ-compatible multi-page z-stack TIFF.
    """
    for stack in stacks:
        slices = ol_files.get(stack)
        if not slices:
            print(f"  [SKIP] No overlay TIFs found for stack: {stack}")
            continue

        frames = []
        for slc in sorted(slices.keys()):
            try:
                img = tifffile.imread(slices[slc])
                frames.append(img)
            except Exception as e:
                print(f"  [WARN] {slices[slc]}: {e}")

        if not frames:
            continue

        # Pad to uniform spatial size if edge slices differ
        max_h = max(f.shape[0] for f in frames)
        max_w = max(f.shape[1] for f in frames)
        padded = []
        for f in frames:
            ph, pw = max_h - f.shape[0], max_w - f.shape[1]
            if ph > 0 or pw > 0:
                pad_width = ((0, ph), (0, pw)) + ((0, 0),) * (f.ndim - 2)
                f = np.pad(f, pad_width)
            padded.append(f)

        stack_arr = np.stack(padded, axis=0)
        axes_str  = "ZYXC" if stack_arr.ndim == 4 else "ZYX"
        out_path  = os.path.join(output_dir, f"OL_zstack_{stack}.tif")
        tifffile.imwrite(out_path, stack_arr, imagej=True)
        print(f"  Saved: {out_path}  ({len(padded)} slices, shape={stack_arr.shape})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    
    args = SimpleNamespace(
        input_dir  = "G:/FluorescentCollagen/20260427_flucol_ows3/flucol_crops/ctFIREout",
        output_dir = "G:/FluorescentCollagen/20260427_flucol_ows3/flucol_crops/ctFIREout/results_test_masked_min30",
        stacks     = ["1mgml", "2mgml", "3mgml"],
        z_step     = 1.0,
        no_overlay = False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stacks = [s for s in args.stacks]

    # 1. Discover
    print("\n[1/5] Discovering files …")
    hist_files, ol_files = discover_files(args.input_dir)
    print(list(hist_files.keys()))
    found = sorted(set(hist_files) | set(ol_files))
    print(f"  Stacks found: {found}")
    for s in found:
        htypes = {ht: len(sl) for ht, sl in hist_files.get(s, {}).items()}
        print(f"  {s:12s}  hist slices: {htypes}  |  overlays: {len(ol_files.get(s, {}))}")

    # 2. Load raw measurements
    print("\n[2/5] Loading raw measurements …")
    data = load_all_measurements(hist_files)

    # 3. Per-slice statistics with z_depth
    print("\n[3/5] Computing per-slice statistics …")
    stats_df = compute_stats(data, args.z_step)
    csv_path = os.path.join(args.output_dir, "ctfire_stats_per_slice.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Stack-level summary (collapsed across all slices)
    agg = (
        stats_df.groupby(["stack", "hist_type"])
        .apply(lambda g: pd.Series({
            "n_slices":     len(g),
            "total_fibers": int(g["n"].sum()),
            "mean":         float(np.average(g["mean"])),
            "std":          float(np.sqrt(np.average(g["std"]))),
            "median":       float(np.average(g["median"])),
        }))
        .reset_index()
    )
    agg_path = os.path.join(args.output_dir, "ctfire_stats_stack_summary.csv")
    agg.to_csv(agg_path, index=False)
    print(f"  Saved: {agg_path}")

    # 4. Plots
    print("\n[4/5] Generating plots …")
    active = [s for s in stacks if s in found] 
    print("Active stacks:", active)  # should show your 3 stacks
    plot_mean_vs_z(stats_df, active, args.output_dir)
    plot_violin_vs_z(data, stats_df, active, args.output_dir)
    plot_fiber_count_vs_z(stats_df, active, args.output_dir)


    # 5. Overlay TIF z-stacks
    if not args.no_overlay:
        print("\n[5/5] Building overlay z-stack TIFs …")
        build_overlay_zstacks(ol_files, active, args.output_dir)
    else:
        print("\n[5/5] Skipping overlay TIF build.")

    print("\n✓ Done. Results in:", args.output_dir)


if __name__ == "__main__":
    main()