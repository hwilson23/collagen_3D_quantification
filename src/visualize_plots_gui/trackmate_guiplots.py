"""
Collagen 3D Quantification Viewer
==================================

A Tkinter GUI for exploring per-cell 3D texture-analysis output alongside the
raw microscopy images they were computed from.

WHAT IT DOES
------------
1. Asks you to pick THREE folders:
     - Mask folder: the parent folder containing your
       "masks3d_per_cell_{pos}_{radius}um" subfolders.
     - Raw channel image folder: where your c1 (cell/nuclei) and sub7000
       (collagen) TIFFs live. Can be a completely different folder tree
       than the mask folder.
     - Dataframe folder: where your "current_final_dataframe_byslice_pos*
       trackmate*.csv" files live.
2. Loads every matching CSV in the dataframe folder and reduces it to one
   row per (position, timepoint, mask_type, cell, feature) -- the mean over
   z-slices -- exactly like your existing aggregation script does.
3. Populates dropdowns for Position, Cell (multi-select), and Feature based
   on what's actually present in the data / image folders.
4. Shows three images side-by-side for the selected position/cell/timepoint/
   z-slice: the collagen channel, the cell/nuclei channel, and a color-coded
   overlay of the r10/r20/r30 masks on top of the collagen channel.
5. Shows three line plots (one per radius: r10, r20, r30) of the selected
   texture feature vs. timepoint, with one line per selected cell. Checking
   "All cells" plots every cell found for that position.

FILE-NAMING ASSUMPTIONS (edit the CONFIG block below if these don't match)
----------------------------------------------------------------------------
  - Collagen channel files: contain "sub7000" in the filename, one file per
    (position, timepoint), TIFF shape (Z, Y, X).
  - Cell/nuclei channel files: contain "c1" in the filename, same shape
    convention as collagen.
  - Mask volumes: one file per (position, radius, cell), TIFF shape
    (T, Z, Y, X) -- matches the np.transpose(mask_img, (2,3,1,0)) used in
    your combining script, which turns (T,Z,Y,X) into (Y,X,Z,T).
  - Mask subfolder naming: "masks3d_per_cell_{pos}_{radius}um"
    (radius in {r10, r20, r30}).

REQUIREMENTS
------------
    pip install pandas numpy tifffile matplotlib

RUN
---
    python collagen_viewer.py
"""

import os
import re
import traceback

import numpy as np
import pandas as pd
import tifffile as tiff

import tkinter as tk
from tkinter import ttk, filedialog

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =====================================================================
# CONFIG -- edit these to match your actual file-naming conventions
# =====================================================================

POS_REGEX = re.compile(r'Pos(\d+)', re.IGNORECASE)
CELL_REGEX = re.compile(r'cell_(\d+)', re.IGNORECASE)
TIMEPOINT_REGEX = re.compile(r'_t(\d+)', re.IGNORECASE)

MASK_FOLDER_TEMPLATE = "masks3d_per_cell_{pos}_{radius}um"
RADII = ["r10", "r20", "r30"]

# Keywords used to find the collagen and cell/nuclei channel image files.
# Matching is case-insensitive substring matching against the filename.
#   - cell/nuclei channel files contain "c1"
#   - collagen channel files contain "sub7000"
CHANNEL_KEYWORDS = {
    "collagen": ["sub7000"],
    "cell": ["C2"],
}

DATAFRAME_MUST_CONTAIN = ["current_final_dataframe_byslice_pos", "trackmate"]

DROP_COLS = ['TotalImageArea', 'distance3d', 'neighbor3d',
             'bin_num3d', 'roi', 'type']

FEATURE_PREFIXES = (
    "texture", "fibercount", "mean_", "median_", "std_", "n_", "z_depth",
    "Area", "Lacunarity", "Alignment", "Total",
    'Endpoints', 'HGU (microns)', 'Branchpoints',
    'Box-Counting Fractal Dimension', '% High Density Matrix',
)

DISPLAY_MASK_COLORS = {  # RGB 0-1, used for the overlay panel
    "r10": (1.0, 0.15, 0.15),
    "r20": (0.15, 1.0, 0.15),
    "r30": (0.15, 0.45, 1.0),
}
OVERLAY_ALPHA = 0.45


# =====================================================================
# DATA LOADING (dataframes)
# =====================================================================

def get_mask_from_colname(col):
    parts = col.lower().split("_")
    if len(parts) >= 2 and parts[1] in ("r10", "r20", "r30", "full"):
        return "full" if parts[1] == "full" else parts[1]
    if "r10" in parts:
        return "r10"
    elif "r20" in parts:
        return "r20"
    elif "r30" in parts:
        return "r30"
    elif "full" in parts:
        return "full"
    return "full"


def clean_feature_name(col):
    tags = r'(?:_cell_masked|_r10_masked|_r20_masked|_r30_masked|_r10|_r20|_r30|_full|_cell)(?=_|$)'
    return re.sub(tags, '', col)


def build_long_dataframe(df_folder):
    """Reproduces the combine/melt/aggregate logic from your combining
    script and returns a tidy dataframe with columns:
    position, timepoint, mask_type, cell, feature, value
    """
    filelist = [
        f for f in os.listdir(df_folder)
        if f.lower().endswith(".csv")
        and all(k in f for k in DATAFRAME_MUST_CONTAIN)
    ]
    if not filelist:
        raise ValueError(
            f"No CSVs matching {DATAFRAME_MUST_CONTAIN} found in {df_folder}"
        )

    all_frames = []
    for f in filelist:
        df = pd.read_csv(os.path.join(df_folder, f))
        df = df.drop(DROP_COLS, axis=1, errors='ignore')
        if 'position' not in df.columns:
            m = re.search(r'pos_(\d+)', f, flags=re.IGNORECASE)
            df['position'] = int(m.group(1)) if m else np.nan
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["position"] = combined["position"].astype("Int64")

    id_candidates = ["image_name", "position", "timepoint", "concentration", "cell"]
    id_vars = [c for c in id_candidates if c in combined.columns]

    mask_groups = {"full": [], "r10": [], "r20": [], "r30": []}
    for col in combined.columns:
        if col in id_vars:
            continue
        if not col.startswith(FEATURE_PREFIXES):
            continue
        mask_groups[get_mask_from_colname(col)].append(col)

    long_frames = []
    for mask_type, cols in mask_groups.items():
        if not cols:
            continue
        temp = combined[id_vars + cols].melt(
            id_vars=id_vars, value_vars=cols,
            var_name="feature", value_name="value",
        )
        temp["mask_type"] = mask_type
        long_frames.append(temp)

    if not long_frames:
        raise ValueError("No recognizable feature columns found in the CSVs.")

    long = pd.concat(long_frames, ignore_index=True)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["feature"] = long["feature"].apply(clean_feature_name)

    group_cols = [c for c in ["position", "timepoint", "mask_type", "cell", "feature"]
                  if c in long.columns]
    grouped = long.groupby(group_cols, as_index=False).agg(value=("value", "mean"))

    if "position" in grouped.columns:
        grouped["position"] = grouped["position"].astype(str)
    if "cell" in grouped.columns:
        grouped["cell"] = grouped["cell"].astype(str)
    if "timepoint" in grouped.columns:
        grouped["timepoint"] = pd.to_numeric(grouped["timepoint"], errors="coerce")

    return grouped


# =====================================================================
# IMAGE / MASK LOADING
# =====================================================================

class ImageLibrary:
    """Scans the mask folder and the (separate) raw-channel folder once,
    and answers lookup questions about where collagen/cell/mask files live
    for a given position/cell."""

    def __init__(self, mask_folder, channel_folder):
        self.mask_folder_root = mask_folder
        self.channel_folder_root = channel_folder

        self.mask_files = []
        for root, _, files in os.walk(mask_folder):
            for fn in files:
                if fn.lower().endswith((".tif", ".tiff")):
                    self.mask_files.append(os.path.join(root, fn))

        self.channel_files = []
        for root, _, files in os.walk(channel_folder):
            for fn in files:
                if fn.lower().endswith((".tif", ".tiff")):
                    self.channel_files.append(os.path.join(root, fn))

        self._cache = {}

    def positions(self):
        found = set()
        for path in self.mask_files + self.channel_files:
            m = POS_REGEX.search(os.path.basename(path))
            if m:
                found.add(f"Pos{m.group(1)}")
        return sorted(found, key=lambda s: int(re.search(r'\d+', s).group()))

    def mask_folder(self, pos, radius):
        candidate = os.path.join(
            self.mask_folder_root,
            MASK_FOLDER_TEMPLATE.format(pos=pos, radius=radius),
        )
        if os.path.isdir(candidate):
            return candidate
        target = MASK_FOLDER_TEMPLATE.format(pos=pos, radius=radius).lower()
        for root, dirs, _ in os.walk(self.mask_folder_root):
            for d in dirs:
                if d.lower() == target:
                    return os.path.join(root, d)
        return None

    def cells_for_position(self, pos):
        cells = set()
        for radius in RADII:
            folder = self.mask_folder(pos, radius)
            if not folder:
                continue
            for fn in os.listdir(folder):
                if pos.lower() in fn.lower():
                    m = CELL_REGEX.search(fn)
                    if m:
                        cells.add(m.group(1))
        return sorted(cells)

    def find_channel_file(self, channel, pos, timepoint):
        """channel is 'collagen' or 'cell'. Searches the raw channel
        folder (not the mask folder). Returns a filepath or None."""
        keywords = CHANNEL_KEYWORDS.get(channel, [])
        t_str1 = f"_t{int(timepoint):02d}"
        t_str2 = f"_t{int(timepoint) + 1:02d}"
        candidates = []
        for path in self.channel_files:
            base_l = os.path.basename(path).lower()
            if pos.lower() not in base_l:
                continue
            if not any(k.lower() in base_l for k in keywords):
                continue
            if t_str1 in base_l or t_str2 in base_l:
                candidates.append(path)
        if not candidates:
            for path in self.channel_files:
                base_l = os.path.basename(path).lower()
                if pos.lower() in base_l and any(k.lower() in base_l for k in keywords):
                    candidates.append(path)
        return candidates[0] if candidates else None

    def find_mask_file(self, pos, radius, cell):
        folder = self.mask_folder(pos, radius)
        if not folder:
            return None
        for fn in os.listdir(folder):
            if pos.lower() in fn.lower() and f"cell_{cell}" in fn.lower():
                return os.path.join(folder, fn)
        return None

    def _read_cached(self, path, reader):
        if path not in self._cache:
            self._cache[path] = reader(path)
        return self._cache[path]

    def load_channel_stack(self, path):
        """Returns array shaped (Z, Y, X)."""
        def reader(p):
            arr = tiff.imread(p)
            arr = np.asarray(arr)
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim > 3:
                arr = arr.reshape((-1,) + arr.shape[-2:])
            return arr
        return self._read_cached(path, reader)

    def load_mask_stack(self, path):
        """Returns array shaped (T, Z, Y, X)."""
        def reader(p):
            arr = tiff.imread(p)
            arr = np.asarray(arr)
            arr = np.squeeze(arr)
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            elif arr.ndim != 4:
                raise ValueError(f"Unexpected mask shape {arr.shape} for {p}")
            return arr
        return self._read_cached(path, reader)


# =====================================================================
# IMAGE HELPERS
# =====================================================================

def normalize_for_display(img2d, p_low=2, p_high=98):
    img2d = img2d.astype(float)
    lo, hi = np.percentile(img2d, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((img2d - lo) / (hi - lo), 0, 1)
    return out


def build_overlay_rgb(background2d, masks_by_radius):
    bg = normalize_for_display(background2d)
    rgb = np.stack([bg, bg, bg], axis=-1)
    for radius in ["r30", "r20", "r10"]:
        mask = masks_by_radius.get(radius)
        if mask is None:
            continue
        m = mask > 0
        if not m.any():
            continue
        color = np.array(DISPLAY_MASK_COLORS[radius])
        for c in range(3):
            rgb[..., c] = np.where(
                m, rgb[..., c] * (1 - OVERLAY_ALPHA) + color[c] * OVERLAY_ALPHA, rgb[..., c]
            )
    return rgb


# =====================================================================
# GUI
# =====================================================================

class CollagenViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Collagen 3D Quantification Viewer")
        self.geometry("1500x950")

        self.grouped_df = None
        self.image_lib = None
        self.current_pos = tk.StringVar()
        self.current_feature = tk.StringVar()
        self.all_cells_var = tk.BooleanVar(value=False)
        self.z_var = tk.IntVar(value=0)
        self.t_var = tk.IntVar(value=0)

        self._build_setup_frame()

    def _build_setup_frame(self):
        self.setup_frame = ttk.Frame(self, padding=20)
        self.setup_frame.pack(fill="both", expand=True)

        ttk.Label(self.setup_frame, text="Collagen 3D Quantification Viewer",
                  font=("Segoe UI", 16, "bold")).pack(pady=(0, 20))

        self.mask_folder_var = tk.StringVar()
        self.channel_folder_var = tk.StringVar()
        self.df_folder_var = tk.StringVar()

        def make_row(label_text, var, browse_title):
            row = ttk.Frame(self.setup_frame)
            row.pack(fill="x", pady=8)
            ttk.Label(row, text=label_text, width=22).pack(side="left")
            ttk.Entry(row, textvariable=var, width=65).pack(side="left", padx=5)
            ttk.Button(row, text="Browse...",
                       command=lambda: self._browse_into(var, browse_title)).pack(side="left")

        make_row("Mask folder:", self.mask_folder_var,
                  "Select folder containing masks3d_per_cell_* subfolders")
        make_row("Raw channel folder:", self.channel_folder_var,
                  "Select folder containing c1 / sub7000 raw images")
        make_row("Dataframe folder:", self.df_folder_var,
                  "Select folder containing the trackmate CSVs")

        ttk.Button(self.setup_frame, text="Load", command=self._load_everything).pack(pady=20)

        self.setup_status = ttk.Label(self.setup_frame, text="", foreground="red")
        self.setup_status.pack()

    def _browse_into(self, var, title):
        folder = filedialog.askdirectory(title=title)
        if folder:
            var.set(folder)

    def _load_everything(self):
        mask_folder = self.mask_folder_var.get().strip()
        channel_folder = self.channel_folder_var.get().strip()
        df_folder = self.df_folder_var.get().strip()

        if not mask_folder or not os.path.isdir(mask_folder):
            self.setup_status.config(text="Please choose a valid mask folder.")
            return
        if not channel_folder or not os.path.isdir(channel_folder):
            self.setup_status.config(text="Please choose a valid raw channel folder.")
            return
        if not df_folder or not os.path.isdir(df_folder):
            self.setup_status.config(text="Please choose a valid dataframe folder.")
            return

        try:
            self.grouped_df = build_long_dataframe(df_folder)
        except Exception as e:
            self.setup_status.config(text=f"Error loading dataframes: {e}")
            traceback.print_exc()
            return

        try:
            self.image_lib = ImageLibrary(mask_folder, channel_folder)
        except Exception as e:
            self.setup_status.config(text=f"Error scanning image folders: {e}")
            traceback.print_exc()
            return

        self.setup_frame.destroy()
        self._build_main_ui()

    def _build_main_ui(self):
        controls = ttk.Frame(self, padding=8)
        controls.pack(fill="x")

        ttk.Label(controls, text="Position:").grid(row=0, column=0, sticky="w")
        self.pos_combo = ttk.Combobox(controls, textvariable=self.current_pos,
                                       state="readonly", width=12)
        self.pos_combo.grid(row=0, column=1, padx=5)
        self.pos_combo.bind("<<ComboboxSelected>>", lambda e: self._on_position_change())

        ttk.Label(controls, text="Feature:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.feature_combo = ttk.Combobox(controls, textvariable=self.current_feature,
                                           state="readonly", width=28)
        self.feature_combo.grid(row=0, column=3, padx=5)
        self.feature_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_plots())

        ttk.Label(controls, text="Cell(s):").grid(row=0, column=4, sticky="w", padx=(20, 0))
        cell_frame = ttk.Frame(controls)
        cell_frame.grid(row=0, column=5, padx=5)
        self.cell_listbox = tk.Listbox(cell_frame, selectmode="extended",
                                        height=4, exportselection=False, width=10)
        self.cell_listbox.pack(side="left")
        cell_scroll = ttk.Scrollbar(cell_frame, orient="vertical",
                                     command=self.cell_listbox.yview)
        cell_scroll.pack(side="left", fill="y")
        self.cell_listbox.config(yscrollcommand=cell_scroll.set)
        self.cell_listbox.bind("<<ListboxSelect>>", lambda e: self._on_cell_selection_change())

        self.all_cells_check = ttk.Checkbutton(
            controls, text="All cells", variable=self.all_cells_var,
            command=self._on_all_cells_toggle,
        )
        self.all_cells_check.grid(row=0, column=6, padx=(10, 0))

        ttk.Button(controls, text="Change folders...", command=self._reset_folders).grid(
            row=0, column=7, padx=(30, 0)
        )

        slider_frame = ttk.Frame(self, padding=(8, 0))
        slider_frame.pack(fill="x")
        ttk.Label(slider_frame, text="Timepoint:").pack(side="left")
        self.t_scale = ttk.Scale(slider_frame, from_=0, to=0, orient="horizontal",
                                  variable=self.t_var, command=lambda v: self._refresh_images())
        self.t_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.t_label = ttk.Label(slider_frame, text="0")
        self.t_label.pack(side="left", padx=(0, 20))

        ttk.Label(slider_frame, text="Z-slice:").pack(side="left")
        self.z_scale = ttk.Scale(slider_frame, from_=0, to=0, orient="horizontal",
                                  variable=self.z_var, command=lambda v: self._refresh_images())
        self.z_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.z_label = ttk.Label(slider_frame, text="0")
        self.z_label.pack(side="left")

        image_panel = ttk.Frame(self, padding=8)
        image_panel.pack(fill="x")

        self.fig_images = Figure(figsize=(13, 4), dpi=100)
        self.ax_collagen = self.fig_images.add_subplot(1, 3, 1)
        self.ax_cell = self.fig_images.add_subplot(1, 3, 2)
        self.ax_overlay = self.fig_images.add_subplot(1, 3, 3)
        for ax, title in zip(
            [self.ax_collagen, self.ax_cell, self.ax_overlay],
            ["Collagen channel", "Cell / nuclei channel", "Mask overlay (r10/r20/r30)"],
        ):
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        self.canvas_images = FigureCanvasTkAgg(self.fig_images, master=image_panel)
        self.canvas_images.get_tk_widget().pack(fill="both", expand=True)

        plot_panel = ttk.Frame(self, padding=8)
        plot_panel.pack(fill="both", expand=True)

        self.fig_plots = Figure(figsize=(13, 4), dpi=100)
        self.ax_r10 = self.fig_plots.add_subplot(1, 3, 1)
        self.ax_r20 = self.fig_plots.add_subplot(1, 3, 2)
        self.ax_r30 = self.fig_plots.add_subplot(1, 3, 3)
        self.canvas_plots = FigureCanvasTkAgg(self.fig_plots, master=plot_panel)
        self.canvas_plots.get_tk_widget().pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            fill="x", side="bottom"
        )

        self._populate_positions()
        self._populate_features()

    def _reset_folders(self):
        for child in self.winfo_children():
            child.destroy()
        self.grouped_df = None
        self.image_lib = None
        self._build_setup_frame()

    def _populate_positions(self):
        df_positions = set(self.grouped_df["position"].unique()) if "position" in self.grouped_df else set()
        img_positions = set(self.image_lib.positions())
        df_positions_norm = set()
        for p in df_positions:
            m = re.search(r'\d+', str(p))
            if m:
                df_positions_norm.add(f"Pos{m.group()}")
        positions = sorted(img_positions | df_positions_norm,
                            key=lambda s: int(re.search(r'\d+', s).group())) \
            if (img_positions | df_positions_norm) else []
        self.pos_combo["values"] = positions
        if positions:
            self.current_pos.set(positions[0])
            self._on_position_change()
        else:
            self.status_var.set("No positions found in image folders or dataframes.")

    def _populate_features(self):
        features = sorted(self.grouped_df["feature"].unique()) if "feature" in self.grouped_df else []
        self.feature_combo["values"] = features
        if features:
            self.current_feature.set(features[0])

    def _on_position_change(self):
        pos = self.current_pos.get()
        pos_num = re.search(r'\d+', pos).group() if re.search(r'\d+', pos) else pos

        img_cells = set(self.image_lib.cells_for_position(pos))
        df_cells = set()
        if "cell" in self.grouped_df.columns:
            sub = self.grouped_df[self.grouped_df["position"].astype(str).str.contains(
                pos_num, na=False)]
            df_cells = set(c for c in sub["cell"].unique() if str(c).lower() != "full")

        cells = sorted(img_cells | df_cells)
        self.cell_listbox.delete(0, "end")
        for c in cells:
            self.cell_listbox.insert("end", c)
        if cells:
            self.cell_listbox.selection_set(0)

        if "timepoint" in self.grouped_df.columns:
            sub = self.grouped_df[self.grouped_df["position"].astype(str).str.contains(
                pos_num, na=False)]
            tps = sub["timepoint"].dropna().unique()
            max_t = int(max(tps)) if len(tps) else 0
        else:
            max_t = 0
        self.t_scale.config(to=max(max_t, 0))
        self.t_var.set(0)

        z_max = 0
        sample_path = self.image_lib.find_channel_file("collagen", pos, 0)
        if sample_path:
            try:
                stack = self.image_lib.load_channel_stack(sample_path)
                z_max = stack.shape[0] - 1
            except Exception:
                pass
        self.z_scale.config(to=max(z_max, 0))
        self.z_var.set(0)

        self._on_cell_selection_change()
        self._refresh_images()

    def _on_all_cells_toggle(self):
        if self.all_cells_var.get():
            self.cell_listbox.selection_set(0, "end")
        self._refresh_plots()

    def _on_cell_selection_change(self):
        self._refresh_images()
        self._refresh_plots()

    def selected_cells(self):
        if self.all_cells_var.get():
            return list(self.cell_listbox.get(0, "end"))
        return [self.cell_listbox.get(i) for i in self.cell_listbox.curselection()]

    def primary_cell(self):
        cells = self.selected_cells()
        return cells[0] if cells else None

    def _refresh_images(self, *_):
        if self.image_lib is None:
            return
        pos = self.current_pos.get()
        cell = self.primary_cell()
        t = int(round(self.t_var.get()))
        z = int(round(self.z_var.get()))
        self.t_label.config(text=str(t))
        self.z_label.config(text=str(z))

        for ax in (self.ax_collagen, self.ax_cell, self.ax_overlay):
            ax.clear()
            ax.axis("off")
        self.ax_collagen.set_title("Collagen channel", fontsize=10)
        self.ax_cell.set_title("Cell / nuclei channel", fontsize=10)
        self.ax_overlay.set_title("Mask overlay (r10/r20/r30)", fontsize=10)

        collagen_slice = None
        try:
            path = self.image_lib.find_channel_file("collagen", pos, t)
            if path:
                stack = self.image_lib.load_channel_stack(path)
                z_clamped = min(z, stack.shape[0] - 1)
                collagen_slice = stack[z_clamped]
                self.ax_collagen.imshow(normalize_for_display(collagen_slice), cmap="gray")
            else:
                self.ax_collagen.text(0.5, 0.5, "collagen image not found\n(sub7000)",
                                       ha="center", va="center", fontsize=9, wrap=True)
        except Exception as e:
            self.ax_collagen.text(0.5, 0.5, f"error:\n{e}", ha="center", va="center", fontsize=8)

        try:
            path = self.image_lib.find_channel_file("cell", pos, t)
            if path:
                stack = self.image_lib.load_channel_stack(path)
                z_clamped = min(z, stack.shape[0] - 1)
                self.ax_cell.imshow(normalize_for_display(stack[z_clamped]), cmap="gray")
            else:
                self.ax_cell.text(0.5, 0.5, "cell channel image not found\n(c1)",
                                   ha="center", va="center", fontsize=9, wrap=True)
        except Exception as e:
            self.ax_cell.text(0.5, 0.5, f"error:\n{e}", ha="center", va="center", fontsize=8)

        try:
            if cell is not None:
                masks_2d = {}
                for radius in RADII:
                    mpath = self.image_lib.find_mask_file(pos, radius, cell)
                    if not mpath:
                        continue
                    mstack = self.image_lib.load_mask_stack(mpath)
                    t_clamped = min(t, mstack.shape[0] - 1)
                    z_clamped = min(z, mstack.shape[1] - 1)
                    masks_2d[radius] = mstack[t_clamped, z_clamped]
                if masks_2d:
                    bg = collagen_slice if collagen_slice is not None else next(iter(masks_2d.values()))
                    overlay = build_overlay_rgb(bg, masks_2d)
                    self.ax_overlay.imshow(overlay)
                else:
                    self.ax_overlay.text(0.5, 0.5, "no masks found for this cell",
                                          ha="center", va="center", fontsize=9, wrap=True)
            else:
                self.ax_overlay.text(0.5, 0.5, "select a cell", ha="center", va="center")
        except Exception as e:
            self.ax_overlay.text(0.5, 0.5, f"error:\n{e}", ha="center", va="center", fontsize=8)

        self.canvas_images.draw_idle()

    def _refresh_plots(self, *_):
        if self.grouped_df is None:
            return
        pos = self.current_pos.get()
        pos_num = re.search(r'\d+', pos).group() if re.search(r'\d+', pos) else pos
        feature = self.current_feature.get()
        cells = self.selected_cells()

        axes = {"r10": self.ax_r10, "r20": self.ax_r20, "r30": self.ax_r30}
        for radius, ax in axes.items():
            ax.clear()
            ax.set_title(f"{feature or ''}  ({radius})", fontsize=10)
            ax.set_xlabel("Timepoint")
            ax.set_ylabel("Value")

        if not feature or not cells:
            self.canvas_plots.draw_idle()
            return

        df = self.grouped_df
        sub = df[
            (df["position"].astype(str).str.contains(pos_num, na=False))
            & (df["feature"] == feature)
        ]

        for radius, ax in axes.items():
            rsub = sub[sub["mask_type"] == radius]
            plotted_any = False
            for cell in cells:
                csub = rsub[rsub["cell"].astype(str) == str(cell)].sort_values("timepoint")
                if csub.empty:
                    continue
                ax.plot(csub["timepoint"], csub["value"], marker="o", label=f"cell {cell}")
                plotted_any = True
            if plotted_any and len(cells) <= 15:
                ax.legend(fontsize=7)
            elif not plotted_any:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)

        self.fig_plots.tight_layout()
        self.canvas_plots.draw_idle()


if __name__ == "__main__":
    app = CollagenViewerApp()
    app.mainloop()