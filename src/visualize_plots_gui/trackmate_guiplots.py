"""
Collagen 3D Quantification Viewer
==================================

A Tkinter GUI for exploring per-cell 3D texture-analysis output alongside the
raw microscopy images they were computed from.

WHAT IT DOES
------------
1. Asks you to pick THREE folders (mask folder, raw channel folder,
   dataframe folder) plus matching keywords (collagen channel, cell channel,
   position, radii). All of these are editable on the loading screen and saved
   to collagen_viewer_settings.json.
2. Handles positionless datasets, 1-indexed / 0-indexed timepoints, and padded
   cell IDs (e.g. cell_0000 vs cell 0) seamlessly.
3. Renders 2D slices across Z and T for raw channel stacks and cell mask overlays.
4. Plots feature values over time for selected cells and radii.

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
import threading
import queue
import json

import numpy as np
import pandas as pd
import tifffile as tiff

import tkinter as tk
from tkinter import ttk, filedialog
from collections import defaultdict

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =====================================================================
# CONFIG -- edit these to match your actual file-naming conventions
# =====================================================================

DEFAULT_POSITION_KEYWORDS = ["Pos"]

# Radius labels must match the suffix used by trackmate_shell_masks.py, i.e.
# "r{outer}um" for a solid sphere (inner_radius_um = 0) or
# "r{inner}to{outer}um" for a shell/doughnut mask.
DEFAULT_RADII = ["r200um", "r150to200um", "r200to250um", "r250to350um"]
CELL_REGEX = re.compile(r'cell_(\d+)', re.IGNORECASE)

# Radius label is already the full folder suffix (e.g. "r200um" or
# "r150to200um") produced by trackmate_shell_masks.py, so it is NOT modified
# here -- do not append "um" again.
MASK_FOLDER_TEMPLATE = "masks3d_per_cell_{pos}_{radius}"

SETTINGS_FILE = "collagen_viewer_settings.json"

DEFAULT_CHANNEL_KEYWORDS = {
    "collagen": "C1",
    "cell": "C2",
    "texture": "autoc,contr,corrm,corrp,cprom,cshad,denth,dissi,dvarh,energ,entro,homom,homop,idmnc,indnc,inf1h,inf2h,maxpr,savgh,senth,sosvh,svarh",
    "R": "R1"
}

DATAFRAME_MUST_CONTAIN = ["dataframe_byslice_pos", "trackmate"]

DROP_COLS = ['TotalImageArea', 'distance3d', 
             'bin_num3d', 'roi', 'type']

FEATURE_PREFIXES = (
    "texture", "fibercount", "mean_", "median_", "std_", "n_", "z_depth",
    "Area", "Lacunarity", "Alignment", "Total",
    'Endpoints', 'HGU (microns)', 'Branchpoints',
    'Box-Counting Fractal Dimension', '% High Density Matrix',
)

RADIUS_PALETTE = [
    (1.0, 0.15, 0.15),  # Red
    (0.15, 1.0, 0.15),  # Green
    (0.15, 0.45, 1.0),  # Blue
    (1.0, 0.8, 0.1),   # Yellow/Orange
    (0.8, 0.2, 1.0),   # Purple
]
OVERLAY_ALPHA = 0.45


def parse_keyword_field(text):
    """Splits a comma-separated keyword field into a clean list of
    non-empty, stripped keywords. Returns [] if empty/whitespace."""
    if not text:
        return []
    return [kw.strip() for kw in text.split(",") if kw.strip()]


def find_position_in_text(text, position_keywords):
    """Searches `text` for position keywords followed by a number.
    If position_keywords is empty, defaults to assuming single position 'Pos0'."""
    if not position_keywords:
        return "Pos0"
    for keyword in position_keywords:
        pattern = re.compile(rf'{re.escape(keyword)}[_\- ]*(\d+)', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return f"{keyword}{match.group(1)}"
    return None


def clean_cell_id(value):
    """Normalizes cell identifiers so '0000', 'cell_0', 0, and '0.0' all evaluate to '0'."""
    if pd.isna(value) or value is None:
        return ""
    s = str(value).strip()
    m = re.search(r'\d+', s)
    if m:
        return str(int(m.group()))
    return s


def extract_timepoint_from_filename(filename):
    """Robustly parses timepoint integers from filenames."""
    #0. Match timepoint after channel tag
    m = re.search(r'CH\d{2}_(\d+)', filename,re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 1. Match standard patterns like _t44, _t044, tp44, frame44, t=44
    m = re.search(r'(?:_t|[\-_b]t|timepoint|tp|frame|t=)[_\- ]*(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 2. Match isolated 't' followed by digits: e.g. t01, T001
    m = re.search(r'(?:^|[_\-\s/])t(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 3. Strip Z-slice tags so Z index does not get confused for timepoint
    clean_name = re.sub(r'(?:_z|[\-_b]z|zslice|z=)[_\- ]*\d+', '', filename, flags=re.IGNORECASE)
    name_no_ext = re.sub(r'\.[^.]+$', '', clean_name)
    numbers = re.findall(r'\d+', name_no_ext)
    if numbers:
        return int(numbers[-1])

    return None


# =====================================================================
# SLICE EXTRACTOR HELPER
# =====================================================================

def extract_2d_slice(stack, t_idx, z_idx, expected_t=1, expected_z=1):
    """Safely extracts a 2D (Y, X) numpy slice from 2D, 3D, 4D, or flattened stacks."""
    if stack is None:
        return None

    arr = np.squeeze(np.asarray(stack))

    if arr.ndim == 2:
        return arr

    elif arr.ndim == 3:
        d0 = arr.shape[0]
        # Flattened 3D stack (T*Z, Y, X)
        if expected_z > 1 and expected_t > 1 and d0 == expected_z * expected_t:
            flat_idx = int(t_idx) * int(expected_z) + int(z_idx)
            flat_idx = min(max(0, flat_idx), d0 - 1)
            return arr[flat_idx, :, :]
        elif expected_z > 1 and d0 == expected_z:
            z_c = min(max(0, int(z_idx)), d0 - 1)
            return arr[z_c, :, :]
        elif expected_t > 1 and d0 == expected_t:
            t_c = min(max(0, int(t_idx)), d0 - 1)
            return arr[t_c, :, :]
        else:
            idx = min(max(0, int(z_idx)), d0 - 1)
            return arr[idx, :, :]

    elif arr.ndim == 4:
        d0, d1 = arr.shape[0], arr.shape[1]
        if d0 == expected_t or (d0 != expected_z and d1 == expected_z):
            # Format (T, Z, Y, X)
            t_c = min(max(0, int(t_idx)), d0 - 1)
            z_c = min(max(0, int(z_idx)), d1 - 1)
            return arr[t_c, z_c, :, :]
        else:
            # Format (Z, T, Y, X)
            z_c = min(max(0, int(z_idx)), d0 - 1)
            t_c = min(max(0, int(t_idx)), d1 - 1)
            return arr[z_c, t_c, :, :]

    elif arr.ndim > 4:
        arr_flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        idx = min(max(0, int(z_idx)), arr_flat.shape[0] - 1)
        return arr_flat[idx, :, :]

    return arr


# =====================================================================
# DATA LOADING (dataframes)
# =====================================================================

def get_mask_from_colname(col, radii):
    """Extracts the radius mask group from column names safely.

    Recognizes both solid-sphere labels ("r200um") and shell/doughnut
    labels ("r150to200um") produced by trackmate_shell_masks.py.
    """
    col_lower = col.lower()

    # 1. Match against user-specified radii (longest first to avoid r10 matching r100,
    #    and to avoid "r200um" matching inside "r200to250um")
    sorted_radii = sorted(radii, key=len, reverse=True)
    for r in sorted_radii:
        pattern = rf'(?:^|_){re.escape(r.lower())}(?:_|$)'
        if re.search(pattern, col_lower):
            return r

    # 2. Match auto-detected _r<number>(to<number>)?(um)? patterns against user radii,
    #    case-insensitively. Handles both "r200um" and "r150to200um" style tags.
    m = re.search(r'_(r\d+(?:to\d+)?)(?:um)?(?:_|$)', col_lower)
    if m:
        detected = m.group(1)
        for r in radii:
            # Compare against the radius label with any trailing "um" stripped too,
            # so "r200um" (detected) still matches a user radius of "r200um".
            r_norm = r.lower()[:-2] if r.lower().endswith('um') else r.lower()
            if r_norm == detected or r.lower() == detected:
                return r
        #print(f"Warning: Detected radius '{detected}' in column '{col}' does not match any user-specified radii {radii}.")
        return None  # Return None if no match found

    if "full" in col_lower:
        return "full"
    else:
        print(f"Warning: Could not determine radius mask for column '{col}'. Defaulting to 'full'.")

    return radii[0] if radii else "full"


STATISTICS = ["mean", "median", "std"]

def split_feature_and_statistic(col, radii, texture_keywords=None):
    text = str(col).lower()
    parts = text.split('_')

    stat_terms = {"mean", "median", "std"}
    statistic = next((p for p in parts if p in stat_terms), "mean")

    radius_terms = {r.lower() for r in radii}
    # Matches both "r200um" (solid sphere) and "r150to200um" (shell) style tags.
    radius_pattern = re.compile(r'^r\d+(?:to\d+)?(?:um)?$')

    # Only strip generic structural tags — NOT the property name itself
    boilerplate = {"texture", "texture3d", "masked"}

    kept = []
    for p in parts:
        if not p or p in stat_terms or p in boilerplate:
            continue
        if p in radius_terms or radius_pattern.match(p):
            continue
        kept.append(p)

    cleaned = "_".join(kept)
    return statistic, cleaned or text

def build_long_dataframe(df_folder, position_keywords=None, radii=None, texture_keywords=None, R_keywords=None):
    if position_keywords is None:
        position_keywords = DEFAULT_POSITION_KEYWORDS
    radii = radii or DEFAULT_RADII
    texture_keywords = texture_keywords or [kw for kw in DEFAULT_CHANNEL_KEYWORDS["texture"].split(",") if kw]
    R_keywords = R_keywords or [kw for kw in DEFAULT_CHANNEL_KEYWORDS["R"].split(",") if kw]
    print("R_keywords in build_long_dataframe:", R_keywords)
    

    filelist = [
        f for f in os.listdir(df_folder)
        if f.lower().endswith(".csv")
        and all(k.lower() in f.lower() for k in DATAFRAME_MUST_CONTAIN)
    ]
    if not filelist:
        raise ValueError(
            f"No CSVs matching {DATAFRAME_MUST_CONTAIN} found in {df_folder}"
        )

    all_frames = []
    for f in filelist:
        df = pd.read_csv(os.path.join(df_folder, f))
        df = df.drop(DROP_COLS, axis=1, errors='ignore')

        assigned = find_position_in_text(f, position_keywords)
        if assigned is None:
            assigned = "Pos0" if not position_keywords else os.path.splitext(f)[0]

        df["position"] = assigned
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["position"] = combined["position"].astype(str)

    id_candidates = ["image_name", "position", "timepoint", "concentration", "cell", "neighbor3d"]
    id_vars = [c for c in id_candidates if c in combined.columns]

    # Use defaultdict to dynamically register any radius without triggering KeyErrors
    mask_groups = defaultdict(list)
    mask_groups["full"] = []
    for r in radii:
        mask_groups[r] = []

    for col in combined.columns:
        if col in id_vars:
            continue
        if not col.startswith(FEATURE_PREFIXES):
            continue
        mask_type = get_mask_from_colname(col, radii)
        if mask_type == "full":
            continue
        mask_groups[mask_type].append(col)

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

    unique_cols = long["feature"].unique()
    mapping = {
        col: split_feature_and_statistic(col, radii, texture_keywords=texture_keywords)
        for col in unique_cols
    }
    feature_stat = long["feature"].map(mapping)
    long[["statistic", "feature"]] = pd.DataFrame(feature_stat.tolist(), index=long.index)

    group_cols = [c for c in ["position", "timepoint", "mask_type", "cell", "feature", "statistic","neighbor3d"] if c in long.columns]
    grouped = long.groupby(group_cols, as_index=False).agg(value=("value", "mean"))

    if "position" in grouped.columns:
        grouped["position"] = grouped["position"].astype(str)
    if "neighbor3d" in grouped.columns:
        grouped["neighbor3d"] = grouped["neighbor3d"].astype(str)
    if "cell" in grouped.columns:
        grouped["cell"] = grouped["cell"].apply(clean_cell_id)
    if "timepoint" in grouped.columns:
        grouped["timepoint"] = pd.to_numeric(grouped["timepoint"], errors="coerce")

    return grouped

# =====================================================================
# IMAGE / MASK LOADING
# =====================================================================

class ImageLibrary:
    def __init__(self, mask_folder, channel_folder, user_z, user_t,
                 channel_keywords=None, position_keywords=None, radii=None,texture_folder=None):
        self.mask_folder_root = mask_folder
        self.channel_folder_root = channel_folder
        self.texture_folder = texture_folder
        self.user_z = user_z
        self.user_t = user_t
        self.position_keywords = (position_keywords if position_keywords is not None 
                                  else DEFAULT_POSITION_KEYWORDS)
        self.radii = (radii or DEFAULT_RADII)

        self.channel_keywords = channel_keywords or {
            "collagen": [DEFAULT_CHANNEL_KEYWORDS["collagen"]],
            "cell": [DEFAULT_CHANNEL_KEYWORDS["cell"]],
            "R": [DEFAULT_CHANNEL_KEYWORDS["R"]]
        }

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

        if texture_folder and os.path.isdir(texture_folder):
            for root, _, files in os.walk(texture_folder):
                for fn in files:
                    if fn.lower().endswith((".tif", ".tiff")):
                        self.channel_files.append(os.path.join(root, fn))


        self._mask_dirs = []
        for root, dirs, _ in os.walk(self.mask_folder_root):
            for d in dirs:
                self._mask_dirs.append(os.path.join(root, d))

        self._cache = {}
        self._mask_folder_cache = {}

    def positions(self):
        if not self.position_keywords:
            return ["Pos0"]
        found = set()
        for path in self.channel_files:
            pos = find_position_in_text(path, self.position_keywords)
            if pos:
                found.add(pos)
        for path in self.mask_files:
            pos = find_position_in_text(path, self.position_keywords)
            if pos:
                found.add(pos)
        return sorted(found) if found else ["Pos0"]

    def mask_folder(self, pos, radius):
        cache_key = (pos, radius)
        if cache_key in self._mask_folder_cache:
            return self._mask_folder_cache[cache_key]

        candidate = os.path.join(
            self.mask_folder_root,
            MASK_FOLDER_TEMPLATE.format(pos=pos, radius=radius),
        )
        if os.path.isdir(candidate):
            self._mask_folder_cache[cache_key] = candidate
            return candidate

        radius_l = radius.lower()
        pos_l = pos.lower() if pos else ""

        # Substring match on folder basenames. Since radius labels like "r200um"
        # and "r150to200um" are matched as full substrings (not prefixes), a
        # solid-sphere label like "r200um" will not incorrectly match a shell
        # folder such as "r200to250um" (the literal text "r200um" does not
        # appear inside "r200to250um").
        radius_matches = [d for d in self._mask_dirs if radius_l in os.path.basename(d).lower()]
        pos_and_radius = [d for d in radius_matches if pos_l in os.path.basename(d).lower()]

        if pos_and_radius and self.position_keywords:
            result = pos_and_radius[0]
        elif radius_matches:
            result = radius_matches[0]
        else:
            result = None

        self._mask_folder_cache[cache_key] = result
        return result

    def cells_for_position(self, pos):
        cells = set()
        for radius in self.radii:
            folder = self.mask_folder(pos, radius)
            if not folder:
                continue
            for fn in os.listdir(folder):
                m = CELL_REGEX.search(fn)
                if m:
                    cells.add(clean_cell_id(m.group(1)))
        return sorted(cells, key=lambda x: int(x) if x.isdigit() else x)

    def find_channel_file(self,channel,pos,timepoint,current_r,override_keywords=None,match_all=False):
        
        keywords = self.channel_keywords.get(channel, [])
        #print(f"Default keywords for channel '{channel}': {keywords}")

        if override_keywords:
            keywords = override_keywords

        # Make a copy so we don't accidentally modify the original
        keywords = list(keywords)

        print(f"Final search keywords: {keywords}")

        target_t = int(timepoint)

        def _kw_in(base_l, kw):
            pattern = rf'(?:^|[_\-\.]){re.escape(str(kw).lower())}(?:[_\-\.]|$)'
            return re.search(pattern, base_l) is not None

        valid_paths = []

        for path in self.channel_files:
            if self.position_keywords:
                found_pos = find_position_in_text(
                    path,
                    self.position_keywords
                )

                # Only exclude on a conflicting position tag
                if found_pos is not None and found_pos != pos:
                    continue

            base_l = os.path.basename(path).lower()

            if match_all:
                if keywords and all(_kw_in(base_l, k) for k in keywords):
                    valid_paths.append(path)
            else:
                if any(_kw_in(base_l, k) for k in keywords):
                    valid_paths.append(path)

        if not valid_paths:
            print("no valid path")
            return None

        for path in valid_paths:
            tp = extract_timepoint_from_filename(
                os.path.basename(path)
            )
            if tp is not None and tp == target_t:
                return path

        for path in valid_paths:
            tp = extract_timepoint_from_filename(
                os.path.basename(path)
            )
            if tp is not None and tp == target_t + 1:
                return path

        print(valid_paths)
        return valid_paths[0]


    def find_mask_file(self, pos, radius, cell):
        folder = self.mask_folder(pos, radius)
        if not folder:
            return None

        pos_l = pos.lower() if pos else ""
        target_cell = clean_cell_id(cell)

        cell_matches = []
        for fn in os.listdir(folder):
            m = CELL_REGEX.search(fn)
            if m and clean_cell_id(m.group(1)) == target_cell:
                cell_matches.append(fn)

        if not cell_matches:
            return None

        if self.position_keywords:
            pos_and_cell = [fn for fn in cell_matches if pos_l in fn.lower()]
            chosen = pos_and_cell[0] if pos_and_cell else cell_matches[0]
        else:
            chosen = cell_matches[0]

        return os.path.join(folder, chosen)

    def _read_cached(self, path, reader):
        if path not in self._cache:
            self._cache[path] = reader(path)
        return self._cache[path]

    def load_channel_stack(self, path):
        def reader(p):
            return np.squeeze(np.asarray(tiff.imread(p)))
        return self._read_cached(path, reader)

    def load_mask_stack(self, path):
        def reader(p):
            return np.squeeze(np.asarray(tiff.imread(p)))
        return self._read_cached(path, reader)


# =====================================================================
# IMAGE HELPERS
# =====================================================================

def normalize_for_display(img2d, p_low=2, p_high=98):
    if img2d is None:
        return np.zeros((100, 100))
    img2d = img2d.astype(float)
    lo, hi = np.percentile(img2d, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((img2d - lo) / (hi - lo), 0, 1)
    return out


def build_overlay_rgb(background2d, masks_by_radius, radii_order, color_map,
                      p_low=2, p_high=98):
    if background2d is None:
        mask_shape = next(
            (mask.shape for mask in masks_by_radius.values() if mask is not None),
            None
        )
        if mask_shape is None:
            return np.zeros((100, 100, 3))
        bg = np.zeros(mask_shape, dtype=float)
    else:
        bg = normalize_for_display(background2d, p_low, p_high)

    rgb = np.stack([bg, bg, bg], axis=-1)

    for radius in reversed(list(radii_order)):
        mask = masks_by_radius.get(radius)
        if mask is None or mask.shape != rgb.shape[:2]:
            continue
        m = mask > 0
        if not m.any():
            continue
        color = np.array(color_map.get(radius, (1.0, 1.0, 1.0)))
        for c in range(3):
            rgb[..., c] = np.where(
                m,
                rgb[..., c] * (1 - OVERLAY_ALPHA) + color[c] * OVERLAY_ALPHA,
                rgb[..., c]
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
        self.current_statistic = tk.StringVar(value="mean")
        self.current_R = tk.StringVar(value="R1")
        self.all_cells_var = tk.BooleanVar(value=False)
        self.all_R_var = tk.BooleanVar(value=False)
        self.z_var = tk.IntVar(value=0)
        self.t_var = tk.IntVar(value=0)
        self.user_z = tk.IntVar(value=46)
        self.user_t = tk.IntVar(value=25)
        self.radii = list(DEFAULT_RADII)
        self.radius_colors = {r: RADIUS_PALETTE[i % len(RADIUS_PALETTE)]
                              for i, r in enumerate(self.radii)}
        self.contrast_values = {
            "collagen": {"low": 2, "high": 98},
            "cell": {"low": 2, "high": 98},
            "texture": {"low": 2, "high": 98},
        }
        self.contrast_channel_var = tk.StringVar(value="Collagen")
        self.contrast_low_var = tk.DoubleVar(value=2)
        self.contrast_high_var = tk.DoubleVar(value=98)

        self._build_setup_frame()

    def _load_all_settings_file(self):
        if not os.path.exists(SETTINGS_FILE):
            return {"experiments": {}, "last_used": None}
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        data.setdefault("experiments", {})
        data.setdefault("last_used", None)
        return data

    def _settings_snapshot(self):
        return {
            "mask_folder": self.mask_folder_var.get(),
            "channel_folder": self.channel_folder_var.get(),
            "texture_folder": self.texture_folder_var.get(),
            "dataframe_folder": self.df_folder_var.get(),
            "z_slices": self.user_z.get(),
            "timepoints": self.user_t.get(),
            "collagen_keyword": self.collagen_keyword_var.get(),
            "cell_keyword": self.cell_keyword_var.get(),
            "position_keywords": self.position_keyword_var.get(),
            "radii": self.radius_var.get(),
            "texture_keyword": self.texture_keyword_var.get(),
            "R_keyword": self.R_keyword_var.get(),
        }

    def _apply_settings(self, settings):
        self.mask_folder_var.set(settings.get("mask_folder", ""))
        self.channel_folder_var.set(settings.get("channel_folder", ""))
        self.texture_folder_var.set(settings.get("texture_folder", ""))
        self.df_folder_var.set(settings.get("dataframe_folder", ""))
        self.user_z.set(settings.get("z_slices", 46))
        self.user_t.set(settings.get("timepoints", 26))
        self.collagen_keyword_var.set(settings.get("collagen_keyword", DEFAULT_CHANNEL_KEYWORDS["collagen"]))
        self.cell_keyword_var.set(settings.get("cell_keyword", DEFAULT_CHANNEL_KEYWORDS["cell"]))
        self.position_keyword_var.set(settings.get("position_keywords", "Pos"))
        self.radius_var.set(settings.get("radii", ",".join(DEFAULT_RADII)))
        self.texture_keyword_var.set(settings.get("texture_keyword", DEFAULT_CHANNEL_KEYWORDS["texture"]))
        self.R_keyword_var.set(settings.get("R_keyword", DEFAULT_CHANNEL_KEYWORDS["R"]))

    def load_settings(self):
        data = self._load_all_settings_file()
        names = sorted(data["experiments"].keys())
        self.experiment_combo["values"] = names

        last_used = data.get("last_used")
        if last_used and last_used in data["experiments"]:
            self.experiment_name_var.set(last_used)
            self._apply_settings(data["experiments"][last_used])
        elif names:
            self.experiment_name_var.set(names[0])
            self._apply_settings(data["experiments"][names[0]])

    def save_settings(self):
        data = self._load_all_settings_file()
        name = self.experiment_name_var.get().strip() or "Default"
        data["experiments"][name] = self._settings_snapshot()
        data["last_used"] = name
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        self.experiment_combo["values"] = sorted(data["experiments"].keys())

    def delete_current_experiment(self):
        data = self._load_all_settings_file()
        name = self.experiment_name_var.get().strip()
        if name in data["experiments"]:
            del data["experiments"][name]
            if data.get("last_used") == name:
                data["last_used"] = None
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=4)
            self.experiment_combo["values"] = sorted(data["experiments"].keys())
            self.setup_status.config(foreground="black", text=f"Deleted experiment '{name}'.")

    def _on_experiment_selected(self, event=None):
        data = self._load_all_settings_file()
        name = self.experiment_name_var.get()
        if name in data["experiments"]:
            self._apply_settings(data["experiments"][name])
    def _build_setup_frame(self):
        self.setup_frame = ttk.Frame(self, padding=20)
        self.setup_frame.pack(fill="both", expand=True)

        ttk.Label(self.setup_frame, text="Collagen 3D Quantification Viewer",
                  font=("Segoe UI", 16, "bold")).pack(pady=(0, 20))
        

        self.experiment_name_var = tk.StringVar(value="Default")
        self.mask_folder_var = tk.StringVar()
        self.channel_folder_var = tk.StringVar()
        self.texture_folder_var = tk.StringVar()
        self.df_folder_var = tk.StringVar()
        self.collagen_keyword_var = tk.StringVar(value=DEFAULT_CHANNEL_KEYWORDS["collagen"])
        self.cell_keyword_var = tk.StringVar(value=DEFAULT_CHANNEL_KEYWORDS["cell"])
        self.position_keyword_var = tk.StringVar(value="Pos")
        self.texture_keyword_var = tk.StringVar(value=DEFAULT_CHANNEL_KEYWORDS["texture"])
        self.R_keyword_var = tk.StringVar(value=DEFAULT_CHANNEL_KEYWORDS["R"])
        self.radius_var = tk.StringVar(value=",".join(DEFAULT_RADII))

        exp_frame = ttk.Frame(self.setup_frame)
        exp_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(exp_frame, text="Experiment name:", width=22).pack(side="left")
        self.experiment_combo = ttk.Combobox(
            exp_frame, textvariable=self.experiment_name_var, width=30
        )
        self.experiment_combo.pack(side="left", padx=5)
        self.experiment_combo.bind("<<ComboboxSelected>>", self._on_experiment_selected)
        ttk.Button(exp_frame, text="Delete", command=self.delete_current_experiment).pack(side="left", padx=(5, 0)) 


        def make_row(label_text, var, browse_title):
            row = ttk.Frame(self.setup_frame)
            row.pack(fill="x", pady=8)
            ttk.Label(row, text=label_text, width=22).pack(side="left")
            ttk.Entry(row, textvariable=var, width=65).pack(side="left", padx=5)
            ttk.Button(row, text="Browse...",
                       command=lambda: self._browse_into(var, browse_title)).pack(side="left")

        make_row("Mask folder:", self.mask_folder_var,
                 "Select folder containing mask subfolders")
        make_row("Raw channel folder:", self.channel_folder_var,
                 "Select folder containing raw channel images")
        make_row("Texture folder:", self.texture_folder_var,
                 "Select folder containing texture images")
        make_row("Dataframe folder:", self.df_folder_var,
                 "Select folder containing the trackmate CSVs")

        def make_keyword_row(label_text, var, hint):
            row = ttk.Frame(self.setup_frame)
            row.pack(fill="x", pady=8)
            ttk.Label(row, text=label_text, width=22).pack(side="left")
            ttk.Entry(row, textvariable=var, width=20).pack(side="left", padx=5)
            ttk.Label(row, text=hint, foreground="gray").pack(side="left", padx=(10, 0))

        make_keyword_row(
            "Collagen keyword(s):", self.collagen_keyword_var,
            "filename must contain this (case-insensitive; comma-separate for multiple)",
        )
        make_keyword_row(
            "Cell keyword(s):", self.cell_keyword_var,
            "filename must contain this (case-insensitive; comma-separate for multiple)",
        )
        make_keyword_row(
            "Position keyword(s):", self.position_keyword_var,
            "searched in path (e.g. Pos); LEAVE BLANK if only 1 position exists",
        )
        make_keyword_row(
            "Radius label(s):", self.radius_var,
            "comma-separated, matching mask folder suffix exactly, "
            "e.g. r200um (solid sphere) or r150to200um (shell, inner-to-outer)",
        )
        make_keyword_row(
            "Texture keyword(s):", self.texture_keyword_var,
            "filename must contain this (case-insensitive; comma-separate for multiple)",
        )
        make_keyword_row(
            "R keyword (for texture radius):", self.R_keyword_var,
            "eg. R1,R3",
        )

        zt_frame = ttk.Frame(self.setup_frame)
        zt_frame.pack(pady=10)

        ttk.Label(zt_frame, text="Z slices").grid(row=0, column=0)
        ttk.Entry(zt_frame, textvariable=self.user_z, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(zt_frame, text="Time points").grid(row=0, column=2)
        ttk.Entry(zt_frame, textvariable=self.user_t, width=6).grid(row=0, column=3, padx=5)

        self.load_button = ttk.Button(self.setup_frame, text="Load", command=self._load_everything)
        self.load_button.pack(pady=20)

        self.setup_status = ttk.Label(self.setup_frame, text="", foreground="red")
        self.setup_status.pack()
        self.load_settings()

    def _browse_into(self, var, title):
        folder = filedialog.askdirectory(title=title)
        if folder:
            var.set(folder)

    def _load_everything(self):
        mask_folder = self.mask_folder_var.get().strip()
        channel_folder = self.channel_folder_var.get().strip()
        texture_folder = self.texture_folder_var.get().strip()
        df_folder = self.df_folder_var.get().strip()

        collagen_keywords = parse_keyword_field(self.collagen_keyword_var.get())
        cell_keywords = parse_keyword_field(self.cell_keyword_var.get())
        position_keywords = parse_keyword_field(self.position_keyword_var.get())
        texture_keywords = parse_keyword_field(self.texture_keyword_var.get())
        R_keywords = parse_keyword_field(self.R_keyword_var.get())
        radius_keywords = parse_keyword_field(self.radius_var.get())
        if not mask_folder or not os.path.isdir(mask_folder):
            self.setup_status.config(text="Please choose a valid mask folder.")
            return
        if not channel_folder or not os.path.isdir(channel_folder):
            self.setup_status.config(text="Please choose a valid raw channel folder.")
            return
        if not df_folder or not os.path.isdir(df_folder):
            self.setup_status.config(text="Please choose a valid dataframe folder.")
            return
        if not collagen_keywords:
            self.setup_status.config(text="Please enter at least one collagen keyword.")
            return
        if not cell_keywords:
            self.setup_status.config(text="Please enter at least one cell keyword.")
            return
        if not radius_keywords:
            self.setup_status.config(text="Please enter at least one radius label (e.g. r200um,r150to200um).")
            return
        if not texture_keywords:
            self.setup_status.config(text="Please enter at least one texture keyword.")
            return
        if not R_keywords:
            self.setup_status.config(text="Please enter at least one R keyword for texture radius.")
            return
        self.save_settings()
        self.load_button.config(state="disabled")
        self.setup_status.config(foreground="black", text="Loading... this can take a while for large folders.")
        self.update_idletasks()
        self._load_result_queue = queue.Queue()
        worker = threading.Thread(
            target=self._load_worker,
            args=(mask_folder, channel_folder, texture_folder, df_folder,
                  collagen_keywords, cell_keywords,
                  position_keywords, radius_keywords, texture_keywords, R_keywords),
            daemon=True,
        )
        worker.start()
        self.after(100, self._poll_load_queue)

    def _load_worker(self, mask_folder, channel_folder, texture_folder, df_folder,
                     collagen_keywords, cell_keywords,
                     position_keywords, radius_keywords, texture_keywords, R_keywords):
        
        try:
            grouped_df = build_long_dataframe(
                df_folder, position_keywords=position_keywords, radii=radius_keywords, texture_keywords=texture_keywords, R_keywords=R_keywords
            )
        except Exception as e:
            traceback.print_exc()
            self._load_result_queue.put(("error", f"Error loading dataframes: {e}"))
            return
        try:
            image_lib = ImageLibrary(
                mask_folder, channel_folder, self.user_z.get(), self.user_t.get(),
                channel_keywords={
                    "collagen": collagen_keywords,
                    "cell": cell_keywords,
                    "texture": texture_keywords,
                    "neighbor3d": R_keywords
                },
                position_keywords=position_keywords,
                radii=radius_keywords,
                texture_folder=texture_folder
            )
        except Exception as e:
            traceback.print_exc()
            self._load_result_queue.put(("error", f"Error scanning image folders: {e}"))
            return
        self._load_result_queue.put(("ok", grouped_df, image_lib, radius_keywords))

    def _poll_load_queue(self):
        try:
            result = self._load_result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_load_queue)
            return

        self.load_button.config(state="normal")

        if result[0] == "error":
            self.setup_status.config(foreground="red", text=result[1])
            return

        _, grouped_df, image_lib, radii = result
        self.grouped_df = grouped_df
        self.image_lib = image_lib
        self.radii = radii
        self.radius_colors = {r: RADIUS_PALETTE[i % len(RADIUS_PALETTE)]
                              for i, r in enumerate(self.radii)}

        self.setup_frame.destroy()
        self._build_main_ui()

    def _build_main_ui(self):
        self.main_container = ttk.Frame(self, padding=8)
        self.main_container.pack(fill="both", expand=True)

        controls = ttk.Frame(self.main_container)
        controls.pack(fill="x", pady=(0, 5))

        ttk.Label(controls, text="Position:").grid(row=0, column=0, sticky="w")
        self.pos_combo = ttk.Combobox(controls, textvariable=self.current_pos,
                                      state="readonly", width=12)
        self.pos_combo.grid(row=0, column=1, padx=5)
        self.pos_combo.bind("<<ComboboxSelected>>", self._on_position_change)

        ttk.Label(controls, text="Feature:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.feature_combo = ttk.Combobox(controls, textvariable=self.current_feature,
                                          state="readonly", width=28)
        self.feature_combo.grid(row=0, column=3, padx=5)
        self.feature_combo.bind("<<ComboboxSelected>>", self._on_feature_change)
        

        ttk.Label(controls, text="Statistic:").grid(row=0, column=4, sticky="w", padx=(20, 0))
        self.stat_combo = ttk.Combobox(controls, textvariable=self.current_statistic,
                                       values=[s.title() for s in STATISTICS],
                                       state="readonly", width=10)
        self.stat_combo.grid(row=0, column=5, padx=5)
        self.stat_combo.bind("<<ComboboxSelected>>", self._refresh_plots)

        ttk.Label(controls, text="R:").grid(row=0, column=6, sticky="w", padx=(20, 0))
        self.R_combo = ttk.Combobox(controls, textvariable=self.current_R,
                                    state="readonly", width=10)
        self.R_combo.grid(row=0, column=7, padx=5)
        self.R_combo.bind("<<ComboboxSelected>>", self._on_R_change)

        self.all_R_check = ttk.Checkbutton(
            controls, text="All R", variable=self.all_R_var,
            command=self._on_all_R_toggle,
        )
        self.all_R_check.grid(row=0, column=7, padx=(80, 0))

        ttk.Label(controls, text="Cell(s):").grid(row=0, column=9, sticky="w", padx=(20, 0))
        cell_frame = ttk.Frame(controls)
        cell_frame.grid(row=0, column=10, padx=5)
        self.cell_listbox = tk.Listbox(cell_frame, selectmode="extended",
                                       height=4, exportselection=False, width=10)
        self.cell_listbox.pack(side="left")
        cell_scroll = ttk.Scrollbar(cell_frame, orient="vertical",
                                    command=self.cell_listbox.yview)
        cell_scroll.pack(side="left", fill="y")
        self.cell_listbox.config(yscrollcommand=cell_scroll.set)
        self.cell_listbox.bind("<<ListboxSelect>>", self._on_cell_selection_change)

        self.all_cells_check = ttk.Checkbutton(
            controls, text="All cells", variable=self.all_cells_var,
            command=self._on_all_cells_toggle,
        )
        self.all_cells_check.grid(row=0, column=11, padx=(10, 0))

        ttk.Button(controls, text="Change folders...", command=self._reset_folders).grid(
            row=0, column=12, padx=(30, 0)
        )

        slider_frame = ttk.Frame(self.main_container, padding=(0, 4))
        slider_frame.pack(fill="x")

        ttk.Label(slider_frame, text="Timepoint:").pack(side="left")
        self.t_scale = ttk.Scale(
            slider_frame, from_=0, to=max(0, self.user_t.get() - 1), orient="horizontal",
            variable=self.t_var, command=lambda v: self._on_scale_change(v, self.t_var, self.t_label)
        )
        self.t_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.t_label = ttk.Label(slider_frame, text="0")
        self.t_label.pack(side="left", padx=(0, 20))

        ttk.Label(slider_frame, text="Z-slice:").pack(side="left")
        self.z_scale = ttk.Scale(
            slider_frame, from_=0, to=max(0, self.user_z.get() - 1), orient="horizontal",
            variable=self.z_var, command=lambda v: self._on_scale_change(v, self.z_var, self.z_label)
        )
        self.z_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.z_label = ttk.Label(slider_frame, text="0")
        self.z_label.pack(side="left")

        contrast_frame = ttk.Frame(self.main_container, padding=(0, 4))
        contrast_frame.pack(fill="x")

        ttk.Label(contrast_frame, text="Contrast Settings:").pack(side="left", padx=(0, 10))
        ttk.Label(contrast_frame, text="Channel:").pack(side="left")
        contrast_combo = ttk.Combobox(contrast_frame, textvariable=self.contrast_channel_var,
                                      values=["Collagen", "Cell", "Texture"], state="readonly", width=10)
        contrast_combo.pack(side="left", padx=5)
        contrast_combo.bind("<<ComboboxSelected>>", self._on_contrast_channel_change)

        ttk.Label(contrast_frame, text="Low %:").pack(side="left", padx=(10, 0))
        low_scale = ttk.Scale(
            contrast_frame, from_=0, to=50, orient="horizontal",
            variable=self.contrast_low_var, command=self._on_contrast_value_change,
        )
        low_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.contrast_low_label = ttk.Label(
            contrast_frame, text=f"{self.contrast_low_var.get():.1f}"
        )
        self.contrast_low_label.pack(side="left", padx=(5, 10))

        ttk.Label(contrast_frame, text="High %:").pack(side="left", padx=(10, 0))
        high_scale = ttk.Scale(
            contrast_frame, from_=50, to=100, orient="horizontal",
            variable=self.contrast_high_var, command=self._on_contrast_value_change,
        )
        high_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.contrast_high_label = ttk.Label(
            contrast_frame, text=f"{self.contrast_high_var.get():.1f}"
        )
        self.contrast_high_label.pack(side="left", padx=(5, 0))

        # Matplotlib visualization panes
        main_paned = ttk.PanedWindow(self.main_container, orient="vertical")
        main_paned.pack(fill="both", expand=True, pady=5)

        img_frame = ttk.Frame(main_paned)
        main_paned.add(img_frame, weight=1)

        self.fig_img = Figure(figsize=(12, 4), dpi=100)
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=img_frame)
        self.canvas_img.get_tk_widget().pack(fill="both", expand=True)

        plot_frame = ttk.Frame(main_paned)
        main_paned.add(plot_frame, weight=1)

        self.fig_plot = Figure(figsize=(12, 3), dpi=100)
        self.canvas_plot = FigureCanvasTkAgg(self.fig_plot, master=plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # Setup combo items
        positions = self.image_lib.positions() if self.image_lib else ["Pos0"]
        self.pos_combo["values"] = positions
        if positions:
            self.pos_combo.current(0)

        if self.grouped_df is not None and "feature" in self.grouped_df.columns:
            features = sorted(self.grouped_df["feature"].unique())
            self.feature_combo["values"] = features
            if features:
                self.feature_combo.current(0)
        
        if self.grouped_df is not None and "neighbor3d" in self.grouped_df.columns:
            available = set(self.grouped_df["neighbor3d"].dropna().unique())
            requested = parse_keyword_field(self.R_keyword_var.get())
            R_values = [r for r in requested if r in available] or sorted(available)
            self.R_combo["values"] = R_values
            if R_values:
                self.current_R.set(R_values[0])

                self._on_position_change()

    def _on_position_change(self, event=None):
        if not self.image_lib:
            return
        pos = self.current_pos.get()
        cells = self.image_lib.cells_for_position(pos)

        self.cell_listbox.delete(0, tk.END)
        for c in cells:
            self.cell_listbox.insert(tk.END, c)

        if cells:
            self.cell_listbox.select_set(0)

        self._refresh_images()
        self._refresh_plots()

    def _on_cell_selection_change(self, event=None):
        if self.cell_listbox.curselection():
            self.all_cells_var.set(False)
        self._refresh_images()
        self._refresh_plots()

    def _on_feature_change(self, event=None):
        self._refresh_images()
        self._refresh_plots()

    def _on_R_change(self, event=None):
        self._refresh_images()
        self._refresh_plots()

    def _on_all_cells_toggle(self):
        if self.all_cells_var.get():
            self.cell_listbox.selection_clear(0, tk.END)
        self._refresh_images()
        self._refresh_plots()

    def _on_all_R_toggle(self):
        self.R_combo.config(state="disabled" if self.all_R_var.get() else "readonly")
        self._refresh_plots()

    def _on_scale_change(self, val, var, label):
        v = int(float(val))
        var.set(v)
        label.config(text=str(v))
        self._refresh_images()


    def _on_contrast_channel_change(self, event=None):
        """Switches the spinbox values when changing channels in the dropdown."""
        ch = self.contrast_channel_var.get().lower()
        if ch in self.contrast_values:
            # Temporarily unbind/disable live updates while loading channel values
            self.contrast_low_var.set(self.contrast_values[ch]["low"])
            self.contrast_high_var.set(self.contrast_values[ch]["high"])
            self._refresh_images()

    def _on_contrast_value_change(self, event=None):
        """Safely reads the contrast fields and redraws the images."""
        ch = self.contrast_channel_var.get().lower()
        if ch not in self.contrast_values:
            return

        try:
            low_val = float(self.contrast_low_var.get())
            high_val = float(self.contrast_high_var.get())
        except (tk.TclError, ValueError):
            # Ignore intermediate states while the user is actively backspacing/typing
            return

        # Ensure low percentile stays below high percentile
        if low_val < high_val:
            self.contrast_values[ch]["low"] = low_val
            self.contrast_values[ch]["high"] = high_val
            self._refresh_images()

    def _reset_folders(self):
        if hasattr(self, "main_container"):
            self.main_container.destroy()
        self._build_setup_frame()

    def _get_selected_cells(self):
        if self.all_cells_var.get():
            return [self.cell_listbox.get(i) for i in range(self.cell_listbox.size())]
        sel_indices = self.cell_listbox.curselection()
        if not sel_indices:
            if self.cell_listbox.size() > 0:
                return [self.cell_listbox.get(0)]
            return []
        return [self.cell_listbox.get(i) for i in sel_indices]

    def _get_selected_R_values(self):
        if self.grouped_df is not None and "neighbor3d" in self.grouped_df.columns:
            available = set(self.grouped_df["neighbor3d"].dropna().unique())
        else:
            available = set()
        requested = parse_keyword_field(self.R_keyword_var.get())
        allowed = [r for r in requested if r in available] or sorted(available)

        if self.all_R_var.get():
            return allowed
        current = self.current_R.get()
        return [current] if current in allowed else allowed[:1]

    def _build_masks(self, pos, t, z, selected_cells):
        masks_by_radius = {}
        for r in self.radii:
            combined_mask = None
            for c in selected_cells:
                mf = self.image_lib.find_mask_file(pos, r, c)
                if not mf:
                    continue
                m_stack = self.image_lib.load_mask_stack(mf)
                m_slice = extract_2d_slice(m_stack, t, z, self.user_t.get(), self.user_z.get())
                if m_slice is None:
                    continue
                m = (m_slice > 0)
                combined_mask = m if combined_mask is None else np.logical_or(combined_mask, m)
            if combined_mask is not None:
                masks_by_radius[r] = combined_mask.astype(np.uint8)
        return masks_by_radius

    def _refresh_images(self):
        print("Refreshing images...")
        if not self.image_lib:
            return
        pos = self.current_pos.get()
        t = self.t_var.get()
        z = self.z_var.get()
        current_feature = self.current_feature.get()
        current_R = self.current_R.get()

        texture_kw = [kw for kw in [current_feature, current_R] if kw]
        channel_overrides = {"texture": texture_kw or None}

        selected_cells = self._get_selected_cells()
        masks_by_radius = self._build_masks(pos, t, z, selected_cells)

        imgs, norms, overlays = {}, {}, {}
        for ch in ("collagen", "cell", "texture"):
            print(f"Loading channel '{ch}' for position '{pos}', timepoint {t}, z-slice {z}..., R: {current_R}")
            f = self.image_lib.find_channel_file(ch, pos, t, current_R, override_keywords=channel_overrides.get(ch), match_all = (ch=="texture"))
            img = None
            if f:
                stack = self.image_lib.load_channel_stack(f)
                img = extract_2d_slice(stack, t, z, self.user_t.get(), self.user_z.get())
            c = self.contrast_values[ch]
            imgs[ch] = img
            norms[ch] = normalize_for_display(img, c["low"], c["high"])
            overlays[ch] = build_overlay_rgb(img, masks_by_radius, self.radii, self.radius_colors,
                                            p_low=c["low"], p_high=c["high"])

        self.fig_img.clear()
        axes = self.fig_img.subplots(1, 6)
        panels = [
            (norms["collagen"], "Collagen Raw", False),
            (norms["cell"], "Cell Raw", False),
            (norms["texture"], "Texture Raw", False),
            (overlays["collagen"], "Collagen Overlay", True),
            (overlays["cell"], "Cell Overlay", True),
            (overlays["texture"], "Texture Overlay", True),
        ]
        for ax, (data, title, _) in zip(axes, panels):
            ax.imshow(data, cmap=None if _ else "gray")
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        self.fig_img.tight_layout()
        self.canvas_img.draw_idle()

    def _refresh_plots(self, event=None):
        if self.grouped_df is None or self.grouped_df.empty:
            return

        pos = self.current_pos.get()
        feat = self.current_feature.get()
        stat = self.current_statistic.get().lower()
        R_values = self._get_selected_R_values()

        df_pos_feat = self.grouped_df[
            (self.grouped_df["position"] == pos) &
            (self.grouped_df["feature"] == feat) &
            (self.grouped_df["statistic"] == stat) &
            (self.grouped_df["neighbor3d"].isin(R_values))
        ]

        self.fig_plot.clear()

        n_radii = len(self.radii)
        if n_radii == 0:
            self.canvas_plot.draw_idle()
            return

        axes = self.fig_plot.subplots(1, n_radii, sharey=True)
        if n_radii == 1:
            axes = [axes]

        linestyles = ['-', '--', ':', '-.']
        selected_cells = self._get_selected_cells()

        for idx, r in enumerate(self.radii):
            ax = axes[idx]
            df_r = df_pos_feat[df_pos_feat["mask_type"] == r]

            if not df_r.empty and selected_cells:
                for r_idx, R_val in enumerate(R_values):
                    df_R = df_r[df_r["neighbor3d"] == R_val]
                    ls = linestyles[r_idx % len(linestyles)]
                    for cell_id in selected_cells:
                        df_cell = df_R[df_R["cell"] == str(cell_id)].sort_values("timepoint")
                        if not df_cell.empty:
                            label = f"Cell {cell_id}"
                            if len(R_values) > 1:
                                label += f" ({R_val})"
                            ax.plot(
                                df_cell["timepoint"],
                                df_cell["value"],
                                marker=".",
                                linewidth=1.5,
                                linestyle=ls,
                                label=label,
                            )

            ax.set_title(f"Radius: {r}", fontsize=10)
            ax.set_xlabel("Timepoint")
            if idx == 0:
                ax.set_ylabel(feat, fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)

            n_lines = len(selected_cells) * max(1, len(R_values))
            if 1 < n_lines <= 10:
                ax.legend(fontsize="xx-small", loc="best")

        self.fig_plot.tight_layout()
        self.canvas_plot.draw_idle()


if __name__ == "__main__":
    app = CollagenViewerApp()
    app.mainloop()