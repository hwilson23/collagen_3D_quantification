# CLAUDE.md — Guidance for AI Assistants

This file tells Claude (and similar tools) what this project is, how it's structured, and what to watch out for when making changes.

---

## What This Project Is

A multi-language research pipeline for analyzing 3D collagen gel microscopy images. It combines outputs from three external tools — CurveAlign/ctFIRE, TWOMBLI, and MATLAB GLCM texture analysis — into a unified feature dataframe, then runs Random Forest ML to rank which features distinguish gel concentrations and imaging modalities.

**Language split:** Python handles data parsing, merging, and ML. MATLAB handles 3D texture feature extraction (compute-heavy, runs on a server).

See [OVERVIEW.md](OVERVIEW.md) for data flow and subproject details.

---

## Configuration — Always Start Here

**`config.json`** (project root) is the single source of truth for all paths and parameters. Both Python and MATLAB read it natively — do not add a second config file.

- **Python scripts** import via `from config import PATHS, EXTERNAL, PARAMS` — `config.py` is a thin wrapper that reads `config.json` and builds `pathlib.Path` objects.
- **MATLAB scripts** call `run('glcm_config.m')` which loads `config.json` via `jsondecode` and exposes a `CFG` struct.

When paths need to change (new machine, new experiment folder), edit **only** `config.json`.

---

## Project Structure

```
collagen_3D_multimetric-main/
├── config.json                        ← edit here for new machine/experiment
├── config.py                          ← Python wrapper (do not edit)
├── src/
│   ├── collagen_dataframe_timeseries/ ← Python pipeline scripts
│   │   ├── applyareamask.py           ← Step 2: apply A0/A1/A2 masks
│   │   ├── splitstack_toindividualimg.py ← Step 1: split OME-TIFF
│   │   ├── ctfireparser.py            ← Step 4: parse ctFIRE fiber CSVs
│   │   ├── parse_into_dataframe_YESslices.py ← Step 7: per-slice merge
│   │   └── parse_into_dataframe_NOslices.py  ← Step 7: stack-level merge
│   ├── collagen_dataframe/            ← ML scripts
│   │   ├── RF_regression.py
│   │   └── RF_classifier.py
│   └── calculate_glcm3D/              ← MATLAB texture scripts
│       ├── glcm_config.m              ← MATLAB wrapper (do not edit)
│       ├── glcm3D_attempt.m           ← endpoint experiment variant
│       ├── GLCM_TOP_attempt.m         ← three-orthogonal-planes variant
│       ├── glcm3D_attempt_withquadrants.m ← timeseries, split into 4 quadrants
│       └── glcm3D_attempt_gpu.m       ← GPU-accelerated variant
├── docs/
│   ├── OVERVIEW.md
│   ├── CLAUDE.md                      ← this file
│   └── subprojects/
└── working-pipeline.md                ← step-by-step pipeline reference
```

---

## Python Import Pattern

Scripts in `src/collagen_dataframe_timeseries/` are two levels below the project root. They add the root to `sys.path` so `config.py` is importable:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import PATHS, EXTERNAL, PARAMS
```

Do not use relative imports or hardcode any paths inside scripts.

---

## MATLAB Config Pattern

Each MATLAB script loads its config at the top with a single `run()` call. The script then accesses everything through the `CFG` struct:

```matlab
run(fullfile(fileparts(mfilename('fullpath')), 'glcm_config.m'));
% CFG.folder_ep, CFG.outpath_ep, CFG.D, CFG.NeighborSize, etc.
```

`mfilename('fullpath')` makes this work regardless of MATLAB's current working directory.

---

## Key Domain Facts

- **Mask regions:** A0 = cell body, A1 = pericellular zone, A2 = extracellular zone. Applied to images before GLCM and TWOMBLI.
- **GLCM quadrant split:** Large stacks are split into 4 quadrants to fit in memory, then recombined with `combine_quadimag.py`.
- **GLCM outputs:** `.mat` cache file + per-feature multi-page `.tif` stacks. Feature name is encoded in the output filename.
- **ctFIRE outputs:** Histogram CSVs per z-slice; `ctfireparser.py` aggregates these before the dataframe merge.
- **TWOMBLI:** Runs separately as an ImageJ macro; its output CSV is an external input to the Python pipeline.
- **Modalities:** FLU (fluorescent collagen, 800 nm) and SHG (second harmonic generation, 890 nm) are kept separate through analysis and produce separate output CSVs.

---

## What Not To Do

- Do not hardcode any absolute paths in scripts — all paths go in `config.json`.
- Do not add a YAML config — MATLAB has no built-in YAML parser. JSON works natively in both languages.
- Do not run `addpath` with absolute paths in MATLAB scripts (was previously hardcoded; now removed).
- The `.venv/` directory is not tracked by git and should not be committed.
- Raw image data and `.mat` files are large and not stored in git.

---

## Running the Pipeline

**Python setup:**
```
uv sync          # installs dependencies from pyproject.toml
```

**Python scripts** are run directly as modules. Most are currently top-level scripts (no `if __name__ == "__main__"` guard), so import them with care.

**MATLAB scripts** are run from within MATLAB. Open `src/calculate_glcm3D/` and run the appropriate `glcm3D_*.m` script. Config is loaded automatically.

**External tools** (CurveAlign/ctFIRE, TWOMBLI) are run separately and their output CSVs are pointed to via `config.json`.