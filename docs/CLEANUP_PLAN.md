# Project Cleanup Plan & Checklist

This document tracks work to make this codebase reproducible and portable. Do one step at a time so each change is independently reversible.

---

## Step 1 — Unified Config ✅ DONE

All hardcoded paths replaced. Single source of truth lives in `config.json` at the project root.

- `config.json` — edit here for a new machine or experiment
- `config.py` — thin Python wrapper; exposes `PATHS`, `EXTERNAL`, `PARAMS`
- `src/calculate_glcm3D/glcm_config.m` — thin MATLAB wrapper; exposes `CFG` struct

Python scripts modified: `applyareamask.py`, `splitstack_toindividualimg.py`, `ctfireparser.py`, `parse_into_dataframe_YESslices.py`, `parse_into_dataframe_NOslices.py`

MATLAB scripts modified: `glcm3D_attempt.m`, `GLCM_TOP_attempt.m`, `glcm3D_attempt_withquadrants.m`, `glcm3D_attempt_gpu.m`

---

## Step 2 — Add `if __name__ == "__main__"` Guards

**Why:** Several scripts execute immediately on import because all their logic sits at the top level. This means you can't import helper functions from them without triggering a full pipeline run, and it makes testing individual functions impossible.

**Scripts to fix:**

- [ ] `splitstack_toindividualimg.py` — wrap body in `def main(): ...` + guard
- [ ] `parse_into_dataframe_YESslices.py` — wrap `## main` section
- [ ] `parse_into_dataframe_NOslices.py` — wrap `## main` section
- [ ] `ctfireparser.py` — already has a `SimpleNamespace`-based args block; wrap it

**Pattern to use in each:**

```python
def main():
    # existing top-level code goes here

if __name__ == "__main__":
    main()
```

---

## Step 3 — `.gitignore` Cleanup

**Why:** Without this, MATLAB cache files, Python bytecode, and generated outputs can accidentally get committed and bloat the repo.

- [ ] Add `.gitignore` at project root with:

  ```gitignore
  # Python
  __pycache__/
  *.pyc
  .ipynb_checkpoints/
  .venv/

  # MATLAB
  *.mat
  *.asv

  # Generated outputs
  data/*.xlsx
  data/*_temp*
  data/*_stride*
  output/

  # OS
  .DS_Store
  Thumbs.db
  ```

- [ ] Remove root-level duplicate CSVs (`*_n2.csv` files) — keep only the copies in `data/`
- [ ] Remove temp files: `data/*_temp*.xlsx`, `data/*_stride5*`

---

## Step 4 — Restructure `src/` to Match Pipeline Steps

**Why:** The current folder names (`collagen_dataframe_timeseries/`, `collagen_dataframe/`) don't map to the pipeline steps described in `working-pipeline.md`. A newcomer can't tell which folder to look in.

**Proposed rename:**

```text
src/
  calculate_glcm3D/              → step3_glcm3d/
  collagen_dataframe_timeseries/ → split across:
      step1_preprocess/          (splitstack_toindividualimg.py)
      step2_masking/             (applyareamask.py)
      step4_fiber_analysis/      (ctfireparser.py, ctfire_statssummary.py)
      step7_aggregation/         (parse_into_dataframe_YESslices.py, _NOslices.py)
  collagen_dataframe/            → step7_aggregation/  (RF_regression.py, RF_classifier.py, notebook)
  ground_truth_data/             → keep name (it's self-explanatory)
```

**Steps to execute this safely:**

1. Create the new folders under `src/`
2. Move files one folder at a time (start with `step1_preprocess/` — only one file)
3. Update `sys.path.insert(0, str(Path(__file__).parents[N]))` in each moved script — the parent count may change if nesting depth changes
4. Test each moved script runs before moving the next folder
5. Delete the old empty folders last

**Note on `sys.path` depth:** Scripts currently two levels below project root (`src/collagen_dataframe_timeseries/`) use `.parents[2]`. After the rename the depth stays the same (`src/step1_preprocess/`), so no path changes needed for those. Verify before and after.

---

## Step 5 — Add `argparse` CLI to Server Scripts

**Why:** `ctfireparser.py` and the GLCM scripts are meant to run on a server or compute cluster. Hardwired args (currently via `SimpleNamespace`) can't be overridden without editing the file.

- [ ] `ctfireparser.py` — replace `SimpleNamespace` block with `argparse`:

  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-dir",  default=str(PATHS["ctfire_out"]))
  parser.add_argument("--output-dir", default=str(PATHS["ctfire_results"]))
  parser.add_argument("--z-step",     type=float, default=PARAMS["z_step"])
  parser.add_argument("--stacks",     nargs="+",  default=PARAMS["stacks"])
  args = parser.parse_args()
  ```

  Config values become defaults; command-line overrides them when needed.

- [ ] Consider the same treatment for `parse_into_dataframe_YESslices.py` if it ever runs non-interactively

---

## Remaining Low-Priority Items

These improve quality but don't block portability.

- [ ] **Fix double-nested directory** — `collagen_3D_multimetric-main/collagen_3D_multimetric-main/` — the outer wrapper folder is an artifact of downloading a GitHub zip. Either flatten or document that the inner folder is the actual project root.

- [ ] **Consolidate the two parse scripts** — `parse_into_dataframe_YESslices.py` and `NOslices.py` share >80% of their code. Refactor into one script with a `--per-slice` flag, or extract shared functions into a `pipeline_utils.py` module.

- [ ] **Make ML column-dropping data-driven** — `RF_regression.py` and `RF_classifier.py` drop columns by hardcoded name list. Replace with: drop non-numeric columns or columns matching a pattern, so upstream renames don't silently break the model.

- [ ] **Move notebook** — `analyze_dataframe.ipynb` (currently in `src/collagen_dataframe/`) belongs in a top-level `notebooks/` folder — notebooks are not source code.

- [ ] **Pin tested environment** — note the MATLAB release version (R20XXa/b) and Python version (≥3.13) used, and add it to `docs/OVERVIEW.md`.

- [ ] **Smoke test with spiral data** — document that running the pipeline on `ground_truth_data/` spiral TIFs verifies a new installation.
