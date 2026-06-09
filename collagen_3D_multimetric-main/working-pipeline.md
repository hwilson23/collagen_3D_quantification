# Collagen 3D Multimetric Pipeline

**Repos:** [hwilson23/collagen_3D_multimetric-main](https://github.com/hwilson23/collagen_3D_multimetric-main) · [hwilson23/image_texture_analysis](https://github.com/hwilson23/image_texture_analysis)

---

## Acquisition

3-channel OME-TIFF time-series acquired via MDA (Multi-Dimensional Acquisition):

| Channel | Excitation | Label |
|---|---|---|
| NucBlue | 740 nm | Cell autofluorescence |
| Fluorescent collagen | 800 nm | No-SHG collagen marker |
| SHG collagen | 890 nm | Forward/backward SHG |

- **740 nm** z-stacks also acquired **before and after** the main time-series as a cell position reference.
- Preset name: `3ch-back` — 3-channel collection with backward SHG geometry. *(details TBD)*

---

## Step 1 — OME-TIFF Preprocessing

**Script:** `splitstack_toindiv_image.py`

- Split OME-TIFF by channel and Z/T dimensions
- Separate time-series into individual TIFFs: `T1, T2, T3, ...`
- Apply background subtraction; encode background value in filename
- Add `tX_sX` suffix for time index and z-slice index

**Reference:** `serp.mm-timestack.m` (MATLAB serpentine time-stack utility)

---

## Step 2 — ROI Masking

**Masks defined manually in ImageJ** (`.ijm` macro) based on the 740 nm z-stack at the final time point.

Three concentric area regions per field:

| Label | Definition |
|---|---|
| **A0** | Cell body (direct cell mask) |
| **A1** | Peri-cellular zone (A0 + dilation) |
| **A2** | Extracellular zone (A0 + further dilation) |

**Script:** `apply_aramask.py` — applies A0/A1/A2 masks to preprocessed images.

---

## Step 3 — 3D GLCM Texture Analysis

**Script:** MATLAB (`glcm_3d_attempt.m`), derived from [VGLCM-TOP](https://github.com/Pedram-Parnianpour/VGLCM-TOP-3D-Texture-Analysis)

- Run on server (memory-intensive)
- Stack split into **4 quadrants** to fit in memory, then recombined with `combine_quadimag.py`
- Outputs: `.mat` cache + per-feature `.tif` stacks
- TOP variant: three orthogonal planes; orthogonal-plane slices removed from output
- Kernel parameters (distance, neighborhood size, quantization level) encoded in output filename by Helen
- Applied independently to A0, A1, A2 masked images

---

## Step 4 — CurveAlign / CT-FIRE Fiber Analysis

**Script:** `ctfire_parser.py` — parses CT-FIRE fiber output files for each area Aₙ

| Script | Aggregation |
|---|---|
| `parse_into_dataframe_YESslices.py` | Per z-slice |
| `parse_into_dataframe_NOslices.py` | Full stack summary |

---

## Step 5 — TWOMBLI

Run on desktop via `twombli-v1` (ImageJ macro + XML defaults)

- Input: masked image (A0/A1/A2 applied)
- Outputs: HDM map, binary mask, `twombli_results.csv`

---

## Step 6 — Image Texture Analysis (pixel-mapped)

**Repo:** [hwilson23/image_texture_analysis](https://github.com/hwilson23/image_texture_analysis)

| Script | Purpose |
|---|---|
| `textureanalysis_map.py` | Pixel-mapped GLCM texture output |
| `textureanalysis_computeglcmonly.py` | GLCM computation only |
| `textrue_outputs_analysis.py` | Analysis of texture outputs *(filename has typo)* |

Uses `scyjava` for ImageJ interop.

---

## Step 7 — Aggregation & Analysis

**Repo:** [hwilson23/collagen_3D_multimetric-main](https://github.com/hwilson23/collagen_3D_multimetric-main)

| Script | Purpose |
|---|---|
| `parse_into_dataframe_YESslices.py` | Aggregate all outputs per z-slice |
| `parse_into_dataframe_NOslices.py` | Aggregate all outputs as stack summary |
| `fanalyze_dataframe.ipynb` | Exploratory plots on exported `.csv` |
| `RF_regression.py` | Random forest regressor to separate conditions and rank feature importance |
