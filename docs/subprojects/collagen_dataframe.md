# Subproject: collagen_dataframe

**Location:** `src/collagen_dataframe/`
**Language:** Python 3.13+
**Role:** Core data integration, parsing, and machine learning pipeline

---

## Purpose

Aggregates outputs from CurveAlign, TWOMBLI, and texture analysis into a single tidy dataframe, then trains Random Forest models to rank which features best distinguish gel concentrations.

---

## Scripts

### Data Parsing & Integration

#### `ctfireparser.py`
Parses CurveAlign / ctFIRE histogram output files for a single image stack.

- Discovers four histogram types per slice: `HistANG` (angle), `HistLEN` (length), `HistSTR` (straightness), `HistWID` (width)
- Computes per-slice statistics: mean, median, std, quartiles, fiber count, z-depth
- Generates three plot types: mean vs z-depth, violin distributions, fiber count vs z-depth
- Assembles per-slice overlay images into a multi-page TIFF z-stack

**Inputs:** `Hist*.csv` files and `OL_*.tif` overlay images from a CurveAlign output folder  
**Outputs:** statistics CSV, PNG plots, multi-page TIFF

---

#### `ctfire_statssummary.py`
Groups ctFIRE per-slice statistics across multiple stacks.

- Reads the per-slice CSV produced by `ctfireparser.py`
- Groups by concentration and ROI
- Produces a pivot table of mean fiber metric values

---

#### `parse_into_dataframe_YESslices.py`
Main pipeline script — builds the final per-slice merged dataframe.

**Steps:**
1. Load CurveAlign CSV → `reshape_CA()` → pivoted fiber metrics per slice
2. Load TWOMBLI CSV → `twombli_slice_data()` → slice number correction applied
3. Load 2D texture TIF folder → `process_img_folder(is_3d=False)` → per-slice texture stats
4. Load 3D texture TIF folder → `process_img_folder(is_3d=True)` → per-slice 3D texture stats
5. Merge all four sources on `[image_name, slice]`
6. Detect and collapse duplicate columns with `find_identical_columns()` / `collapse_identical_columns()`
7. Split by `type` column (FLU vs SHG) and export separate CSVs

**Outputs:**
- `final_dataframe_byslice_FLU.csv`
- `final_dataframe_byslice_SHG.csv`
- `finalcollapsed_dataframe_byslice.csv`

---

#### `parse_into_dataframe_NOslices.py`
Stack-level (collapsed) variant of the pipeline above.

- Aggregates TWOMBLI measurements by averaging across slices within a stack
- Produces one row per image stack instead of one row per slice
- Otherwise the same merge logic as the YESslices script

---

### Machine Learning

#### `RF_regression.py`
Random Forest regressor to predict gel concentration (1 / 2 / 3 mg/ml).

- Loads the final collapsed dataframe
- Subsamples every 5th slice to reduce redundancy within a stack
- Uses Leave-One-Group-Out cross-validation (groups = `image_name`) to prevent data leakage
- Trains on 100 trees
- Computes permutation feature importance ranked by mean accuracy decrease
- Compares full model vs. top-10 features reduced model

**Metrics reported:** R², MSE  
**Output:** feature importance bar plot

---

#### `RF_classifier.py`
Random Forest classifier to separate FLU from SHG modality (multi-class).

- Same CV strategy as the regressor (LOGO)
- Uses balanced accuracy and raw accuracy scoring
- Top-10 feature selection and comparison

---

### Exploratory Analysis

#### `analyze_dataframe.ipynb`
Jupyter notebook for interactive EDA.

- Loads the final collapsed dataframe
- Correlation heatmap of all numeric features
- Violin plots of features vs. concentration, colored by gel type (FLU / SHG)
- Manual feature filtering and selection experiments

---

## Feature Categories in the Final Dataframe

| Category | Source | Example columns |
|---|---|---|
| Fiber geometry | CurveAlign | `mean_HistANG`, `std_HistLEN`, `median_HistSTR` |
| Bulk morphology | TWOMBLI | `Area`, `Lacunarity`, `Alignment`, `Branchpoints` |
| 2D texture | Texture tool | `texture_mean_contrastmean`, `texture_mean_entropymean` |
| 3D texture | MATLAB GLCM | `texture3d_mean_autoc`, `texture3d_median_entro` |
| Metadata | Pipeline | `image_name`, `slice`, `concentration`, `type` |

---

## Known Issues / Hardcoded Items

- All input/output paths are absolute paths pointing to the original developer's machine (`G:/`, `C:\Users\hwilson23\...`) — must be updated before running
- Feature column names dropped in the ML scripts are hardcoded by name (brittle to upstream column renames)
- FLU/SHG split happens at the end of the merge pipeline, not at the source
