# Collagen 3D Multimetric — Project Overview

## What This Project Does

This project builds a unified analysis pipeline for 3D collagen gel microscopy. It pulls measurements from three independent image analysis tools, merges them into a single feature dataframe, then uses machine learning to determine which features best distinguish gel concentrations and imaging modalities.

**Biological context:** Collagen hydrogels at 1, 2, and 3 mg/ml are imaged with two microscopy modalities — fluorescence (FLU) and Second Harmonic Generation (SHG). The goal is to understand how gel structure changes with concentration and which quantitative metrics capture those differences.

---

## Three Input Sources

| Tool | What it measures | Output format |
|---|---|---|
| **CurveAlign / ctFIRE** | Individual fiber geometry (angle, length, straightness, width) | Histogram CSV files per slice |
| **TWOMBLI** | Bulk morphology (area, lacunarity, alignment, branchpoints, endpoints) | Summary CSV per stack |
| **Texture analysis** | 2D & 3D Gray-Level Co-occurrence Matrix (GLCM) texture features | Multi-page TIFF stacks |

---

## Data Flow

```
CurveAlign CSV (fiber histograms)
        ↓ ctfireparser.py
TWOMBLI CSV (morphology)          →  parse_into_dataframe_YESslices.py
        ↓                                        ↓
2D Texture TIF stacks             →  merged per-slice dataframe
3D Texture TIF stacks (MATLAB)    →  (FLU and SHG separated)
                                             ↓
                                 final_dataframe_byslice_FLU.csv
                                 final_dataframe_byslice_SHG.csv
                                 finalcollapsed_dataframe_byslice.csv
                                             ↓
                              ┌──────────────┼──────────────┐
                              ↓              ↓               ↓
                        RF_regression   RF_classifier  analyze_dataframe
                        (predict conc)  (classify type) (EDA & plots)
```

---

## Subprojects

| Subproject | Language | Purpose |
|---|---|---|
| [`collagen_dataframe`](subprojects/collagen_dataframe.md) | Python | Data integration, parsing, ML |
| [`calculate_glcm3D`](subprojects/calculate_glcm3D.md) | MATLAB | 3D voxelwise texture feature extraction |
| [`ground_truth_data`](subprojects/ground_truth_data.md) | Python | Synthetic spiral data for GLCM validation |

---

## Output Files

| File | Contents |
|---|---|
| `data/final_dataframe_byslice_FLU.csv` | Per-slice features for fluorescence modality |
| `data/final_dataframe_byslice_SHG.csv` | Per-slice features for SHG modality |
| `data/finalcollapsed_dataframe_byslice.csv` | Combined FLU + SHG, deduplicated columns |

Each row is one image slice. Columns span fiber metrics, morphology metrics, 2D texture features, and 3D texture features (~90 columns total).

---

## Dependencies

**Python** (managed with `uv`, requires Python ≥ 3.13):
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (Random Forest, cross-validation)
- `tifffile` (multi-page TIFF I/O)
- `openpyxl` (Excel export)

**MATLAB**:
- Image Processing Toolbox
- Parallel Computing Toolbox
- (Optional) GPU Computing Toolbox

**External software** (not in repo — provides raw inputs):
- CurveAlign / ctFIRE
- TWOMBLI
