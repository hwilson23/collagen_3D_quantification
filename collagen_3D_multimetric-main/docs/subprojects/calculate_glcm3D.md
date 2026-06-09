# Subproject: calculate_glcm3D

**Location:** `src/calculate_glcm3D/`
**Language:** MATLAB
**Role:** Voxelwise 3D Gray-Level Co-occurrence Matrix (GLCM) texture feature extraction

---

## Purpose

Computes local 3D GLCM texture features for every masked voxel in a volumetric fluorescence microscopy image. Produces one output TIFF per texture feature, which the Python pipeline then reads as a texture map.

Derived from:
- [VGLCM-TOP-3D-Texture-Analysis](https://github.com/Pedram-Parnianpour/VGLCM-TOP-3D-Texture-Analysis)
- [Parnianpour et al. 2015 (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0117759)

---

## Scripts

### `glcm3D_attempt.m` — CPU version (primary)

Reads all `.tif` stacks in a folder and computes 3D GLCM texture maps in parallel across files.

**Workflow:**
1. Read multi-page TIFF as a 3D array
2. Build mask: any voxel with intensity > 0 is inside the ROI
3. Quantize intensity values to `OPT.quantLevel` gray levels
4. For each masked voxel, compute offsets within a sphere of radius `OPT.D` and a local neighborhood of radius `OPT.NeighborSize`
5. Build a normalized local GLCM per voxel
6. Extract all requested texture features from each local GLCM
7. Assemble per-feature 3D maps and save as multi-page TIFFs
8. Cache intermediate GLCM matrices in `.mat` files to skip recomputation on re-run

**Parallelization:** `parfor` loop across input files — scales with CPU core count.

---

### `glcm3D_attempt_gpu.m` — GPU-accelerated version

Same logic as the CPU version but uses CUDA `gpuArray` to accelerate the inner GLCM computation. Results are gathered back to CPU for file I/O. Requires a CUDA-capable GPU and MATLAB's GPU Computing Toolbox.

> Note: this file contains hardcoded paths to the original developer's machine. Update `folder` and `outpath` before use.

---

## Configuration Parameters

| Parameter | Variable | Description |
|---|---|---|
| Distance | `OPT.D` | Radius for directional offset vectors. Controls how far apart co-occurring voxel pairs can be. |
| Neighborhood size | `OPT.NeighborSize` | Radius for local neighborhood averaging around each voxel. |
| Quantization level | `OPT.quantLevel` | Number of gray level bins. Typical values: `8`, `16`, `32`. Start with `8` for testing. |
| Features | `OPT.glcm_properties` | Cell array of feature name strings to compute (see table below). |
| Input folder | `folder` | Full path to folder containing input `.tif` files. |
| Output folder | `outpath` | Full path to folder for output TIFs and cached `.mat` files. |

---

## Input / Output

**Input:** Multi-page `.tif` files (one file = one 3D image stack). The mask is derived from the image — pre-apply your mask before running (set background voxels to 0).

**Output per input file:**
- One multi-page `.tif` per texture feature (e.g., `autoc.tif`, `entro.tif`)
- One `.mat` cache file with intermediate GLCM data

---

## Supported Texture Features (22 total)

| Key | Feature |
|---|---|
| `autoc` | Autocorrelation |
| `contr` | Contrast |
| `corrm` | Correlation (MATLAB definition) |
| `corrp` | Correlation (paper definition) |
| `cprom` | Cluster Prominence |
| `cshad` | Cluster Shade |
| `dissi` | Dissimilarity |
| `energ` | Energy |
| `entro` | Entropy |
| `homom` | Homogeneity (MATLAB definition) |
| `homop` | Homogeneity (paper definition) |
| `maxpr` | Maximum Probability |
| `sosvh` | Sum of Squares Variance |
| `savgh` | Sum Average |
| `svarh` | Sum Variance |
| `senth` | Sum Entropy |
| `dvarh` | Difference Variance |
| `denth` | Difference Entropy |
| `inf1h` | Information Measure of Correlation 1 |
| `inf2h` | Information Measure of Correlation 2 |
| `indnc` | Inverse Difference Normalized |
| `idmnc` | Inverse Difference Moment Normalized |

---

## Internal Functions

| Function | Description |
|---|---|
| `ComputeOffsets(dis)` | Generates all 3D integer offset vectors within a sphere of radius `dis` |
| `AllOffsetsAllNeighbors(I, D, NeighborSizeRad, mask)` | Precomputes start/end voxel index mappings for all offset and neighbor combinations |
| `CreateGLCM_Local(I, NL, GL, D, NeighborSizeRad, mask)` | Builds a normalized local GLCM for every masked voxel |
| `computeGLCMLocalFeat(glcm, mask, GLCM_feat)` | Iterates over masked voxels and extracts texture features from each local GLCM |
| `computeFeature(glcm, GLCM_feat_all)` | Computes all requested features from a single voxel's 2D GLCM |

---

## Memory Considerations

Memory usage scales with:
```
numMaskedVoxels × numOffsets × (numNeighbors + 1)
```

- Start with `OPT.quantLevel = 8` for testing; increase to `16` or `32` for final analysis
- Reduce `OPT.D` and `OPT.NeighborSize` if running out of memory
- `parfor` parallelizes across files — performance scales with the number of input files and available CPU cores

---

## MATLAB Requirements

- **Image Processing Toolbox** (imread, imwrite, TIFF operations)
- **Parallel Computing Toolbox** (parfor)
- **GPU Computing Toolbox** (GPU version only, optional)
