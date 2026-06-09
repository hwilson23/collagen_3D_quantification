# collagen_3D_multimetric

# combining multiple software outputs for collagen analysis
-workflow for aggregating collagen data from several sources (CurveAlign, TWOMBLI, texture) and parsing

-parse_into_dataframe_NOslices.py aggregrates data by stack

-parse_into_dataframe_YESslices.py aggregrates data per image slice (also has the 3D glcm functions)

-fanalyze_dataframe.ipynb runs test plots on exported full .csv dataframe

-RF_regression.py trains random forest regressor on .csv dataframe to separate given condition and ranks importance of features (assembled from the different software outputs)


-spiral files are ground truth data that can be used for glcm comparisons and testing



# 3D GLCM Texture Analysis

MATLAB implementation of voxelwise 3D Gray-Level Co-occurrence Matrix (GLCM) texture feature extraction from volumetric fluorescence microscopy images.

Derived from:
- https://github.com/Pedram-Parnianpour/VGLCM-TOP-3D-Texture-Analysis
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0117759
- claude (sonnet 4.6) used to help modify code.

Requires:
- **Image Processing Toolbox** 
- **Parallel Computing Toolbox** 

### Input Data Format
- 3D image stacks saved as multi-page `.tif` files
- The mask is derived directly from the image — any voxel with intensity `> 0` is treated as inside the mask. Pre-apply your mask to the image before running.

| Distance | `OPT.D` | Radius for directional offset vectors. Controls how far apart co-occurring voxel pairs can be. |
| Neighborhood size | `OPT.NeighborSize` | Radius for local neighborhood averaging around each voxel. |
| Quantization level | `OPT.quantLevel` | Number of gray level bins. Typical values: `8`, `16`, `32` |
| Features | `OPT.glcm_properties` | Cell array of feature name strings to compute. |
| Input folder | `folder` | Full path to folder containing input `.tif` files |
| Output folder | `outpath` | Full path to folder where output `.tif` files and cached `.mat` files will be saved |

---
## Supported Texture Features

Pass any combination of these strings in `OPT.glcm_properties`:

| String | Feature |
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


## Memory Considerations

Memory usage scales with: numMaskedVoxels × numOffsets × (numNeighbors + 1)
Start with `OPT.quantLevel = 8` for testing, increase to `16` or `32` for final analysis
`parfor` parallelises across files — performance scales with the number of input files and available CPU cores

---

## Functions
| `ComputeOffsets(dis)` | Generates all 3D integer offset vectors within a sphere of radius `dis` |
| `AllOffsetsAllNeighbors(I, D, NeighborSizeRad, mask)` | Precomputes start/end voxel index mappings for all offset and neighbor combinations |
| `CreateGLCM_Local(I, NL, GL, D, NeighborSizeRad, mask)` | Builds a normalized local GLCM for every masked voxel |
| `computeGLCMLocalFeat(glcm, mask, GLCM_feat)` | Iterates over masked voxels and extracts texture features from each local GLCM |
| `computeFeature(glcm, GLCM_feat_all)` | Computes all requested features from a single voxel's 2D GLCM |
