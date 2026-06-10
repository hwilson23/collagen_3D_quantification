# Subproject: ground_truth_data

**Location:** `src/ground_truth_data/`
**Language:** Python
**Role:** Synthetic spiral data for validating 3D GLCM texture functions

---

## Purpose

Generates synthetic 3D image volumes containing tube-shaped spiral structures with known geometry. Used to validate that the GLCM texture analysis correctly captures structural properties before applying it to real microscopy data.

---

## Script

### `create_spiral_data.py`

Creates an Archimedean spiral centerline in 3D space, fills voxels within a tube radius of the centerline, and applies an intensity gradient for a realistic appearance.

**Parameters:**
- Volume size (voxels)
- Voxel size (physical units)
- Tube radius
- Number of spiral turns
- Maximum spiral radius
- Z-spread (flat 2D spiral vs. helix rising through Z)

**Output:** Multi-page TIFF with embedded metadata

---

## Pre-generated Test Files

| File | Description |
|---|---|
| `spiral_cone_3dgauss.tif` | Spiral with 3D Gaussian intensity gradient |
| `spiral_cone_binary.tif` | Binary spiral (voxels are 0 or 1) |
| `spiral_cone_noiseaddedtwicemasked.tif` | Spiral with added noise, masked twice |

These TIFs can be passed directly to `glcm3D_attempt.m` to verify the GLCM pipeline produces expected texture feature maps before running on real data.

---

## When to Use

- After any change to the GLCM MATLAB code, run the pipeline on these synthetic files first
- Useful for comparing CPU vs. GPU outputs to check numerical consistency
- The binary version (`spiral_cone_binary.tif`) gives deterministic texture values useful for unit-checking specific features
