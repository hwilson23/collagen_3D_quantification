# Machine Learning: Technical Methods Summary

This document summarizes the data science and machine learning pipeline for high-dimensional MRI datasets, based on established multi-parametric imaging workflows.

## 1. Data Preprocessing Pipeline
*   **Normalization**: Applied **z-score normalization** to all features to ensure a standard normal distribution (mean=0, variance=1), preventing features with large scales from dominating the model.
*   **Dimensionality Reduction Target**: Reducing ~100 initial features to a parsimonious subset of highly informative descriptors.
*   **Label Engineering**: For classification tasks, continuous variables were discretized into binary classes (High/Low) using a **median-split** approach.

## 2. Feature Categories
Features are categorized by their mathematical derivation from the region of interest (ROI):

### I. First-Order Statistics (18 Features)
Describe the distribution of voxel intensities within the ROI:
*   Energy, Total Energy, Entropy.
*   Mean, Median, Maximum, Minimum, Range.
*   Standard Deviation, Variance.
*   Skewness (asymmetry), Kurtosis (peakedness).
*   Uniformity, Root Mean Square (RMS).

### II. Shape-Based Descriptors (9 Features)
Describe the 2D/3D geometric properties of the ROI:
*   Mesh Volume, Surface Area, Surface-to-Volume Ratio.
*   Sphericity, Compactness.
*   Maximum 2D/3D Diameter.
*   Major/Minor Axis Length, Elongation, Flatness.

### III. Texture Matrices (70 Features)
These describe spatial relationships between voxels of specific intensities.

#### A. GLCM (Gray Level Co-occurrence Matrix) - 24 Features
Describes how often pairs of voxels with specific values and in a specified spatial relationship occur:
*   **Autocorrelation**: Measure of the magnitude of fineness and coarseness of texture.
*   **Contrast / Dissimilarity**: Measure of local variations.
*   **Correlation**: Linear dependency of gray levels between voxel pairs.
*   **Cluster Prominence / Shade**: Measures of skewness and lack of symmetry in the matrix.
*   **Energy / Homogeneity**: Measures of local uniformity.
*   **Maximum Probability**: The most frequent gray-level occurrence.
*   **Entropy**: Measure of randomness in the neighborhood.
*   **Inverse Difference (ID) / Inverse Difference Moment (IDM)**.

#### B. GLRLM (Gray Level Run Length Matrix) - 16 Features
Quantifies gray level runs, which are defined as the number of consecutive voxels that have the same gray level value:
*   **Short Run Emphasis (SRE) / Long Run Emphasis (LRE)**.
*   **Gray Level Non-Uniformity (GLN)**.
*   **Run Length Non-Uniformity (RLN)**.
*   **Run Percentage (RP)**.
*   **Low Gray Level Run Emphasis (LGRE) / High Gray Level Run Emphasis (HGRE)**.
*   **Short Run Low Gray Level Emphasis (SRLGE) / Short Run High Gray Level Emphasis (SRHGE)**.
*   **Long Run Low Gray Level Emphasis (LRLGE) / Long Run High Gray Level Emphasis (LRHGE)**.

#### C. GLSZM (Gray Level Size Zone Matrix) - 16 Features
Quantifies gray level zones (groups of connected voxels with the same gray level):
*   **Small Area Emphasis (SAE) / Large Area Emphasis (LAE)**.
*   **Gray Level Non-Uniformity (GLN) / Zone Size Non-Uniformity (ZSN)**.
*   **Zone Percentage (ZP)**.
*   **Low Gray Level Zone Emphasis (LGLZE) / High Gray Level Zone Emphasis (HGLZE)**.
*   **Small Area Low Gray Level Emphasis (SALGLE) / Small Area High Gray Level Emphasis (SAHGLE)**.
*   **Large Area Low Gray Level Emphasis (LALGLE) / Large Area High Gray Level Emphasis (LAHGLE)**.

#### D. GLDM (Gray Level Dependence Matrix) - 14 Features
Quantifies gray level dependencies (number of connected voxels within distance $\delta$ that are dependent on the center voxel):
*   Small Dependence Emphasis (SDE), Large Dependence Emphasis (LDE).
*   Gray Level Non-Uniformity (GLN), Dependence Non-Uniformity (DN).
*   Dependence Percentage (DP).
*   Low/High Gray Level Dependence Emphasis.

---

## 3. Feature Selection (FS) Techniques
Critical for preventing overfitting in high-dimensional spaces:
*   **Correlation Analysis**: Pruning redundant features with high inter-correlation.
*   **Variance Thresholding**: Removing features with low variance (constant features).
*   **SelectKBest**: Univariate selection based on Mutual Information scores.
*   **Recursive Feature Elimination (RFE)**: Iteratively removing features with the lowest coefficients in a Logistic Regression model.
*   **LASSO (L1-Regularization)**: Shrinking non-essential feature coefficients to zero.
*   **Tree-Based Importance**: Utilizing Gini impurity or Mean Decrease Accuracy (MDA) from Random Forest estimators.

## 4. Machine Learning Algorithms
*   **Random Forest (RF)**: Ensemble bagging method; typically best for high-dimensional radiomics.
*   **Support Vector Machine (SVM)**: Effective for non-linear boundaries via the kernel trick.
*   **K-Nearest Neighbors (KNN)**: Distance-based classifier; highly sensitive to feature scaling.
*   **Logistic Regression (LR)**: Linear baseline; often combined with L1/L2 penalties.
*   **Decision Tree (DT)**: Hierarchical splitting; used for high interpretability.
*   **Naive Bayes (NB)**: Probabilistic classifier assuming conditional independence.

## 5. Validation Framework
*   **Cross-Validation**: 5-Fold Stratified Cross-Validation on the training set.
*   **Independent Testing**: 20% hold-out set reserved for final evaluation.
*   **Performance Metrics**: 
    *   **ROC AUC**: For discriminative power.
    *   **F1-Score**: To balance Precision and Recall.
    *   **Accuracy / Balanced Accuracy**: Overall performance measurement.
