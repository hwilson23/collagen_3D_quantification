import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def run_diagnostic_analysis():
    # 1. Data Loading
    file_path = 'collagen_3D_multimetric-main/data/final_dataframe_byslice_FLU_n2.csv'
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # 2. Target Encoding
    # The prompt specified concentration_corrp, but in this specific file it might be concentration_denth
    target_col = 'concentration_corrp'
    if target_col not in df.columns:
        if 'concentration_denth' in df.columns:
            target_col = 'concentration_denth'
            print(f"Warning: '{target_col}' used as target instead of 'concentration_corrp'.")
        else:
            print("Error: Target column not found.")
            return

    # Map target to numerical values
    target_mapping = {'1mgml': 1, '2mgml': 2, '3mgml': 3}
    df['target_numeric'] = df[target_col].map(target_mapping)
    df = df.copy() # De-fragment
    
    # Identify non-numeric columns to drop (except target and image_name)
    # We keep image_name for grouping info later
    cols_to_drop = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if 'image_name' in cols_to_drop:
        cols_to_drop.remove('image_name')
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)
    
    # Feature set (dropping non-numeric and targets)
    features_df = df.drop(columns=cols_to_drop + [target_col, 'target_numeric'])
    # Ensure only numeric features are left
    features_df = features_df.select_dtypes(include=[np.number])

    # 3. Variance Analysis
    variances = features_df.var()
    low_variance_features = variances[variances < 0.01].index.tolist()
    
    print("\n--- Variance Analysis ---")
    print(f"Features with zero or near-zero variance (< 0.01): {len(low_variance_features)}")
    for feat in low_variance_features:
        print(f"  - {feat}: {variances[feat]:.6f}")

    # 4. Multicollinearity Mapping
    corr_matrix = features_df.corr()
    
    # Filter for correlations > |0.85|
    high_corr = corr_matrix.where(np.abs(corr_matrix) > 0.85)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(high_corr, annot=False, cmap='coolwarm', center=0)
    plt.title("High Correlation Heatmap (|r| > 0.85)")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("\nSaved correlation_heatmap.png")

    # 5. Separability Visualization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df.fillna(0)) # Handle NaNs if any

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df = df.copy() # De-fragment
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='pca-one', y='pca-two', hue=target_col, data=df, palette='viridis', alpha=0.7)
    plt.title('PCA Analysis')
    plt.savefig('pca_analysis.png')
    print("\nSaved pca_analysis.png")
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

    # UMAP or t-SNE
    if HAS_UMAP:
        print("Performing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(scaled_features)
        method_name = "UMAP"
    else:
        print("UMAP not available, performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        embedding = tsne.fit_transform(scaled_features)
        method_name = "t-SNE"

    df['embedding-one'] = embedding[:, 0]
    df['embedding-two'] = embedding[:, 1]
    df = df.copy() # De-fragment
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='embedding-one', y='embedding-two', hue=target_col, data=df, palette='viridis', alpha=0.7)
    plt.title(f'{method_name} Analysis')
    plt.savefig('umap_analysis.png') # File name as requested
    print(f"Saved umap_analysis.png (using {method_name})")

    # 6. Feature Grouping Awareness
    unique_images = df['image_name'].nunique()
    
    # 7. Output Summary Report
    print("\n==========================================")
    print("      DIAGNOSTIC EDA SUMMARY REPORT       ")
    print("==========================================")
    print(f"Dataset Shape: {df.shape}")
    print(f"Unique Images: {unique_images}")
    print(f"Target Variable: {target_col}")
    print(f"Number of Features Analyzed: {features_df.shape[1]}")
    print(f"Low Variance Features: {len(low_variance_features)}")
    print(f"PCA Explained Variance (2 components): {sum(pca.explained_variance_ratio_):.4f}")
    if HAS_UMAP:
        print("UMAP visualization generated.")
    else:
        print("t-SNE visualization generated (UMAP fallback).")
    print("All plots saved to current directory.")
    print("==========================================\n")

if __name__ == "__main__":
    run_diagnostic_analysis()
