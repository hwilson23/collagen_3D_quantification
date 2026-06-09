"""
Diagnostic: check whether features in the merged dataframe correlate with
concentration (1/2/3 mg/ml) and imaging modality (FLU vs SHG).

Run from the project root:
    uv run python src/collagen_dataframe/diagnostics.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

DATA_PATH = Path("data/finalcollapsed_dataframe_byslice.csv")
OUT_DIR = Path("data")

# Columns that are metadata, not features
META_COLS = {"image_name", "slice", "concentration_corrp", "type_autoc", "roi_senth"}

CONC_ORDER = {"1mgml": 1, "2mgml": 2, "3mgml": 3}


def load_and_label(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["concentration"] = df["concentration_corrp"].map(CONC_ORDER)
    df["modality"] = (df["type_autoc"] == "shg").astype(int)  # 0=FLU, 1=SHG
    return df


def spearman_vs_target(df: pd.DataFrame, target: str, label: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in META_COLS and c not in ("concentration", "modality") and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    y = df[target].values
    for col in feature_cols:
        x = df[col].dropna()
        common = df[target].loc[x.index]
        if len(x) < 10:
            continue
        r, p = spearmanr(x, common)
        rows.append({"feature": col, "rho": r, "p_value": p, "abs_rho": abs(r)})
    result = pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)
    result["target"] = label
    return result


def summarize(df_corr: pd.DataFrame, target: str, n: int = 10):
    sig = df_corr[df_corr["p_value"] < 0.05]
    print(f"\n=== {target} ===")
    print(f"  Features tested : {len(df_corr)}")
    print(f"  Significant (p<0.05): {len(sig)}  ({100*len(sig)/len(df_corr):.0f}%)")
    print(f"  Median |rho| across all: {df_corr['abs_rho'].median():.3f}")
    print(f"  Top {n} by |Spearman rho|:")
    top = df_corr.head(n)[["feature", "rho", "p_value"]]
    for _, row in top.iterrows():
        sig_marker = "*" if row["p_value"] < 0.05 else " "
        print(f"    {sig_marker} {row['feature']:<55s}  rho={row['rho']:+.3f}  p={row['p_value']:.2e}")


def plot_top_features(df: pd.DataFrame, corr_conc: pd.DataFrame, corr_mod: pd.DataFrame, n: int = 6):
    top_conc = corr_conc.head(n)["feature"].tolist()
    top_mod = corr_mod.head(n)["feature"].tolist()
    all_top = list(dict.fromkeys(top_conc + top_mod))  # deduplicated, order preserved

    n_cols = 3
    n_rows = int(np.ceil(len(all_top) / n_cols))
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    palette = {"flu": "#4878CF", "shg": "#D65F5F"}

    for idx, feat in enumerate(all_top):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        for modality, grp in df.groupby("type_autoc"):
            ax.scatter(grp["concentration"], grp[feat], alpha=0.35, s=12,
                       color=palette.get(modality, "grey"), label=modality.upper())
        ax.set_xlabel("Concentration (mg/ml)")
        ax.set_ylabel(feat[:30])
        ax.set_title(feat[:40], fontsize=8)
        ax.set_xticks([1, 2, 3])
        if idx == 0:
            ax.legend(fontsize=7, title="Modality")

    fig.suptitle("Top correlated features vs concentration", y=1.01, fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "diagnostic_top_features.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {out}")


def main():
    print(f"Loading {DATA_PATH}")
    df = load_and_label(DATA_PATH)
    print(f"  Shape: {df.shape}  |  rows per concentration: {df.groupby('concentration_corrp').size().to_dict()}")

    corr_conc = spearman_vs_target(df, "concentration", "concentration")
    corr_mod = spearman_vs_target(df, "modality", "FLU_vs_SHG")

    summarize(corr_conc, "Concentration (1/2/3 mg/ml)")
    summarize(corr_mod, "Modality (FLU=0 vs SHG=1)")

    plot_top_features(df, corr_conc, corr_mod)

    # Save full correlation tables for reference
    out_csv = OUT_DIR / "diagnostic_correlations.csv"
    pd.concat([corr_conc, corr_mod]).to_csv(out_csv, index=False)
    print(f"Full correlation table -> {out_csv}")


if __name__ == "__main__":
    main()
