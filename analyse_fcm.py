from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from analysis_tools import (
    load_fcm_out,
    fuzzy_validity,
    infer_feature_columns,
    cluster_feature_means,
    cluster_feature_fuzzy_means,
    cluster_feature_zscores,
    cluster_feature_fuzzy_zscores,
    top_drivers_per_cluster,
    list_ambiguous_points,
    center_distances,
    crisp_separation_stats
)

def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)

def plot_membership_histograms(df: pd.DataFrame, mu_cols: Sequence[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in mu_cols:
        plt.figure()
        plt.hist(df[col].to_numpy(dtype=float), bins=30)
        plt.title(f"Membership distribution: {col}")
        plt.xlabel("membership")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png", dpi=160)
        plt.close()

    # max membership histogram
    U = df.loc[:, mu_cols].to_numpy(dtype=float)
    maxu = U.max(axis=1)
    plt.figure()
    plt.hist(maxu, bins=30)
    plt.title("Max membership distribution")
    plt.xlabel("max membership")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_max_membership.png", dpi=160)
    plt.close()

def plot_cluster_feature_profiles(means: pd.DataFrame, out_path: Path, top_features: Optional[List[str]] = None) -> None:
    # bar-style profile per cluster; if many features, optionally restrict
    if top_features is not None:
        means = means.loc[:, top_features]

    plt.figure(figsize=(max(8, 0.6 * means.shape[1]), 4 + 0.6 * means.shape[0]))
    for cl in means.index:
        plt.plot(range(means.shape[1]), means.loc[cl].to_numpy(dtype=float), marker="o", label=f"cluster {cl}")
    plt.xticks(range(means.shape[1]), means.columns, rotation=45, ha="right")
    plt.title("Cluster feature means (scaled space)")
    plt.ylabel("mean value")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_pca_scatter(df: pd.DataFrame, feature_cols: Sequence[str], hard_cluster_col : str, mu_cols: Sequence[str], out_path: Path) -> None:
    # Work on a copy of numeric feature matrix
    Xdf = df.loc[:, feature_cols].copy()

    # Drop columns that are entirely NaN or have zero variance (PCA hates them)
    Xdf = Xdf.dropna(axis=1, how="all")
    nunique = Xdf.nunique(dropna=True)
    Xdf = Xdf.loc[:, nunique > 1]

    # Impute remaining NaNs with column medians
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    # Final safety: remove any row with non-finite values
    X = Xdf.to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1)

    X = X[mask]
    U = df.loc[mask, mu_cols].to_numpy(dtype=float)
    labels = df.loc[mask, hard_cluster_col].to_numpy(dtype=int)
    maxu = U.max(axis=1)

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    plt.figure()
    sizes = 10 + 80 * (maxu - maxu.min()) / max(1e-12, (maxu.max() - maxu.min()))
    plt.scatter(Z[:, 0], Z[:, 1], s=sizes, c=labels, alpha=0.8)
    plt.title("PCA scatter (color=hard cluster, size=max membership)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_tsne_scatter(df: pd.DataFrame, feature_cols : Sequence[str], hard_cluster_col : str, mu_cols: Sequence[str], out_dir: Path) -> None:
    X = df.loc[:, feature_cols]
    U = df.loc[:, mu_cols].to_numpy(dtype=float)
    labels = df.loc[:, hard_cluster_col].to_numpy(dtype=int)
    maxu = U.max(axis=1)
    sizes = 10 + 80 * (maxu - maxu.min()) / max(1e-12, (maxu.max() - maxu.min()))
    for i in range(20, 41, 10):
        tsne = TSNE(n_components = 2, perplexity = i, random_state = 42)
        tsne_result = tsne.fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            size=sizes,
            hue=labels,
            palette='viridis',
            alpha=0.7,
            legend='brief'
        )
        plt.title(f"t-SNE projection (Perplexity: {i})")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir, f"T-sne_perp_{i}.png")
        plt.savefig(out_path, dpi=160)
        plt.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./fcm_out", help="Path to fcm_out directory")
    ap.add_argument("--amb_threshold", type=float, default=0.6, help="Ambiguity threshold for max membership")
    ap.add_argument("--amb_top_n", type=int, default=50, help="How many ambiguous points to export")
    ap.add_argument("--drivers_top_k", type=int, default=10, help="Top |z| features per cluster")
    ap.add_argument("--id_cols", type=str, default="", help="Comma-separated id columns to keep in ambiguous list")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    analysis_dir = out_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    tables_dir = analysis_dir / "tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_fcm_out(str(out_dir))
    df_raw = loaded.assignments_raw
    df_norm = loaded.assignments_norm
    mu_cols = loaded.mu_cols
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    # ---- Quantitative fuzzy validity
    val = fuzzy_validity(df_norm, mu_cols, ambiguous_threshold=args.amb_threshold)
    validity_df = pd.DataFrame([{
        "partition_coefficient": val.partition_coefficient,
        "partition_entropy": val.partition_entropy,
        "avg_max_membership": val.avg_max_membership,
        "frac_ambiguous": val.frac_ambiguous,
        "effective_clusters": val.effective_clusters,
        "hard_cluster_sizes": str(val.hard_cluster_sizes),
    }])
    save_df(validity_df, tables_dir / "fuzzy_validity.csv")

    # Separation stats (on memberships)
    sep = crisp_separation_stats(df_norm, mu_cols)
    save_df(sep, tables_dir / "membership_separation_describe.csv")

    # ---- Qualitative: cluster feature profiles
    feature_cols_raw = infer_feature_columns(df_raw, mu_cols, id_cols)
    means = cluster_feature_means(df_raw, feature_cols_raw)
    save_df(means, tables_dir / "cluster_feature_means_raw.csv")
    feature_cols_norm = infer_feature_columns(df_norm, mu_cols, id_cols)
    means = cluster_feature_means(df_norm, feature_cols_norm)
    save_df(means, tables_dir / "cluster_feature_means_norm.csv")

    zmeans = cluster_feature_zscores(df_norm, feature_cols_norm)
    save_df(zmeans, tables_dir / "cluster_feature_zmeans.csv")

    drivers = top_drivers_per_cluster(zmeans, top_k=args.drivers_top_k)
    # save as one csv per cluster
    for cl, ddf in drivers.items():
        (tables_dir / "drivers"/ "hard_clustering").mkdir(parents=True, exist_ok=True)
        ddf.to_csv(tables_dir / "drivers" / "hard_clustering" / f"cluster_{cl}_top_drivers.csv", index=False)

    fuz_means = cluster_feature_fuzzy_means(df_raw, feature_cols_raw)
    save_df(fuz_means, tables_dir / "cluster_feature_fuzzy_means_raw.csv")
    fuz_means = cluster_feature_fuzzy_means(df_norm, feature_cols_norm)
    save_df(fuz_means, tables_dir / "cluster_feature_fuzzy_means_norm.csv")

    fuz_zmeans = cluster_feature_fuzzy_zscores(df_norm, feature_cols_norm)
    save_df(fuz_zmeans, tables_dir / "cluster_feature_fuzzy_zmeans.csv")

    drivers = top_drivers_per_cluster(fuz_zmeans, top_k=args.drivers_top_k)
    # save as one csv per cluster
    for cl, ddf in drivers.items():
        (tables_dir / "drivers"/ "fuzzy_clustering").mkdir(parents=True, exist_ok=True)
        ddf.to_csv(tables_dir / "drivers" / "fuzzy_clustering" / f"cluster_{cl}_top_drivers.csv", index=False)

    # ---- Ambiguous points list (good for qualitative inspection)
    amb = list_ambiguous_points(
        df_norm, mu_cols,
        top_n=args.amb_top_n,
        threshold=args.amb_threshold,
        id_cols=id_cols,
    )
    amb.to_csv(tables_dir / "ambiguous_points.csv", index=False)

    # ---- Centers distances (if available)
    if loaded.centers is not None:
        cd = center_distances(loaded.centers)
        save_df(cd, tables_dir / "center_distances.csv")

    # ---- Plots
    plot_membership_histograms(df_norm, mu_cols, plots_dir)
    plot_cluster_feature_profiles(means, plots_dir / "cluster_feature_means.png")
    plot_pca_scatter(df_norm, feature_cols_norm, 'cluster_hard', mu_cols, plots_dir / "pca_scatter.png")
    plot_tsne_scatter(df_norm, feature_cols_norm, 'cluster_hard', mu_cols, plots_dir)

    print("Wrote analysis to:", analysis_dir)


if __name__ == "__main__":
    main()
