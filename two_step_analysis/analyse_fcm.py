from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from run_all_fs import FUZ_RANGE, ID_COLS, CLUSTERING_FEATURES

import sys
sys.path.append(str(Path()))
from analysis_tools import (
    fuzzy_validity,
    cluster_feature_means,
    cluster_feature_fuzzy_means,
    cluster_feature_zscores,
    cluster_feature_fuzzy_zscores,
    top_drivers_per_cluster,
    list_ambiguous_points,
    crisp_separation_stats
)

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

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_path", type=str, default="", help="Input dataframes")
    ap.add_argument("--amb_threshold", type=float, default=0.6, help="Ambiguity threshold for max membership")
    ap.add_argument("--drivers_top_k", type=int, default=4, help="Top |z| features per cluster")
    args = ap.parse_args()
    df_path = Path(args.df_path)
    df = pd.read_csv(df_path)
    for m in FUZ_RANGE:
        out_dir = df_path.parent / 'analysis' / df_path.name.replace('df_class_', '').replace('.csv', '') / str(round(m,1))
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        tables_dir = out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        mu_cols = [c for c in df.columns if c.startswith(f"mu_{round(m,1)}_")]
        # ---- Quantitative fuzzy validity
        val = fuzzy_validity(df, mu_cols, f'cluster_hard_{round(m,1)}', ambiguous_threshold=args.amb_threshold)
        validity_df = pd.DataFrame([{
            "partition_coefficient": val.partition_coefficient,
            "partition_entropy": val.partition_entropy,
            "avg_max_membership": val.avg_max_membership,
            "frac_ambiguous": val.frac_ambiguous,
            "effective_clusters": val.effective_clusters,
            "hard_cluster_sizes": str(val.hard_cluster_sizes),
        }])
        validity_df.to_csv(tables_dir / "fuzzy_validity.csv", index=True)
        # Separation stats (on memberships)
        sep = crisp_separation_stats(df, mu_cols)
        sep.to_csv(tables_dir / "membership_separation_describe.csv", index = True)
        # ---- Qualitative: cluster feature profiles
        means = cluster_feature_means(df, CLUSTERING_FEATURES, f'cluster_hard_{round(m,1)}')
        means.to_csv(tables_dir / "cluster_feature_means_norm.csv", index = True)
        zmeans = cluster_feature_zscores(df, CLUSTERING_FEATURES, f'cluster_hard_{round(m,1)}')
        zmeans.to_csv(tables_dir / "cluster_feature_zmeans.csv", index = True)
        
        drivers = top_drivers_per_cluster(zmeans, top_k=args.drivers_top_k)
        # save as one csv per cluster
        for cl, ddf in drivers.items():
            (tables_dir / "drivers"/ "hard_clustering").mkdir(parents=True, exist_ok=True)
            ddf.to_csv(tables_dir / "drivers" / "hard_clustering" / f"cluster_{cl}_top_drivers.csv", index=False)
        fuz_means = cluster_feature_fuzzy_means(df, mu_cols, CLUSTERING_FEATURES, f"mu_{round(m,1)}_")
        fuz_means.to_csv(tables_dir / "cluster_feature_fuzzy_means_norm.csv", index=True)
        fuz_zmeans = cluster_feature_fuzzy_zscores(df, mu_cols, CLUSTERING_FEATURES, f"mu_{round(m,1)}_")
        fuz_zmeans.to_csv(tables_dir / "cluster_feature_fuzzy_zmeans.csv", index=True)
        drivers = top_drivers_per_cluster(fuz_zmeans, top_k=args.drivers_top_k)
        # save as one csv per cluster
        for cl, ddf in drivers.items():
            (tables_dir / "drivers"/ "fuzzy_clustering").mkdir(parents=True, exist_ok=True)
            ddf.to_csv(tables_dir / "drivers" / "fuzzy_clustering" / f"cluster_{cl}_top_drivers.csv", index=False)
        # ---- Ambiguous points list (good for qualitative inspection)
        amb = list_ambiguous_points(
            df, mu_cols, f'cluster_hard_{round(m,1)}',
            threshold=args.amb_threshold,
            id_cols=ID_COLS,
        )
        amb.to_csv(tables_dir / "ambiguous_points.csv", index=False)
        # ---- Plots
        plot_membership_histograms(df, mu_cols, plots_dir)
        plot_cluster_feature_profiles(means, plots_dir / "cluster_feature_means.png")
    
if __name__ == "__main__":
    main()
