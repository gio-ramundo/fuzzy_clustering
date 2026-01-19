from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocess import preprocess_dataframe, _islands_filter
from fcm import FuzzyCMeans
from metrics import fuzzy_diagnostics, _silhouette, _xie_beni_index

# -------------------- USER SETTINGS --------------------
CSV_PATH = "data/df_complete.csv"
OUT_ROOT = "hyperparameter_definition"

# columns lists
ID_COLS = ["ALL_Uniq"]
EXCLUDE_COLUMNS = ['hdd', 'cdd', 'total_precipitation', 'Shape_Leng', 'total_precipitation', 'elevation_max']
AREA_VALUES = [3, 100000]
POP_VALUES = [200, 1000000]
# Resource-only feature set for Step 2
RESOURCE_FEATURES = [
    "solar_power",
    "solar_seas_ind",
    "wind_power",
    "wind_std",
    "offshore_wind_potential",
    "hydro_potential",
    "geothermal_potential",
    "evi",
    "res_area",
]

# ------------------------------------------------------

def plot_pca_scatter(df: pd.DataFrame, feature_cols : Sequence[str], out_dir : Path, ec_label) -> None:
    # Work on a copy of numeric feature matrix
    Xdf = df.loc[:, feature_cols]

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
    
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.8)
    plt.title(f"PCA scatter class {ec_label}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    out_path = Path(out_dir, f"PCA_{ec_label}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_tsne_scatter(df: pd.DataFrame, feature_cols : Sequence[str], out_dir : Path, ec_label) -> None:
    # Work on a copy of numeric feature matrix
    Xdf = df.loc[:, feature_cols]

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
    
    for i in range(20, 41, 10):
        tsne = TSNE(n_components = 2, perplexity = i, random_state = 42)
        tsne_result = tsne.fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            s=50
        )
        plt.title(f"t-SNE projection ec_class {ec_label} (Perplexity: {i})")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        out_path = Path(out_dir, f"T-sne_{ec_label}_perp_{i}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160)
        plt.close()

# Hyperparam iterations
def _hyper_iteration(df : pd.Dataframe, feature_cols : Sequence[str], out_dir : Path, ec_label : str) -> None:
    df_work = df.copy()
    X = df_work[feature_cols]
    n_values = range(2, 7)
    m_values = range(15, 31, 3)
    m_values = [val/10 for val in m_values]
    # Dataframes to store statistics
    pc_df = pd.DataFrame(index=n_values, columns=m_values)
    pe_df = pd.DataFrame(index=n_values, columns=m_values)
    xie_df = pd.DataFrame(index=n_values, columns=m_values)
    sil_df = pd.DataFrame(index=n_values, columns=m_values)
    avg_max_mem_df = pd.DataFrame(index=n_values, columns=m_values)
    frac_amb_df = pd.DataFrame(index=n_values, columns=m_values)
    n_iter_df = pd.DataFrame(index=n_values, columns=m_values)
    out_dir1 = Path(out_dir, 'summary_json')
    out_dir1.mkdir(parents=True, exist_ok=True)
    for n in n_values:
        for m in m_values:
            model = FuzzyCMeans(
                n_clusters = n,
                m=m,
                max_iter=600,
                tol=1e-5,
                seed=429,
            )
            res = model.fit(X)
            labels = model.hard_labels(res.memberships)
            diag = fuzzy_diagnostics(res.memberships, ambiguous_threshold=0.6)
            df_work['hard_cluster'] = labels
            for i in range(n):
                df_work[f'mu_{i}'] = res.memberships[:, i]
            sil = _silhouette(df_work, RESOURCE_FEATURES, 'hard_cluster', m, 'mu_')
            xie = _xie_beni_index(df_work, RESOURCE_FEATURES, 'hard_cluster', m, 'mu_')
            summary = {
                "n_rows_total": int(X.shape[0]),
                "n_clusters": int(n),
                "m": float(m),
                "n_iter": int(res.n_iter),
                "converged": bool(res.converged),
                "final_objective": float(res.objective_history[-1]),
                "fuzzy_diagnostics": asdict(diag),
                "cluster_sizes_hard": {str(k): int(np.sum(labels == k)) for k in range(n)},
                "xie_beni_index" : xie[1],
                "fuzzy silhouette" : sil[1]
            }
            (out_dir1 / f"n={n}_m={round(m,1)}.json").write_text(json.dumps(summary, indent=2))
            pc_df.loc[n,m] = diag.partition_coefficient
            pe_df.loc[n,m] = diag.partition_entropy
            sil_df.loc[n,m] = sil[1]
            xie_df.loc[n,m] = xie[1]
            avg_max_mem_df.loc[n,m] = diag.avg_max_membership
            frac_amb_df.loc[n,m] = diag.frac_ambiguous
            n_iter_df.loc[n,m] = int(res.n_iter)
    print(f"All n,m combination runs for hyperparameters evaluation completed for class {ec_label}.")
    # Exportation of statistics graph
    dict_stat = {
        "partition_coefficient" : pc_df.astype(float),
        "partition_entropy" : pe_df.astype(float),
        "xie_beni_index" : xie_df.astype(float),
        "silhouette_score" : sil_df.astype(float),
        "average_max_membership" : avg_max_mem_df.astype(float),
        "fraction_ambiguos" : frac_amb_df.astype(float),
        "n_iter" : n_iter_df.astype(float)
    }
    for stat in dict_stat:
        for n in n_values:
            out_dir1 = Path(out_dir, stat, "n_clust_fix")
            out_dir1.mkdir(parents=True, exist_ok=True)
            plt.plot(m_values, dict_stat[stat].loc[n])
            plt.xlabel("m")
            plt.ylabel(stat)
            plt.title(f"{stat} for n_clust = {n}")
            plt.tight_layout()
            plt.savefig(out_dir1 / f"{n}.png", dpi=160)
            plt.close()
        for m in m_values:
            out_dir1 = Path(out_dir, stat, "m_fix")
            out_dir1.mkdir(parents=True, exist_ok=True)
            plt.plot(n_values, dict_stat[stat][m])
            plt.xlabel("n")
            plt.ylabel(stat)
            plt.title(f"{stat} for m = {round(m,1)}")
            plt.tight_layout()
            plt.savefig(out_dir1 / f"{round(m,1)}.png", dpi=160)
            plt.close()
        sns.heatmap(dict_stat[stat], annot=True, cmap="viridis")
        out_dir1 = Path(out_dir, stat)
        plt.xlabel("m")
        plt.ylabel("n")
        plt.title(f"{stat} heatmap")
        plt.tight_layout()
        plt.savefig(out_dir1 / f"heatmap.png", dpi=160)
        plt.close()
    
#XIE GRANDEZZA INSTABILE CON CLUSTER PICCOLI, CON N E M CHE SALGONO I CENTRI SI SOVRAPPONGONO, O RIRUN CON SEED DIVERSI O INGNORA PER IL MOMENTO
def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = _islands_filter(df, POP_VALUES, AREA_VALUES)
    ec_classes = df['ec_labels'].unique()
    for ec_class in ec_classes:
        out_dir_visualization = Path(OUT_ROOT, ec_class, "data_visualization")
        out_dir_elbow = Path(OUT_ROOT, ec_class)
        df_work = df[df['ec_labels']==ec_class].copy()
        prep = preprocess_dataframe(
            df_work,
            drop_non_numeric=True,
            id_columns = ID_COLS,
            exclude_columns= EXCLUDE_COLUMNS,
            missing="median",
            scale="robust",
        )
        plot_pca_scatter(prep.df_kept_norm, RESOURCE_FEATURES, out_dir_visualization, ec_class)
        plot_tsne_scatter(prep.df_kept_norm, RESOURCE_FEATURES, out_dir_visualization, ec_class)
        _hyper_iteration(prep.df_kept_norm, RESOURCE_FEATURES, out_dir_elbow, ec_class)

if __name__ == "__main__":
    main()