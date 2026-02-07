from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys
sys.path.append(str(Path()))
from preprocess import _islands_filter, preprocess_dataframe

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=Path(Path(__file__).resolve().parent,"classes_pca_exp_var.log"),
    filemode="w"
)

# Hyperparameters import
from run_all_fs import ID_COLS, EC_LABELS, CLUSTERING_FEATURES, AREA_VALUES, POP_VALUES, N_CLUSTERS, FUZ_RANGE

def main() -> None:
    # Input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_path", type=str, default="", help="Input dataframes")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder")
    args = ap.parse_args()
    df_path = Path(args.df_path)
    out_dir = Path(args.out_dir)
    df = pd.read_csv(df_path)
    df = _islands_filter(df, POP_VALUES, AREA_VALUES)
    rng = np.random.default_rng(420)
    # Consumption classes subdivision
    for ec_cls in N_CLUSTERS:
        df_cls = df[df[EC_LABELS] == ec_cls][ID_COLS + CLUSTERING_FEATURES].copy()
        prep = preprocess_dataframe(
            df_cls,
            drop_non_numeric=True,
            id_columns= ID_COLS,
            missing="median",
            scale="robust",
        )
        n_cluster = N_CLUSTERS [ec_cls]
        # Variance explained by PCA
        df_expo = prep.df_kept_norm
        pca = PCA()
        pca.fit(prep.X)
        explained_variance = 0
        # Explained variance analysis
        vals = []
        for i in range(9):
            explained_variance += pca.explained_variance_ratio_[i]
            vals.append(f"{explained_variance:.4f}")
        logging.info(f"Variance explained by PCA components for class {ec_cls}: %s"," ".join(vals))
        # Fuzzy clustering
        for m in FUZ_RANGE:
            best_value = 1e20
            for i in range(20):
                _, Uc, _, _, jm, _, _ = fuzz.cluster.cmeans(prep.X.T, n_cluster, m, error=1e-5, maxiter=100000, seed = rng.integers(10,1000))
                if jm[-1] < best_value:
                    best_value = jm[-1]
                    U = Uc.copy()
            for k in range(n_cluster):
                df_expo[f"mu_{round(m,1)}_{k}"] = U[k, :]
            labels_fuz = np.argmax(U, axis=0)
            df_expo[f"cluster_hard_{round(m,1)}"] = labels_fuz
            # PCA 2 dimensions projection
            plt.figure(figsize=(10, 10))
            maxu = U.max(axis=0)
            sizes = 10 + 80 * (maxu - maxu.min()) / max(1e-12, (maxu.max() - maxu.min()))
            pca = PCA(n_components=2, random_state=0)
            Z = pca.fit_transform(prep.X)
            scatter = plt.scatter(
                Z[:, 0], 
                Z[:, 1],
                s = sizes,
                c = df_expo[f'cluster_hard_{round(m,1)}'],
                cmap='tab10',
                alpha=0.7
            )
            clusters = np.unique(df_expo[f'cluster_hard_{round(m,1)}'])
            colors = scatter.cmap(scatter.norm(clusters))
            handles = [
                plt.Line2D(
                    [], [], marker='o', linestyle='',
                    color=colors[i], label=f'Cluster {clusters[i]}'
                )
                for i in range(len(clusters))
            ]
            plt.legend(handles=handles, title="Clusters")
            out_plot_dir = out_dir / f'plot_{ec_cls}'
            out_plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_plot_dir/f'PCA_m_{round(m,1)}.png')
            plt.close()
            
            # t-sne two dimensions projections
            tsne = TSNE(n_components = 2, perplexity = 30, random_state = 42)
            tsne_result = tsne.fit_transform(prep.X)
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                x=tsne_result[:, 0],
                y=tsne_result[:, 1],
                s=sizes,
                c=df_expo[f'cluster_hard_{round(m,1)}'],
                cmap='tab10',
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5
            )
            plt.legend(handles=handles, title="Clusters")
            plt.title(f"t-SNE projection ec_class {ec_cls} (m: {round(m,1)})")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            out_path = Path(out_plot_dir, f"T-sne_m_{round(m,1)}.png")
            plt.savefig(out_path)
            plt.close()
        # Final dataframe exportation
        df_expo.to_csv(out_dir / f'df_class_{ec_cls}.csv', index=False)
        
if __name__ == "__main__":
    main()

#Variance explained by PCA components for class XS:
#0.5605106631737463 0.8079497378436648 0.857197797531665 0.8927082299369055 0.9269914838741593 0.9525470127623451 0.9720469198711954 0.9899547529253162 1.0
#Variance explained by PCA components for class S:
#0.545018424406708 0.8196623089752785 0.8703743504684338 0.9147858264478466 0.9400612873721828 0.9622002388334177 0.9802092492933079 0.9921084696965599 1.0  
#Variance explained by PCA components for class M:
#0.4770060229046438 0.8102130852897094 0.914631097516834 0.9506021424911935 0.9687832954768263 0.9794961128657814 0.9897944153429209 0.9959024983790603 0.9999999999999999  
#Variance explained by PCA components for class L:
#0.46002682476165363 0.6476299977297757 0.790402755342065 0.8545697156183155 0.9104039153892968 0.9470164163735165 0.973265299143745 0.992089283217231 0.9999999999999997  