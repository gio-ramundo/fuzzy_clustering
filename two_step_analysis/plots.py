from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import sys
sys.path.append(str(Path()))
from preprocess import _islands_filter, preprocess_dataframe

# Hyperparameters import
from run_all_fs import ID_COLS, EC_LABELS, CLUSTERING_FEATURES, AREA_VALUES, POP_VALUES

def main() -> None:
    # Input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_path", type=str, default="", help="Input dataframes")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder")
    args = ap.parse_args()
    df_path = Path(args.df_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(df_path)
    df = _islands_filter(df, POP_VALUES, AREA_VALUES)
    # Consumption classes subdivision
    for ec_cls in df[EC_LABELS].unique():
        df_cls = df[df[EC_LABELS] == ec_cls][ID_COLS + CLUSTERING_FEATURES].copy()
        prep = preprocess_dataframe(
            df_cls,
            drop_non_numeric=True,
            id_columns= ID_COLS,
            missing="median",
            scale="robust",
        )
        # PCA 2 dimensions projection
        pca = PCA()
        pca.fit(prep.X)
        plt.figure(figsize=(10, 10))
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(prep.X)
        plt.scatter(
            Z[:, 0], 
            Z[:, 1],
            alpha=0.7
        )
        plt.title(f"PCA projection ec_class {ec_cls}")
        plt.savefig(out_dir/f'PCA_{ec_cls}_class.png')
        plt.close()
        # t-SNE two dimensions projections
        tsne = TSNE(n_components = 2, perplexity = 30, random_state = 42)
        tsne_result = tsne.fit_transform(prep.X)
        plt.figure(figsize=(8, 6))
        plt.scatter(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            alpha=0.7
        )
        plt.title(f"t-SNE projection ec_class {ec_cls}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_dir/f't-SNE_{ec_cls}_class.png')
        plt.close()
        # Elbow method
        inertia = []
        silhouette_avg = []
        k_range = range(2,7)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(prep.X)
            inertia.append(kmeans.inertia_)
            silhouette_avg.append(silhouette_score(prep.X, cluster_labels))
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertia, marker='o', color='b', linestyle='--')
        plt.xlabel('Clusters number (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_avg, marker='o', color='r', linestyle='--')
        plt.xlabel('Clusters number (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir/f'kmeans_elbow_silhouette_{ec_cls}_class.png')
        plt.close()

        
if __name__ == "__main__":
    main()