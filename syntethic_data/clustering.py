from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from run_all_sd import FUZ_RANGE

def main() -> None:
    # Input values
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp_dir", type=str, default="", help="Input dataframes")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder")
    args = ap.parse_args()
    inp_dir = Path(args.inp_dir)
    out_dir = Path(args.out_dir)
    df = pd.read_csv(inp_dir)
    n_original_clusters = max(list(df['clusters']))+1
    features_columns = [col for col in df.columns if col != 'clusters']
    # n_cluster identification, identify number that maximes correspondence in normal point (no outliers, no overlap)
    mask = ~df['clusters'].isin([-1, -2, -3])
    y_true = df.loc[mask, 'clusters']
    best_k = None
    best_ari = -1
    X = df[features_columns].copy()
    for k in range(n_original_clusters, n_original_clusters + 4):
        kmeans = KMeans(
            n_clusters=k,
            n_init=30,
            random_state = 420
        )
        labels = kmeans.fit_predict(X)
        ari = adjusted_rand_score(y_true, labels[mask])
        if ari > best_ari:
            best_ari = ari
            best_k = k
            df['kmeans'] = labels
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Plot exportation, works only with two dimensons
    #plt.figure(figsize=(10, 10))
    #scatter = plt.scatter(
    #    df['0'],
    #    df['1'],
    #    c=df['kmeans'],
    #    cmap='tab10',
    #    alpha=0.7
    #)
    #clusters = np.unique(df['kmeans'])
    #colors = scatter.cmap(scatter.norm(clusters))
    #handles = [
    #    plt.Line2D(
    #        [], [], marker='o', linestyle='',
    #        color=colors[i], label=f'Cluster {clusters[i]}'
    #    )
    #    for i in range(len(clusters))
    #]
    #plt.legend(handles=handles, title="Clusters")
    #plt.savefig(out_plot_dir/inp_dir.name.replace('df_', '').replace('.csv','_km.png'))
    #plt.close()

    # Fuzzy C-Means
    for m in FUZ_RANGE:
        _, U, _, _, _, _, _ = fuzz.cluster.cmeans(X.values.T, best_k, m, error=1e-5, maxiter=100000, seed = 420)
        for k in range(best_k):
            df[f"mu_{round(m,1)}_{k}"] = U[k, :]
        labels_fuz = np.argmax(U, axis=0)
        df[f"cluster_hard_{round(m,1)}"] = labels_fuz

        #plt.figure(figsize=(10, 10))
        #maxu = U.max(axis=0)
        #sizes = 10 + 80 * (maxu - maxu.min()) / max(1e-12, (maxu.max() - maxu.min()))
        #scatter = plt.scatter(
        #    df['0'],
        #    df['1'],
        #    s = sizes,
        #    c = df[f'cluster_hard_{m}'],
        #    cmap='tab10',
        #    alpha=0.7
        #)
        #clusters = np.unique(df[f'cluster_hard_{m}'])
        #colors = scatter.cmap(scatter.norm(clusters))
        #handles = [
        #    plt.Line2D(
        #        [], [], marker='o', linestyle='',
        #        color=colors[i], label=f'Cluster {clusters[i]}'
        #    )
        #    for i in range(len(clusters))
        #]
        #plt.legend(handles=handles, title="Clusters")
        #plt.savefig(out_dir/inp_dir.name.replace('df_', '').replace('.csv','_fcm.png'))
    
    df.to_csv(out_dir / inp_dir.name, index=False)
        
if __name__ == "__main__":
    main()