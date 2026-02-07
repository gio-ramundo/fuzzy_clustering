from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

import numpy as np
import pandas as pd

from run_all_sd import NOISE_VALUES, STABILITY_INDICATORS, FUZ_RANGE

# Function to align label of two different clustering
def align_labels(labels_ref : np.array, labels : np.array) -> np.array:
    if len(labels_ref) != len(labels):
        raise ValueError("Clustering runs refers to different datasets.")
    n_cluster_ref = labels_ref.max() + 1
    n_cluster = labels.max() + 1
    if n_cluster_ref != n_cluster:
        raise ValueError("Clustering runs have different cluster numbers.")
    cost_matrix = confusion_matrix(labels_ref, labels)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    label_map = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([label_map[l] for l in labels])

# Input list of labels of different runs, output stability statistics
def stability_statistics(list_labels, n_samples, n_runs):
    results = []
    pointwise_quota = np.zeros(n_samples)
    for i in range(n_samples):
        most_freq_cluster, _ = mode(list_labels[:, i])
        count = np.sum(list_labels[:, i] == most_freq_cluster)
        pointwise_quota[i] = count / n_runs
    results.append(pointwise_quota)
    # Consensus matrix
    consensus = np.zeros((n_samples, n_samples))
    for k in range(n_runs):
        labels = list_labels[k, :].reshape(-1, 1)
        consensus += (labels == labels.T).astype(float)
    consensus /= n_runs
    #for threshold in [0.5, 0.7, 0.9]:
    mask = (consensus >= 0.5) & (~np.eye(n_samples, dtype=bool))
    C_filtered = np.where(mask, consensus, 0)
    sum_cij = np.sum(C_filtered, axis=1)
    count_cij = np.sum(mask, axis=1)
    stability_avg = np.divide(sum_cij, count_cij, 
                          out=np.zeros_like(sum_cij), 
                          where=count_cij != 0)
    results.append(stability_avg)
    return results

# Stability evaluation repeating clustering process with different initialization
def repetition_stability(df : pd.DataFrame, n : int, feature_cols : List[str], n_runs = 200, random_state : int = 420) -> np.array:
    rng_func = np.random.default_rng(random_state)
    X = df[feature_cols].values
    n_samples = X.shape[0]
    all_labels = np.zeros((n_runs, n_samples), dtype=int)
    for run in range(n_runs):
        seed = rng_func.integers(0, 1_000_000)
        km = KMeans(
            n_clusters=n,
            n_init=30,
            random_state=seed
        )
        labels = km.fit_predict(X)
        if run == 0:
            all_labels[run] = labels
        else:
            all_labels[run] = align_labels(all_labels[0], labels)
    return stability_statistics(all_labels, n_samples, n_runs)

# Stability evaluation with perturbation
def perturbation_stability(df : pd.DataFrame, n : int, feature_cols : List[str], noise_std : float, n_runs: int = 200, random_state : int = 420) -> np.array:
    rng_func = np.random.default_rng(random_state)
    X = df[feature_cols].values
    n_samples = X.shape[0]
    all_labels = np.zeros((n_runs, n_samples), dtype=int)
    for run in range(n_runs):
        X_perturbed = X + rng_func.normal(0, noise_std, size=X.shape)
        seed = rng_func.integers(0, 1_000_000)
        km = KMeans(
            n_clusters=n,
            n_init=30,
            random_state=seed
        )
        labels = km.fit_predict(X_perturbed)
        if run == 0:
            all_labels[run] = labels
        else:
            all_labels[run] = align_labels(all_labels[0], labels)
    return stability_statistics(all_labels, n_samples, n_runs)

def main() -> None:
    # Input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_path", type=str, default="", help="Input dataframes")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder")
    ap.add_argument("--seed", type=int, default="", help="Random seed")
    args = ap.parse_args()
    df_path = Path(args.df_path)
    df = pd.read_csv(df_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cluster = len([c for c in df.columns if c.startswith(f"mu_1.5")])
    feature_cols = [c for c in df.columns if (c not in ['clusters', 'kmeans']) 
                    and not (c.startswith(f"mu_"))
                    and not (c.startswith(f"cluster_hard"))]
    # Stability columns definition
    #methods = ["st_r_"]
    methods = []
    #rep_stability_columns = ["st_r_"+ind for ind in STABILITY_INDICATORS]
    #results_list = repetition_stability(df, n_cluster, feature_cols, 200, random_state=int(args.seed))
    #for i, name in enumerate(rep_stability_columns):
    #    df[name] = results_list[i]
    perturbation_columns = []
    for noise in NOISE_VALUES:
        methods.append(f"st_p_{noise}_")
        columns = [f'st_p_{noise}_'+ ind for ind in STABILITY_INDICATORS]
        perturbation_columns += columns
        results_list = perturbation_stability(df, n_cluster, feature_cols, noise, 200, random_state=int(args.seed))
        for i, name in enumerate(columns):
            df[name] = results_list[i]
    #stability_columns = rep_stability_columns + perturbation_columns
    stability_columns = perturbation_columns
    other_columns = [col for col in df.columns if col not in stability_columns]
    ordered_stability_columns = [met + ind for ind in STABILITY_INDICATORS for met in methods]
    df = df[other_columns + ordered_stability_columns]
    # Fuzziness columns definition
    for m in FUZ_RANGE:
        mu_cols = [c for c in df.columns if c.startswith(f"mu_{round(m,1)}")]
        U = df[mu_cols].values
        # Normal points tends to 0, overlap points to ln(0.5)/ln(n_clust), outliers to 1
        df[f"fuz_{round(m,1)}_entropy"] = -np.sum(U * np.log(U + 1e-12), axis=1)/(np.log(n_cluster))
        U_sorted = np.sort(U, axis=1)[:, ::-1]
        # Normal points tends to 1, overlap points to 0, outliers to 0
        df[f"fuz_{round(m,1)}_gap"] = U_sorted[:, 0] - U_sorted[:, 1]
        # Normal points tends to 0, overlap points to 0.5, outliers to 1-1/c
        #df[f"fuz_{round(m,1)}_index"] = 1.0 - np.sum(U ** 2, axis=1)
    df.to_csv(out_dir / df_path.name, index=False)
    
if __name__ == "__main__":
    main()