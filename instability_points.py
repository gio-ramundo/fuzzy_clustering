from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr, spearmanr

from analysis_tools import load_fcm_out

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

# Metodologia non valida
def repetition_stability(df : pd.DataFrame, n : int, cluster_hard_label : str, mu_cols : List[str], n_ref = 30, n_conf = 30) -> np.array:
    rng = np.random.RandomState(42)
    feature_cols = [col for col in df.columns if col not in mu_cols+[cluster_hard_label]]
    X = df[feature_cols].values
    n_samples = X.shape[0]
    similarity_sum = np.zeros(n_samples)
    total_comparisons = 0
    for _ in range(n_ref):
        seed_ref = rng.randint(0, 1_000_000)
        km_ref = KMeans(
            n_clusters=n,
            n_init=1,
            random_state=seed_ref
        )
        labels_ref = km_ref.fit_predict(X)
        for _ in range(n_conf):
            seed_conf = rng.randint(0, 1_000_000)
            km_conf = KMeans(
            n_clusters=n,
            n_init=1,
            random_state=seed_conf
            )
            labels_conf = km_conf.fit_predict(X)
            labels_aligned = align_labels(labels_ref, labels_conf)
            similarity_sum += (labels_aligned == labels_ref)
            total_comparisons += 1
    similarity = similarity_sum/total_comparisons
    return similarity

# Rivedi come viene aggiunto il rumore
def perturbation_stability(df : pd.DataFrame, n : int, cluster_hard_label : str, mu_cols : List[str], noise_std : float, n_runs: int = 1000) -> np.array:
    rng = np.random.RandomState(42)
    feature_cols = [col for col in df.columns if col not in mu_cols+[cluster_hard_label]]
    X = df[feature_cols].values
    kmeans_ref = KMeans(n_clusters=n, n_init=20, random_state=42)
    ref_labels = kmeans_ref.fit_predict(X)
    n_samples = X.shape[0]
    stability = np.zeros(n_samples)
    for _ in range(n_runs):
        X_perturbed = X + rng.normal(0, noise_std, size=X.shape)
        kmeans = KMeans(n_clusters=n, n_init=20, random_state=rng.randint(0, 1_000_000))
        comp_labels = kmeans.fit_predict(X_perturbed)
        labels_aligned = align_labels(ref_labels, comp_labels)
        # Aggiorna stabilità
        stability += (labels_aligned == ref_labels)
    stability /= n_runs
    return stability

# Metodologia non valida (studia meglio il bootstrap)
def bootstrap_stability(df : pd.DataFrame, n : int, cluster_hard_label : str, mu_cols : List[str], n_bootstrap: int = 1000):
    rng = np.random.RandomState(42)
    feature_cols = [col for col in df.columns if col not in mu_cols+[cluster_hard_label]]
    X = df[feature_cols].values
    kmeans_ref = KMeans(n_clusters=n, n_init=20, random_state=42)
    ref_labels = kmeans_ref.fit_predict(X)
    n_samples = X.shape[0]
    stability_counts = np.zeros(n_samples)
    presence_counts = np.zeros(n_samples)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, n_samples, replace=True)
        X_boot = X[idx]
        kmeans = KMeans(n_clusters=n, n_init=20)
        new_labels = kmeans.fit_predict(X_boot)
        new_labels_aligned = align_labels(ref_labels[idx], new_labels)
        for orig_idx, boot_label, ref_label in zip(idx, new_labels_aligned, ref_labels[idx]):
            presence_counts[orig_idx] += 1
            if boot_label == ref_label:
                stability_counts[orig_idx] += 1
    stability = stability_counts / presence_counts
    return stability

def correlation_matrixes(df : pd.DataFrame, fuzzy_cols : List[str], stab_cols : List[str], output_dir : Path) -> None:
    pearson_mat = pd.DataFrame(
    index=fuzzy_cols,
    columns=stab_cols,
    dtype=float
    )
    spearman_mat = pearson_mat.copy()
    for fuzz_col in fuzzy_cols:
        for stab_col in stab_cols:
            x = df[fuzz_col]
            y = df[stab_col]
            mask = x.notna() & y.notna()
            x_clean = x[mask]
            y_clean = y[mask]
            if len(x_clean) > 2  and x_clean.nunique() > 1 and y_clean.nunique() > 1:
                pearson_mat.loc[fuzz_col, stab_col] = pearsonr(x_clean, y_clean)[0]
                spearman_mat.loc[fuzz_col, stab_col] = spearmanr(x_clean, y_clean)[0]
            else:
                pearson_mat.loc[fuzz_col, stab_col] = np.nan
                spearman_mat.loc[fuzz_col, stab_col] = np.nan
    out_dir = Path(output_dir, 'fuzziness_stability_link')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir, "correlation_matrices.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        pearson_mat.to_excel(writer, sheet_name="Pearson")
        spearman_mat.to_excel(writer, sheet_name="Spearman")

def fuzziness_by_intervals(df : pd.DataFrame, fuzzy_cols : List[str], stab_cols : List[str], output_dir : Path, bin_width : float=0.1) -> None:
    bins = np.arange(0, 1 + bin_width, bin_width)
    labels = [f"{round(bins[i],2)}–{round(bins[i+1],2)}"
              for i in range(len(bins) - 1)]
    out_dir = Path(output_dir, 'fuzziness_stability_link')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir, "fuzziness_interval.xlsx")
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for fcol in fuzzy_cols:
            temp = df.copy()
            temp["fuzz_bin"] = pd.cut(
                temp[fcol],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            mean_stab = (
                temp
                .groupby("fuzz_bin")[stab_cols]
                .mean()
            )
            counts = (
                temp
                .groupby("fuzz_bin")
                .size()
                .rename("n_obs")
            )
            summary = mean_stab.join(counts)
            summary.to_excel(writer, sheet_name=fcol)

def fuzziness_by_quantiles(df : pd.DataFrame, fuzzy_cols : List[str], stab_cols : List[str], output_dir : Path, n_quantiles : int=10) -> None:
    out_dir = Path(output_dir, 'fuzziness_stability_link')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir, "fuzziness_quantiles.xlsx")
    quantile_labels = [f"{round(i/n_quantiles, 2)}–{round((i+1)/n_quantiles, 2)}"
        for i in range(n_quantiles)]
    with pd.ExcelWriter(out_path) as writer:
        for fcol in fuzzy_cols:
            temp = df.copy()
            temp["fuzz_quantile"] = pd.qcut(
                temp[fcol],
                q=n_quantiles,
                labels=quantile_labels,
                duplicates="drop"
            )
            mean_stab = (
                temp
                .groupby("fuzz_quantile")[stab_cols]
                .mean()
            )
            counts = (
                temp
                .groupby("fuzz_quantile")
                .size()
                .rename("n_obs")
            )
            summary = mean_stab.join(counts)
            summary.to_excel(writer, sheet_name=fcol)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./fcm_out", help="Path to fcm_out directory")
    ap.add_argument("--amb_top_n", type=int, default=50, help="How many ambiguous points to export")
    ap.add_argument("--id_cols", type=str, default="", help="Comma-separated id columns to keep in ambiguous list")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_fcm_out(str(out_dir))
    df_norm = loaded.assignments_norm
    mu_cols = [c for c in df_norm.columns if c.startswith("mu_")]
    n_cluster = len(mu_cols)
    # Stability columns definition
    df_norm['stab_rep'] = repetition_stability(df_norm, n_cluster, 'cluster_hard', mu_cols, 30, 30)
    for noise in [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]:
        df_norm[f'stab_pert_{noise}'] = perturbation_stability(df_norm, n_cluster, 'cluster_hard', mu_cols, noise)
    df_norm['stab_boot'] = bootstrap_stability(df_norm, n_cluster, 'cluster_hard', mu_cols)
    stability_columns = [c for c in df_norm.columns if c.startswith("stab")]
    # Fuzzy columns definition
    U = df_norm[mu_cols].values
    fuzzy_entropy = -np.sum(U * np.log(U + 1e-12), axis=1)
    fuzzy_entropy_norm = fuzzy_entropy / np.log(n_cluster)
    df_norm["fuz_entropy"] = fuzzy_entropy
    df_norm["fuz_entropy_norm"] = fuzzy_entropy_norm
    U_sorted = np.sort(U, axis=1)[:, ::-1]
    df_norm["fuz_gap"] = U_sorted[:, 0] - U_sorted[:, 1]
    fuz_index = 1.0 - np.sum(U ** 2, axis=1)
    df_norm["fuz_index"] = fuz_index
    effective_clusters = 1.0 / np.sum(U ** 2, axis=1)
    df_norm["fuz_eff_n_cl"] = effective_clusters
    fuzziness_columns = [c for c in df_norm.columns if c.startswith("fuz_")]
    # Matrixes with correlation indexes
    correlation_matrixes(df_norm, fuzziness_columns, stability_columns, out_dir)
    fuzziness_by_intervals(df_norm, fuzziness_columns, stability_columns, out_dir)
    fuzziness_by_quantiles(df_norm, fuzziness_columns, stability_columns, out_dir)

if __name__ == "__main__":
    main()
