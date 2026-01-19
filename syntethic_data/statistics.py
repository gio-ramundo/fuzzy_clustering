from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from scipy.special import entr
from scipy.stats import pearsonr, spearmanr, mode

import numpy as np
import pandas as pd

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
    entropy_matrix = entr(consensus) + entr(1 - consensus)
    point_entropy = np.nansum(entropy_matrix, axis=1) / n_samples
    results.append(point_entropy)
    for threshold in [0.5, 0.7, 0.9]:
        mask = (consensus >= threshold) & (~np.eye(n_samples, dtype=bool))
        C_filtered = np.where(mask, consensus, 0)
        sum_cij = np.sum(C_filtered, axis=1)
        count_cij = np.sum(mask, axis=1)
        stability_avg = np.divide(sum_cij, count_cij, 
                              out=np.zeros_like(sum_cij), 
                              where=count_cij != 0)
        results.append(stability_avg)
    stability_matrix = np.abs(consensus - 0.5) * 2
    stability_scores = np.mean(stability_matrix, axis=1)
    results.append(stability_scores)
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

# Correlations matrix
def correlation_matrixes(df : pd.DataFrame, fuzzy_cols : List[str], stab_cols : List[str], output_dir : Path, output_name : str) -> None:
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
    out_dir = Path(output_dir, 'correlations_matrixes')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir, output_name+".xlsx")
    with pd.ExcelWriter(out_path) as writer:
        pearson_mat.to_excel(writer, sheet_name="Pearson")
        spearman_mat.to_excel(writer, sheet_name="Spearman")

# Fuzziness interval stability
def fuzziness_by_intervals(df : pd.DataFrame, fuzzy_cols : List[str], stab_cols : List[str], output_dir : Path, output_name : str, bin_width : float=0.1) -> None:
    bins = np.arange(0, 1 + bin_width, bin_width)
    labels = [f"{round(bins[i],2)}–{round(bins[i+1],2)}"
              for i in range(len(bins) - 1)]
    out_dir = Path(output_dir, 'fuzziness_inetrvals')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir, output_name+".xlsx")
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

# Fuzziness quantiles stability
def fuzziness_by_quantiles(df : pd.DataFrame, fuzzy_cols : List[str], stab_cols : List[str], output_dir : Path, output_name : str, n_quantiles : int=10) -> None:    
    out_dir = Path(output_dir, 'fuzziness_intervals')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir, output_name+".xlsx")
    with pd.ExcelWriter(out_path) as writer:
        for fcol in fuzzy_cols:
            temp = df.copy()
            temp["fuzz_quantile"] = pd.qcut(
                temp[fcol],
                q=n_quantiles,
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
    ap.add_argument("--inp_dir", type=str, default="", help="Input dataframes")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder")
    ap.add_argument("--seed", type=int, default="", help="Random seed")
    args = ap.parse_args()
    inp_dir = Path(args.inp_dir)
    df = pd.read_csv(inp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cluster = len(mu_cols)
    feature_cols = [c for c in df.columns if c not in mu_cols+['cluster_hard', 'clusters', 'kmeans']]
    # Stability columns definition
    rep_stability_columns = ['st_r_const_ass', 'st_r_ent', 'st_r_thr_0.5', 'st_r_thr_0.7', 'st_r_thr_0.9', 'st_r_0.5_val']
    results_list = repetition_stability(df, n_cluster, feature_cols, 200, random_state=int(args.seed))
    for i, name in enumerate(rep_stability_columns):
        df[name] = results_list[i]
    mu_cols = [c for c in df.columns if c.startswith("mu_")]
    perturbation_columns = []
    for noise in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1]:
        columns = [f'st_gp_{noise}_const_ass', f'st_gp_{noise}_ent', f'st_gp_{noise}_thr_0.5', f'st_gp_{noise}_thr_0.7', f'st_gp_{noise}_thr_0.9', f'st_gp_{noise}_0.5_val']
        perturbation_columns += columns
        results_list = perturbation_stability(df, n_cluster, feature_cols, noise, 200, random_state=int(args.seed))
        for i, name in enumerate(columns):
            df[name] = results_list[i]
    stability_columns = rep_stability_columns + perturbation_columns
    # Fuzzy columns definition
    U = df[mu_cols].values
    # Punti normali tende a 0, punti overlap ln(0.5), outliers ln(n_clust)
    df["fuz_entropy"] = np.sum(U * np.log(U + 1e-12), axis=1)
    U_sorted = np.sort(U, axis=1)[:, ::-1]
    # Punti normali tende a 1, punti overlap 0, outliers 0
    df["fuz_gap"] = U_sorted[:, 0] - U_sorted[:, 1]
    # Punti normali tende a 0, punti overlap 0.5, outliers 1-1/c
    df["fuz_index"] = 1.0 - np.sum(U ** 2, axis=1)
    # Punti normali tende a 1, punti overlap 2, outliers c
    df["fuz_eff_n_cl"] = 1.0 / np.sum(U ** 2, axis=1)
    fuzziness_columns = [c for c in df.columns if c.startswith("fuz_")]
    # Summary statistics, all dataset
    out_name = inp_dir.name.replace('df_', '').replace('.csv','')
    correlation_matrixes(df, fuzziness_columns, stability_columns, out_dir, out_name)
    fuzziness_by_intervals(df, fuzziness_columns, stability_columns, out_dir, out_name)
    fuzziness_by_quantiles(df, fuzziness_columns, stability_columns, out_dir, out_name)
    # Normal data
    df1 = df[~df['clusters'].isin([-1, -2, -3])]
    correlation_matrixes(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_norm")
    fuzziness_by_intervals(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_norm")
    fuzziness_by_quantiles(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_norm")
    # Type 1 outliers
    df1 = df[df['clusters'].isin([-2])]
    correlation_matrixes(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_out1")
    fuzziness_by_intervals(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_out1")
    fuzziness_by_quantiles(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_out1")
    # Type 2 outliers
    df1 = df[df['clusters'].isin([-3])]
    correlation_matrixes(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_out2")
    fuzziness_by_intervals(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_out2")
    fuzziness_by_quantiles(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_out2")
    # General outliers
    df1 = df[df['clusters'].isin([-2, -3])]
    correlation_matrixes(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_out")
    fuzziness_by_intervals(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_out")
    fuzziness_by_quantiles(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_out")
    # Overlap points
    df1 = df[df['clusters'].isin([-1])]
    correlation_matrixes(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_ovlap")
    fuzziness_by_intervals(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_ovlap")
    fuzziness_by_quantiles(df1, fuzziness_columns, stability_columns, out_dir, out_name+"_ovlap")

if __name__ == "__main__":
    main()

#code to test stability functions values excluding o erlap points or outliers or both
    #print('totale')
    #    for col in columns:
    #        print(np.mean(df[col]))
    #    print(' ')
    #    print('no_overlap')
    #    df1 = df[~df['clusters'].isin([-1])]
    #    results_list = global_perturbation_stability(df1, n_cluster, feature_cols, noise, 50, random_state=int(args.seed))
    #    for i, name in enumerate(columns):
    #        df1[name] = results_list[i]
    #    for col in columns:
    #        print(np.mean(df1[col]))
    #    print(' ')
    #    print('no_outliers')
    #    df1 = df[~df['clusters'].isin([-2,-3])]
    #    results_list = global_perturbation_stability(df1, max(df['clusters'])+1, feature_cols, noise, 50, random_state=int(args.seed))
    #    for i, name in enumerate(columns):
    #        df1[name] = results_list[i]
    #    for col in columns:
    #        print(np.mean(df1[col]))
    #    print(' ')
    #    print('normali')
    #    df1 = df[~df['clusters'].isin([-1,-2,-3])]
    #    results_list = global_perturbation_stability(df1, max(df['clusters'])+1, feature_cols, noise, 50, random_state=int(args.seed))
    #    for i, name in enumerate(columns):
    #        df1[name] = results_list[i]
    #    for col in columns:
    #        print(np.mean(df1[col]))
    #    print(' ')
    #