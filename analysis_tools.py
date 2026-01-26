from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from itertools import combinations

import json
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class LoadedFCMOutputs:
    assignments_raw: pd.DataFrame  # original columns + mu_k + cluster_hard, raw data
    assignments_norm: pd.DataFrame  # original columns + mu_k + cluster_hard, normalized data
    centers: Optional[pd.DataFrame]
    summary: Optional[dict]
    mu_cols: List[str]


def load_fcm_out(out_dir: str) -> LoadedFCMOutputs:
    out = Path(out_dir)
    assign_path = out / "assignments_memberships_raw.csv"
    if not assign_path.exists():
        raise FileNotFoundError(f"Missing: {assign_path}")
    df_raw = pd.read_csv(assign_path)
    mu_cols = [c for c in df_raw.columns if c.startswith("mu_")]
    if not mu_cols:
        raise ValueError("No membership columns found (expected columns like mu_0, mu_1, ...).")
    if "cluster_hard" not in df_raw.columns:
        raise ValueError("Missing 'cluster_hard' column in assignments_memberships_raw.csv")

    out = Path(out_dir)
    assign_path = out / "assignments_memberships_norm.csv"
    if not assign_path.exists():
        raise FileNotFoundError(f"Missing: {assign_path}")
    df_norm = pd.read_csv(assign_path)
    mu_cols = [c for c in df_norm.columns if c.startswith("mu_")]
    if not mu_cols:
        raise ValueError("No membership columns found (expected columns like mu_0, mu_1, ...).")
    if "cluster_hard" not in df_norm.columns:
        raise ValueError("Missing 'cluster_hard' column in assignments_memberships_norm.csv")

    centers_path = out / "centers.csv"
    centers = pd.read_csv(centers_path) if centers_path.exists() else None

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    return LoadedFCMOutputs(assignments_raw=df_raw, assignments_norm=df_norm, centers=centers, summary=summary, mu_cols=mu_cols)


# -----------------------------
# Fuzzy validity + diagnostics
# -----------------------------

@dataclass(frozen=True)
class FuzzyValidity:
    partition_coefficient: float
    partition_entropy: float
    avg_max_membership: float
    frac_ambiguous: float
    effective_clusters: float  # exp(mean entropy) or 1/sum(u^2) analogue
    hard_cluster_sizes: Dict[int, int]


def fuzzy_validity(
    df: pd.DataFrame,
    mu_cols: Sequence[str],
    cluster_hard : str,
    ambiguous_threshold: float = 0.6
) -> FuzzyValidity:
    U = df.loc[:, mu_cols].to_numpy(dtype=float)
    n, c = U.shape
    eps = 1e-12

    pc = float(np.sum(U * U) / n)
    pe = float(-np.sum(U * np.log(U + eps)) / n)
    maxu = U.max(axis=1)
    avg_max = float(np.mean(maxu))
    frac_amb = float(np.mean(maxu < ambiguous_threshold))

    # "effective number of clusters" per point ~ exp(entropy), then average
    ent_i = -np.sum(U * np.log(U + eps), axis=1)
    eff = float(np.mean(np.exp(ent_i)))

    sizes = df[cluster_hard].value_counts().sort_index()
    sizes_dict = {int(k): int(v) for k, v in sizes.items()}

    return FuzzyValidity(
        partition_coefficient=pc,
        partition_entropy=pe,
        avg_max_membership=avg_max,
        frac_ambiguous=frac_amb,
        effective_clusters=eff,
        hard_cluster_sizes=sizes_dict,
    )

def list_ambiguous_points(
    df: pd.DataFrame,
    mu_cols: Sequence[str],
    cluster_hard,
    top_n: int = 30,
    threshold: float = 0.6,
    id_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    id_cols = list(id_cols) if id_cols else []
    U = df.loc[:, mu_cols].to_numpy(dtype=float)
    maxu = U.max(axis=1)
    hard = np.argmax(U, axis=1)
    second = np.partition(U, -2, axis=1)[:, -2]
    margin = maxu - second

    tmp = df.copy()
    tmp["_maxu"] = maxu
    tmp["_hard_from_mu"] = hard
    tmp["_second"] = second
    tmp["_margin"] = margin

    amb = tmp[tmp["_maxu"] < threshold].sort_values(["_maxu", "_margin"], ascending=[True, True]).head(top_n)
    keep_cols = [c for c in id_cols if c in amb.columns] + [cluster_hard] + list(mu_cols) + ["_maxu", "_second", "_margin"]
    return amb.loc[:, keep_cols]

# -----------------------------------------
# Cluster profiling / qualitative assessment
# -----------------------------------------

def cluster_feature_means(df: pd.DataFrame, feature_cols: Sequence[str], cluster_hard : str) -> pd.DataFrame:
    return df.groupby(cluster_hard)[feature_cols].mean().sort_index()

def cluster_feature_zscores(df: pd.DataFrame, feature_cols: Sequence[str], cluster_hard : str) -> pd.DataFrame:
    # z-score means relative to global mean/std for interpretability
    X = df[feature_cols].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    means = cluster_feature_means(df, feature_cols, cluster_hard).to_numpy(dtype=float)
    z = (means - mu) / sigma
    return pd.DataFrame(z, index=cluster_feature_means(df, feature_cols, cluster_hard).index, columns=feature_cols)

def cluster_feature_fuzzy_means(df: pd.DataFrame, mu_cols : Sequence[str], feature_cols: Sequence[str], mu_prefix : str = "mu_") -> pd.DataFrame:
    out = []
    for mu in mu_cols:
        w = df[mu]
        out.append((df[feature_cols].mul(w, axis=0).sum() / w.sum()))
    res = pd.DataFrame(out, index=[int(c.replace(mu_prefix, "")) for c in mu_cols])
    return res.sort_index()

def cluster_feature_fuzzy_zscores(df: pd.DataFrame, mu_cols : Sequence[str], feature_cols: Sequence[str], mu_prefix : str = "mu_") -> pd.DataFrame:
    # z-score means relative to global mean/std for interpretability
    X = df[feature_cols].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    means = cluster_feature_fuzzy_means(df, mu_cols, feature_cols, mu_prefix).to_numpy(dtype=float)
    z = (means - mu) / sigma
    return pd.DataFrame(z, index=cluster_feature_fuzzy_means(df, mu_cols, feature_cols, mu_prefix).index, columns=feature_cols)

def top_drivers_per_cluster(z_means: pd.DataFrame, top_k: int = 8) -> Dict[int, pd.DataFrame]:
    # returns, for each cluster, features with largest |z|
    out: Dict[int, pd.DataFrame] = {}
    for cl in z_means.index:
        s = z_means.loc[cl]
        top = s.reindex(s.abs().sort_values(ascending=False).head(top_k).index)
        out[int(cl)] = top.reset_index().rename(columns={"index": "feature", cl: "z"})
    return out


# -----------------------------
# Separation diagnostics
# -----------------------------

def center_distances(centers: pd.DataFrame) -> pd.DataFrame:
    C = centers.to_numpy(dtype=float)
    # pairwise euclidean
    d = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
    return pd.DataFrame(d, index=[f"c{i}" for i in range(C.shape[0])], columns=[f"c{j}" for j in range(C.shape[0])])


def crisp_separation_stats(df: pd.DataFrame, mu_cols: Sequence[str]) -> pd.DataFrame:
    # For each point: ratio of nearest vs second nearest membership (proxy for separation)
    U = df.loc[:, mu_cols].to_numpy(dtype=float)
    maxu = U.max(axis=1)
    second = np.partition(U, -2, axis=1)[:, -2]
    margin = maxu - second
    ratio = maxu / np.maximum(second, 1e-12)

    return pd.DataFrame({
        "max_membership": maxu,
        "second_membership": second,
        "margin": margin,
        "ratio": ratio,
    }).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T