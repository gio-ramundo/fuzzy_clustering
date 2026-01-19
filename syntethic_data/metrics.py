from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple

@dataclass(frozen=True)
class FuzzyDiagnostics:
    partition_coefficient: float  # higher -> crisper
    partition_entropy: float      # lower -> crisper
    avg_max_membership: float     # higher -> crisper
    frac_ambiguous: float         # fraction with max membership < threshold


def fuzzy_diagnostics(U: np.ndarray, ambiguous_threshold: float = 0.6) -> FuzzyDiagnostics:
    U = np.asarray(U, dtype=float)
    n, c = U.shape
    pc = float(np.sum(U * U) / n)

    eps = 1e-12
    pe = float(-np.sum(U * np.log(U + eps)) / n)

    maxu = U.max(axis=1)
    avg_max = float(maxu.mean())
    frac_amb = float(np.mean(maxu < ambiguous_threshold))

    return FuzzyDiagnostics(
        partition_coefficient=pc,
        partition_entropy=pe,
        avg_max_membership=avg_max,
        frac_ambiguous=frac_amb,
    )

def _silhouette(df : pd.DataFrame, features_col : List[str], hard_label_col : str, m : float, mu_prefix : str = "mu_") -> Tuple[float, float]:
    X = df[features_col]
    score_hard = silhouette_score(X, df[hard_label_col])
    sample_silhouette_values = silhouette_samples(X, df[hard_label_col])
    mu_cols = [c for c in df.columns if c.startswith(mu_prefix)]
    U = df[mu_cols].values
    sorted_U = np.sort(U, axis=1)
    u_p = sorted_U[:, -1] # Max
    u_q = sorted_U[:, -2] # Second max
    weights = u_p - u_q
    score_fuz = np.sum(weights * sample_silhouette_values) / np.sum(weights) # Weighted mean
    return score_hard, score_fuz

def _xie_beni_index(df : pd.DataFrame, features_col : List[str],  hard_label_col : str, m : float, mu_prefix : str = "mu_") -> Tuple[float, float]:
    # Data array
    X = np.array(df[features_col])
    # Hard clustering
    labels = np.asarray(sorted(df[hard_label_col].unique()))
    centroids_hard = []
    for label in labels:
        c = X[df[hard_label_col] == label].mean(axis=0)
        centroids_hard.append(c)
    centroids_hard = np.array(centroids_hard)
    # Numerator
    sum_sq_dist = 0
    for i, center in enumerate(centroids_hard):
        cluster_points = X[df[hard_label_col] == labels[i]]
        sum_sq_dist += np.sum(np.linalg.norm(cluster_points - center, axis=1)**2)
    # Denominator
    if len(centroids_hard) > 1:
        centroid_distances = pdist(centroids_hard, metric='sqeuclidean')
        min_dist_centers = np.min(centroid_distances)
    else:
        min_dist_centers = 1e-20
    hard_xb_index = sum_sq_dist / (X.shape[0] * min_dist_centers)
    # Fuzzy
    mu_cols = [c for c in df.columns if c.startswith(mu_prefix)]
    U = np.array(df[mu_cols])
    Um = U ** m
    # Centroids
    num = Um.T @ X
    den = Um.sum(axis=0).reshape(-1, 1)
    fuz_centroids = num / den
    # Variation data-fuz_centroids
    total_variation = 0
    for i in range(fuz_centroids.shape[0]):
        diff = X - fuz_centroids[i]
        dist_sq = np.sum(diff**2, axis=1)
        total_variation += np.sum((U[:, i]**m) * dist_sq)
    # Min fuz_centroids distance
    if fuz_centroids.shape[0] > 1:
        centroid_distances = pdist(fuz_centroids, metric='sqeuclidean')
        min_centroid_dist = np.maximum(np.min(centroid_distances), 1e-20)
    else:
        min_centroid_dist = 1e-20
    xb_index = total_variation / (X.shape[0] * min_centroid_dist)
    return hard_xb_index, xb_index