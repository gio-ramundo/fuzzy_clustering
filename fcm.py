from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FCMResult:
    centers: np.ndarray          # (c, d)
    memberships: np.ndarray      # (n, c)
    objective_history: np.ndarray
    n_iter: int
    converged: bool


class FuzzyCMeans:
    """
    Simple Fuzzy C-Means (FCM) clustering with Euclidean distance.

    memberships u_ij in [0,1], sum_j u_ij = 1 for each i.

    References: classic Bezdek FCM update rules.
    """
    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        max_iter: int = 600,
        tol: float = 1e-5,
        seed: int = 42,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if m <= 1.0:
            raise ValueError("m must be > 1")
        self.c = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)

    def _init_memberships(self, n: int) -> np.ndarray:
        U = self.rng.random((n, self.c))
        U /= U.sum(axis=1, keepdims=True)
        return U

    @staticmethod
    def _dist2(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # squared euclidean distances: (n, c)
        # X: (n, d), centers: (c, d)
        # returns D2[i,j] = ||X_i - center_j||^2
        diff = X[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _update_centers(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        Um = U ** self.m  # (n,c)
        denom = Um.sum(axis=0, keepdims=True).T  # (c,1)
        # avoid division by zero
        denom = np.maximum(denom, 1e-12)
        centers = (Um.T @ X) / denom
        return centers

    def _update_memberships(self, D2: np.ndarray) -> np.ndarray:
        # Handle zero distances (point equals a center)
        # If D2[i,j] == 0, set membership 1 for that cluster and 0 for others.
        n, c = D2.shape
        U_new = np.zeros((n, c), dtype=float)

        zero_mask = D2 <= 1e-18
        has_zero = zero_mask.any(axis=1)
        if np.any(has_zero):
            # For rows with one or more zeros, assign equally among zero-distance centers
            rows = np.where(has_zero)[0]
            for i in rows:
                z = zero_mask[i]
                U_new[i, z] = 1.0 / z.sum()

        # For other rows, apply FCM formula
        rows = np.where(~has_zero)[0]
        if rows.size > 0:
            D2nz = D2[rows]  # (r,c)
            # u_ij = 1 / sum_k ( (d_ij / d_ik)^(1/(m-1)) )
            # Using squared distances works the same as distances with consistent exponent.
            power = 1.0 / (self.m - 1.0)
            ratio = (D2nz[:, :, None] / D2nz[:, None, :]) ** power  # (r,c,c)
            denom = ratio.sum(axis=2)  # (r,c)
            U_new[rows] = 1.0 / np.maximum(denom, 1e-12)

        # Ensure row-stochastic
        U_new /= U_new.sum(axis=1, keepdims=True)
        return U_new

    def _objective(self, D2: np.ndarray, U: np.ndarray) -> float:
        return float(np.sum((U ** self.m) * D2))

    def fit(self, X: np.ndarray) -> FCMResult:
        X = np.asarray(X, dtype=float)
        n, d = X.shape

        U = self._init_memberships(n)
        centers = self._update_centers(X, U)

        obj_hist = []
        converged = False

        for it in range(1, self.max_iter + 1):
            D2 = self._dist2(X, centers)
            obj = self._objective(D2, U)
            obj_hist.append(obj)

            U_new = self._update_memberships(D2)
            centers_new = self._update_centers(X, U_new)

            # convergence checks
            delta = np.linalg.norm(centers_new - centers)
            if delta < self.tol:
                converged = True
                U, centers = U_new, centers_new
                break

            U, centers = U_new, centers_new

        # final objective
        D2 = self._dist2(X, centers)
        obj_hist.append(self._objective(D2, U))

        return FCMResult(
            centers=centers,
            memberships=U,
            objective_history=np.array(obj_hist, dtype=float),
            n_iter=it,
            converged=converged,
        )

    @staticmethod
    def hard_labels(U: np.ndarray) -> np.ndarray:
        return np.argmax(U, axis=1)
