from pathlib import Path
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=Path(Path(__file__).resolve().parent,"dataset_generation.log"),
    filemode="w"
)

# Dataset generatin function
def generate_cluster_dataset(
    n_rows: int, # Elements number
    n_cols: int, # Normal features number
    n_zero_cols: int, # Features with many 0 number
    n_clusters: int, # Number of clusters
    n_overlap: int, # Overlap points number
    n_outliers: int, # Outlier points number
    separation: float = 3.0, # Clusters centers separation parameter
    cluster_std: float = 1.0, # Clusters standard deviation parameter
    outlier_scale: float = 3.0, # Outliers separation parameter
    zero_share : float = 0.7, # Share of zero in zero columns
    random_state: int = 420
) -> None:
    np.random.seed(random_state)
    cluster_weights = np.random.dirichlet(alpha=np.ones(n_clusters)*1.6) # Random clusters dimensions
    cluster_sizes = (cluster_weights * n_rows).astype(int)
    cluster_sizes[0] += n_rows - cluster_sizes.sum()
    X = []
    y = []
    # Centers directions
    directions = np.random.randn(n_clusters, n_cols+n_zero_cols)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Values in zero_cols
    if n_zero_cols > 0:
        for col_idx in range(n_cols, n_cols+n_zero_cols):
            directions[:, col_idx] = np.abs(directions[:, col_idx])
            mask = np.random.random(directions.shape[0]) < zero_share
            directions[mask, col_idx] = 0.0
    # Centers distances from centers, lognormal distribution (many near, some far)
    radii = np.random.gamma(shape=6, scale=separation * cluster_std/6, size=(n_clusters, 1))
    centers = directions * radii
    # Points generation from centers
    for i in range(n_clusters):
        pts = np.random.normal(
            loc=centers[i],
            scale=cluster_std,
            size=(cluster_sizes[i], n_cols + n_zero_cols)
        )
        for col_idx in range(n_cols, n_cols+n_zero_cols):
            if centers[i, col_idx] == 0:
                pts[:, col_idx] = np.abs(pts[:, col_idx])
                mask = np.random.random(cluster_sizes[i]) < zero_share
                pts[mask, col_idx] = 0.0
            else:
                pts[:, col_idx] = np.maximum(0, pts[:, col_idx])
        X.append(pts)
        y.extend([int(i)] * cluster_sizes[i])
    # Overlap points, between closest clusters
    if n_overlap > 0:
        min_dist = np.inf
        closest_pair = None
        for i, j in combinations(range(n_clusters), 2):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist, closest_pair = d, (i, j)
        i, j = closest_pair
        overlap_center = (centers[i] + centers[j]) / 2
        overlap_pts = np.random.normal(
            loc=overlap_center,
            scale=cluster_std * 1.2,
            size=(n_overlap, n_cols + n_zero_cols)
        )
        for col_idx in range(n_cols, n_cols+n_zero_cols):
            if overlap_center[col_idx] == 0:
                overlap_pts[:, col_idx] = np.abs(overlap_pts[:, col_idx])
                mask = np.random.random(n_overlap) < zero_share
                overlap_pts[mask, col_idx] = 0.0
            else:
                overlap_pts[:, col_idx] = np.maximum(0, overlap_pts[:, col_idx])
        X.append(overlap_pts)
        y.extend([-1] * n_overlap)
    X = np.vstack(X)
    # Outliers generation
    if n_outliers > 0:
        # Few variables outliers
        indices = np.random.choice(X.shape[0], n_outliers, replace=False)
        perturbed_outliers = []
        for i in indices:
            # Number of variable to outlie
            if n_cols+n_zero_cols == 2:
                k = 1
            elif n_cols+n_zero_cols == 3:
                k = np.random.choice([1,2], size=1, p=[0.9, 0.1])[0]
            else:
                k = np.random.choice([1,2,3], size=1, p=[0.63, 0.3, 0.07])[0]
            feats = np.random.choice(n_cols+n_zero_cols, k, replace=False)
            noise = np.random.normal(0, cluster_std*0.4, size=n_cols+n_zero_cols)
            for feat in feats:
                noise[feat] = np.random.normal(
                    0, outlier_scale * cluster_std
                )
            for ind in range(n_cols, n_cols+n_zero_cols):
                noise[ind] = np.abs(noise[ind])
            out = X[i] + noise
            perturbed_outliers.append(out)
        X = np.vstack([X, perturbed_outliers]) 
        y.extend([-2] * n_outliers)
        # General outliers
        outlier_range = separation * cluster_std * outlier_scale
        outliers = np.random.uniform(
            low=-outlier_range,
            high=outlier_range,
            size=(n_outliers, n_cols + n_zero_cols)
        )
        if n_zero_cols > 0:
            for col_idx in range(n_cols, n_cols + n_zero_cols):
                outliers[:, col_idx] = np.abs(outliers[:, col_idx])
                mask = np.random.random(n_outliers) < zero_share
                outliers[mask, col_idx] = 0.0
        X=np.vstack([X,outliers])
        y.extend([-3] * n_outliers)
    return X, y, closest_pair

# Different generations
n_rows_dict = {'XS' : 1234,
               'S' : 460,
               'M' : 231,
               'L' : 126,
               'TEST' : 2500}
n_clusters_dict = {'XS' : 3,
                   'S': 3,
                   'M': 4,
                   'L': 2,
                   'TEST' : 5}
separation_dict = {2 : {'LOW' : 5, 'HIGH' : 9},
                   4 : {'LOW' : 4, 'HIGH' : 8.25},
                   6 : {'LOW' : 3, 'HIGH' : 6.5}
                }
rng = np.random.default_rng(6899)
# Diffrent dimensions
for dimensions in [[2,0], [4,1], [6,3]]:
    logging.info(f'Total dimensions = {dimensions[0]+dimensions[1]}')
    # Different dataset sizes
    for dim, n_rows in n_rows_dict.items():
        out_dir = Path(Path(__file__).resolve().parent, 'dataframes', f'{dimensions[0]+dimensions[1]} dimensions', 'raw', dim)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_clusters = n_clusters_dict[dim]
        # Different centers separation
        for descr,separation in separation_dict[dimensions[0]].items():
            logging.info(f'Class {dim}, separation {descr}')
            if descr == 'LOW':
                if dim == 'L':
                    separation *= 1.4
                if dim == 'M':
                    separation *= 1.2
                if dim == 'TEST':
                    separation *= 0.8
            for k in range(10):
                logging.info(f'Run {k}')
                array, labels, couple = generate_cluster_dataset(n_rows=n_rows,
                                                                 n_cols=dimensions[0],
                                                                 n_zero_cols=dimensions[1],
                                                                 n_clusters=n_clusters,
                                                                 n_overlap=int(n_rows/(n_clusters)),
                                                                 n_outliers=0,
                                                                 separation=separation,
                                                                 cluster_std=1,
                                                                 random_state=rng.integers(100,2000))
                df = pd.DataFrame(array)
                pca = PCA()
                pca.fit(df.values)
                explained_variance = 0
                # Explained variance analysis
                vals = []
                for i in range(len(df.columns)):
                    explained_variance += pca.explained_variance_ratio_[i]
                    vals.append(f"{explained_variance:.4f}")
                logging.info("Variance explained by PCA components: %s"," ".join(vals))
                df['clusters'] = labels
                # Exportation
                output_path = Path(out_dir, f'df_{dim}_sep_{descr}_{k}.csv')
                df.to_csv(output_path, index = False)
                # Plot if dimensions = 2
                if dimensions[0] + dimensions[1] == 2:
                    plt.figure(figsize=(10, 10))
                    scatter = plt.scatter(
                        df[0],
                        df[1],
                        c=df['clusters'],
                        cmap='tab10',
                        alpha=0.7
                        )
                    clusters = np.unique(df['clusters'])
                    colors = scatter.cmap(scatter.norm(clusters))
                    handles = [
                        plt.Line2D(
                            [], [], marker='o', linestyle='',
                            color=colors[i], label=f'Cluster {clusters[i]}'
                        )
                        for i in range(len(clusters))
                        ]
                    plt.legend(handles=handles, title="Clusters")
                    out_plot_dir = Path(Path(__file__).resolve().parent, f'dataframes/{dimensions[0]+dimensions[1]} dimensions/plot', dim)
                    out_plot_dir.mkdir(parents=True, exist_ok=True)
                    output_path = Path(out_plot_dir, f'cluster_{n_clusters_dict[dim]}_sep_{descr}_{k}.png')
                    plt.savefig(output_path)
                    plt.close()