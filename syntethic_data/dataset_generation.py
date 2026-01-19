from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

def generate_cluster_dataset(
    n_rows: int,
    n_cols: int,
    n_zero_cols: int,
    n_clusters: int,
    n_overlap: int,
    n_outliers: int,
    separation: float = 3.0,
    cluster_std: float = 1.0,
    outlier_scale: float = 1.3,
    zero_share : float = 0.7,
    random_state: int = 420
) -> None:
    np.random.seed(random_state)
    assert n_overlap + n_outliers * 2 < n_rows, "Overlap + outliers >= n_rows"
    cluster_weights = np.random.dirichlet(alpha=np.ones(n_clusters))
    cluster_sizes = (cluster_weights * n_rows).astype(int)
    cluster_sizes[0] += n_rows - cluster_sizes.sum()
    X = []
    y = []

    # Centri tutti con la stessa distanza dal centro
    #centers = np.random.randn(n_clusters, n_cols+n_zero_cols)
    #centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    #centers *= separation * cluster_std

    # Direzioni casuali e raggi diversi in un range di valori, meno probabile che centri siano sovrapposti risetto al metodo successivo
    directions = np.random.randn(n_clusters, n_cols+n_zero_cols)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Componenti 0 con centri in 0
    if n_zero_cols > 0:
        for col_idx in range(n_cols, n_cols+n_zero_cols):
            directions[:, col_idx] = np.abs(directions[:, col_idx])
            mask = np.random.random(directions.shape[0]) < zero_share
            directions[mask, col_idx] = 0.0
    # Raggi da distri uniforme
    #radii = np.random.uniform(
    #    low=0.5 * separation * cluster_std,
    #    high=1.5 * separation * cluster_std,
    #    size=(n_clusters, 1)
    #)
    # Raggi da distri lognormal, molti vicini alcuni lontani
    radii = np.random.lognormal(
        mean=np.log(separation * cluster_std),
        sigma=0.4,
        size=(n_clusters, 1)
    )
    centers = directions * radii

    # CENTRI COMPLETAMENTE CASUALI
    #centers = np.random.normal(
    #    loc=0.0,
    #    scale=separation * cluster_std,
    #    size=(n_clusters, n_cols+n_zero_cols)
    #)
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
    if n_overlap > 0:
        # Punti overlap solo tra cluster 0 e 1
        #if n_overlap > 0 and n_clusters >= 2:
        #    overlap_center = (centers[0] + centers[1]) / 2
        #    overlap_pts = np.random.normal(
        #        loc=overlap_center,
        #        scale=cluster_std * 1.2,
        #        size=(n_overlap, n_cols + n_zero_cols)
        #    )
        #    X.append(overlap_pts)
        #    y.extend([-2] * n_overlap)

        # Overlap in base alla distanza dei centri
        min_dist = np.inf
        max_dist = -np.inf
        closest_pair = None
        farthest_pair = None
        for i, j in combinations(range(n_clusters), 2):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist, closest_pair = d, (i, j)
            if d > max_dist:
                max_dist, farthest_pair = d, (i,j)
        # QUA PUOI PROVARE CON FARTHEST POINTS, o anche con una coppia random
        i, j = closest_pair
        overlap_center = (centers[i] + centers[j]) / 2
        overlap_pts = np.random.normal(
            loc=overlap_center,
            scale=cluster_std * 1.5,
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
            noise = np.random.normal(0, cluster_std*3.5, size=n_cols+n_zero_cols)
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

n_rows_dict = {'M' : 400,
               'L' : 1000}
n_clusters_dict = {'M': [3, 5],
                   'L': [6, 9]}
n_cols_list = [6]
separation_dict = {'ALTA' : {'M' : 14, 'L' : 21},
                   'MEDIA' : {'M' : 10, 'L' : 15},
                   'BASSA' : {'M' : 8, 'L' : 12}
                   }
outlier_scale_dict = [7]
rng = np.random.default_rng(420)
for dim, n_rows in n_rows_dict.items():
    out_dir = Path(Path(__file__).resolve().parent, 'dataframes/raw', dim)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n_cluster in n_clusters_dict[dim]:
        for descr,separation in separation_dict.items():
            array, labels, couple = generate_cluster_dataset(n_rows=n_rows,
                                                            n_cols=2,
                                                            n_zero_cols=0,
                                                            n_clusters=n_cluster,
                                                            n_overlap=int(n_rows/(2*n_cluster)),
                                                            n_outliers=int(n_rows/(4*n_cluster)),
                                                            separation=separation[dim],
                                                            cluster_std=1,
                                                            random_state=rng.integers(100,2000))
            df = pd.DataFrame(array)
            df['clusters'] = labels
            output_path = Path(out_dir, f'df_cluster_{n_cluster}_sep_{descr}.csv')
            df.to_csv(output_path, index = False)
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
            output_path = Path(out_dir, f'cluster_{n_cluster}_sep_{descr}.png')
            plt.savefig(output_path)
            plt.close()

# RPOVARE PIU VALORI DI RIGHE COLONNE N-CLUSTER
# AGGIUNGERE ZERO COL, PRIMA 1 POI DUE POI 3 AUMENTANDO LA 0 SHARE
# VARIARE OVERLAP POINTS E OUTLIERS POINTS
# VARIARE OUTLIER SCALE