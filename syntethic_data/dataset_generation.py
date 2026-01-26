from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    print(f'Total dimensions = {dimensions[0]+dimensions[1]}')
    # Different dataset sizes
    for dim, n_rows in n_rows_dict.items():
        out_dir = Path(Path(__file__).resolve().parent, 'dataframes', f'{dimensions[0]+dimensions[1]} dimensions', 'raw', dim)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_clusters = n_clusters_dict[dim]
        # Different centers separation
        for descr,separation in separation_dict[dimensions[0]].items():
            print(f'Class {dim}, separation {descr}')
            if descr == 'LOW':
                if dim == 'L':
                    separation *= 1.4
                if dim == 'M':
                    separation *= 1.2
                if dim == 'TEST':
                    separation *= 0.8
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
            print('Variance explained by PCA components:')
            for i in range(len(df.columns)):
                explained_variance += pca.explained_variance_ratio_[i]
                print(explained_variance, end=" ")
            print(' ')
            df['clusters'] = labels
            # Exportation
            output_path = Path(out_dir, f'df_{dim}_sep_{descr}.csv')
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
                output_path = Path(out_plot_dir, f'cluster_{n_clusters_dict[dim]}_sep_{descr}.png')
                plt.savefig(output_path)
                plt.close()

#Total dimensions = 2
#Class XS, separation LOW
#Variance explained by PCA components:
#0.875732057207197 0.9999999999999999  
#Class XS, separation HIGH
#Variance explained by PCA components:
#0.9193041351666832 1.0  
#Class S, separation LOW
#Variance explained by PCA components:
#0.7855081693408502 1.0
#Class S, separation HIGH
#Variance explained by PCA components:
#0.9361374206066122 1.0
#Class M, separation LOW
#Variance explained by PCA components:
#0.8247606652182111 1.0
#Class M, separation HIGH
#Variance explained by PCA components:
#0.9259721793470749 1.0
#Class L, separation LOW
#Variance explained by PCA components:
#0.9355927020133648 0.9999999999999999
#Class L, separation HIGH
#Variance explained by PCA components:
#0.941691656609187 1.0
#Class TEST, separation LOW
#Variance explained by PCA components:
#0.9418100463788895 0.9999999999999999
#Class TEST, separation HIGH
#Variance explained by PCA components:
#0.7819679554788134 1.0
#Total dimensions = 5
#Class XS, separation LOW
#Variance explained by PCA components:
#0.5617701686919021 0.7345947550257855 0.8429671685331316 0.942147412435249 0.9999999999999999
#Class XS, separation HIGH
#Variance explained by PCA components:
#0.7393919163480617 0.8994803823749244 0.9391008858790381 0.9753809273745885 1.0
#Class S, separation LOW
#Variance explained by PCA components:
#0.5702413137386225 0.7966206342722849 0.8769187697638529 0.9470800290235215 1.0
#Class S, separation HIGH
#Variance explained by PCA components:
#0.7405581467086491 0.8506885517522028 0.9118249890844122 0.9654060968184885 0.9999999999999999
#Class M, separation LOW
#Variance explained by PCA components:
#0.4491068689964971 0.806643630989228 0.8982216568432525 0.9626386970206405 0.9999999999999998
#Class M, separation HIGH
#Variance explained by PCA components:
#0.637372995674234 0.9111078213954358 0.960294724279886 0.9925773589811469 1.0
#Class L, separation LOW
#Variance explained by PCA components:
#0.6934214088513344 0.8046656062998385 0.8974155214118613 0.9800273974185139 0.9999999999999999
#Class L, separation HIGH
#Variance explained by PCA components:
#0.5697880771269207 0.7200566998119146 0.8547867604260884 0.9593258147310725 0.9999999999999999
#Class TEST, separation LOW
#Variance explained by PCA components:
#0.3818934559592954 0.7106690598281573 0.8682454234002707 0.9749993242041912 0.9999999999999999
#Class TEST, separation HIGH
#Variance explained by PCA components:
#0.6323505124591802 0.8787142315065892 0.9507809315655801 0.9827107961920872 0.9999999999999999
#Total dimensions = 9
#Class XS, separation LOW
#Variance explained by PCA components:
#0.356002692594022 0.5032170492640867 0.6094801252532227 0.7118751298555602 0.809393624713182 0.9046702702818793 0.9412211201883122 0.9746662667047084 1.0
#Class XS, separation HIGH
#Variance explained by PCA components:
#0.6068491378530452 0.758342935822621 0.8061971393244907 0.8531212130140131 0.8977786658618078 0.9403748298994153 0.9651916216593479 0.9869134241233881 1.0
#Class S, separation LOW
#Variance explained by PCA components:
#0.34457643729392545 0.5298412795864065 0.6257020705590873 0.7188388102487873 0.8041403789349532 0.8881719036052748 0.9378064947868108 0.9759401149190029 1.0
#Class S, separation HIGH
#Variance explained by PCA components:
#0.8276567310982043 0.8860693906917093 0.9088135034648203 0.9304688556062014 0.9512032275773842 0.9701284476171673 0.9843307708894515 0.9946758755263166 1.0
#Class M, separation LOW
#Variance explained by PCA components:
#0.4124431556590693 0.552620189412821 0.6560134125705951 0.7584294386616032 0.8431204248397609 0.9165960973530356 0.9566814193881368 0.9798936090840612 0.9999999999999998
#Class M, separation HIGH
#Variance explained by PCA components:
#0.6046489466004705 0.7863769906563799 0.8682265678562445 0.9045586974097112 0.9360697674917272 0.9666931652970835 0.9810004853135154 0.9911408515909448 0.9999999999999999
#Class L, separation LOW
#Variance explained by PCA components:
#0.4413059510778257 0.5362515476574502 0.6293502599405287 0.7186521468618725 0.8040076841606856 0.8649314917751159 0.9235217172430714 0.9694594626138311 0.9999999999999999
#Class L, separation HIGH
#Variance explained by PCA components:
#0.6383915858403647 0.70468799477505 0.7692564154171115 0.8257158572198471 0.8791861572435615 0.9276585853774078 0.9695691062413945 0.9871294665025823 0.9999999999999998
#Class TEST, separation LOW
#Variance explained by PCA components:
#0.2740298644149762 0.4452326026858645 0.5719275423517317 0.6873753710887451 0.7917167178762019 0.893998968649271 0.9422253443390527 0.974774709639392 1.0
#Class TEST, separation HIGH
#Variance explained by PCA components:
#0.5555958013936855 0.7146920048152514 0.7945221837723806 0.8455363370470312 0.895173638454566 0.9427182631716475 0.9745963209865187 0.9883568577317565 1.0000000000000002