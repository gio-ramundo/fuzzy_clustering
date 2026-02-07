from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df_path = Path(Path(__file__).resolve().parent) / 'df_class_M.csv'
df = pd.read_csv(df_path)

res_col = ['solar_power', 'solar_seas_ind', 'wind_power', 'wind_std',
       'offshore_wind_potential', 'hydro_potential', 'geothermal_potential',
       'evi', 'res_area']

mu_cols = [c for c in df.columns if c.startswith(f'mu_2.4')]
mu_cols1 = [c for c in df.columns if c.startswith(f'mu_2.7')]
cl_hard = 'cluster_hard_2.4'
cl_hard1 = 'cluster_hard_2.7'

pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(df[res_col])

plt.figure(figsize=(8, 8))
maxu = df[mu_cols].to_numpy().max(axis=1)
sizes = 10 + 80 * (maxu - maxu.min()) / max(1e-12, (maxu.max() - maxu.min()))
scatter = plt.scatter(
    Z[:, 0], 
    Z[:, 1],
    s = sizes,
    c = df[cl_hard].astype(int),
    cmap='tab10',
    alpha=0.7
)
clusters = np.unique(df[cl_hard])
colors = scatter.cmap(scatter.norm(clusters))
handles = [
    plt.Line2D(
        [], [], marker='o', linestyle='',
        color=colors[i], label=f'Cluster {clusters[i]}'
    )
    for i in range(len(clusters))
]
plt.tight_layout() # Evita che la legenda venga tagliata al salvataggio
plt.savefig(Path(Path(__file__).resolve().parent) / f'M_2.4.pdf')
plt.close()

dict = {0 : 1, 2 : 0, 1 : 2, 3:3}
df[cl_hard1] = df[cl_hard1].map(dict)
plt.figure(figsize=(8, 8))
maxu = df[mu_cols1].to_numpy().max(axis=1)
sizes = 10 + 80 * (maxu - maxu.min()) / max(1e-12, (maxu.max() - maxu.min()))
scatter = plt.scatter(
    Z[:, 0], 
    Z[:, 1],
    s = sizes,
    c = df[cl_hard1].astype(int),
    cmap='tab10',
    alpha=0.7
)
clusters = np.unique(df[cl_hard1])
colors = scatter.cmap(scatter.norm(clusters))
handles = [
    plt.Line2D(
        [], [], marker='o', linestyle='',
        color=colors[i], label=f'Cluster {clusters[i]}'
    )
    for i in range(len(clusters))
]
plt.legend(handles=handles, loc='upper center', framealpha=1, fontsize=18, facecolor='white')
plt.tight_layout() # Evita che la legenda venga tagliata al salvataggio
plt.savefig(Path(Path(__file__).resolve().parent) / f'M_2.7.pdf')