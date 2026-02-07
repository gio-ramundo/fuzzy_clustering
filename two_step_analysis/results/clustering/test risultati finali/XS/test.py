from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_path = Path(Path(__file__).resolve().parent) / 'df_class_XS.csv'
df = pd.read_csv(df_path)

res_col = ['solar_power', 'solar_seas_ind', 'wind_power', 'wind_std',
       'offshore_wind_potential', 'hydro_potential', 'geothermal_potential',
       'evi', 'res_area']

mu_cols = [c for c in df.columns if c.startswith(f'mu_2.4')]
plt.figure(figsize=(6,6))
plt.figure()
df['max_mu'] = df[mu_cols].max(axis=1)
sizes = df['max_mu'].to_numpy()
sizes = 5 + 70 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
plt.figure(figsize=(8, 8))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,          # dimensione base
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 16
})
scatter = plt.scatter(
    df['solar_power'],
    df['offshore_wind_potential'],
    s=sizes,
    c=df['cluster_hard_2.1'],
    alpha=0.7
)
clusters = np.unique(df[f'cluster_hard_2.1'])
colors = scatter.cmap(scatter.norm(clusters))
handles = [
    plt.Line2D(
        [], [], marker='o', linestyle='',
        color=colors[i], label=f'Cluster {clusters[i]}'
    )
    for i in range(len(clusters))
]
plt.legend(handles=handles, framealpha=1, facecolor='white')
output_path = Path(Path(__file__).resolve().parent)/f'solar_wind_scatter.pdf'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)
plt.close()

plt.figure(figsize=(8, 8))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,          # dimensione base
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 16
})
plt.scatter(
    df['solar_power'],
    df['hydro_potential'],
    s=sizes,
    c=df['cluster_hard_2.1'],
    alpha=0.7
)
output_path = Path(Path(__file__).resolve().parent)/f'solar_hydro_scatter.pdf'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)
plt.close()

plt.figure(figsize=(8, 8))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,          # dimensione base
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 16
})
plt.scatter(
    df['offshore_wind_potential'],
    df['hydro_potential'],
    s=sizes,
    c=df['cluster_hard_2.1'],
    alpha=0.7
)
output_path = Path(Path(__file__).resolve().parent)/f'wind_hydro_scatter.pdf'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)