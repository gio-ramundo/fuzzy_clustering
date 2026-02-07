from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_path = Path(Path(__file__).resolve().parent) / 'df_class_L.csv'
df = pd.read_csv(df_path)
res_col = ['solar_power', 'solar_seas_ind', 'wind_power', 'wind_std',
       'offshore_wind_potential', 'hydro_potential', 'geothermal_potential',
       'evi', 'res_area']

mu_cols = [c for c in df.columns if c.startswith(f'mu_2.1')]
combined = pd.concat([df[res_col], df[mu_cols]], axis=1)
corr_matrix = combined.corr().loc[res_col, mu_cols]
plt.figure(figsize=(12, 2))
plt.rcParams.update({
    "font.family": "serif", 
    "font.weight" : "500",
    "axes.labelweight": "500",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 5 
})
ax = sns.heatmap(corr_matrix, 
        annot=True, 
        annot_kws={"size": 7},
        fmt=".2f",      # Forza 2 decimali per pulizia visiva
        cmap='RdBu_r',  # Red-Blue (invertito) è spesso più intuitivo per la correlazione
        vmin=-1, vmax=1, # Fissa la scala
        linewidths=.5,  # Aggiunge una griglia tra i quadratini
        cbar_kws={"shrink": .8}) # Rimpicciolisce la barra dei colori
ax.figure.subplots_adjust(left=0.8)
mu_labels = [r'$\mu_0$', r'$\mu_1$']
res_labels = ['Solar power', 'Solar seasonal ind', 'Wind power', 'Wind std', 'Offshore pot.', 'Hydroelectric pot.', 'Geothermal pot.', 'EVI', 'RES area']
ax.set_xticklabels(mu_labels, family='serif')
ax.set_yticklabels(res_labels, family='serif', rotation=0)
output_path = Path(Path(__file__).resolve().parent)/f'L_corr.pdf'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)