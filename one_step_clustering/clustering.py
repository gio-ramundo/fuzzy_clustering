from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import sys
sys.path.append(str(Path()))
from preprocess import _islands_filter, preprocess_dataframe

ID_COLS = ['ALL_Uniq']
# MAX_MIN LIMIT VALUES
AREA_VALUES = [3, 100000]
POP_VALUES = [200, 1000000]
EXCLUDE_COLUMNS = ['Shape_Leng','total_precipitation','hdd','cdd','elevation_max']

df_path = Path('data/df_complete.csv')
df = pd.read_csv(df_path)
# Filtering and preprocessing
df = _islands_filter(df, POP_VALUES, AREA_VALUES)
prep = preprocess_dataframe(
        df,
        drop_non_numeric=True,
        exclude_columns= EXCLUDE_COLUMNS,
        id_columns= ID_COLS,
        missing="median",
        scale="robust",
    )
rng = np.random.default_rng(420)
# XLSX with number of elements of different ec_labels in different clusters
out_root = Path(Path(__file__).resolve().parent, 'dataframes')
out_root.mkdir(parents=True, exist_ok=True)
xlsx_path = out_root / 'ec_labels_consistency.xlsx'
with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
    for n in range(4, 16):
        seed = rng.integers(0, 1_000_000)
        # KMeans clustering
        km = KMeans(
            n_clusters=n,
            n_init=30,
            random_state=seed
        )
        labels = km.fit_predict(prep.X)
        # dataframes exportation
        df1 = prep.df_kept_norm
        df1['cluster'] = labels
        out_dir = out_root / 'norm'
        out_dir.mkdir(parents=True, exist_ok=True)
        df1.to_csv(out_dir / f'df_{n}_clusters.csv', index = False)
        df1 = prep.df_kept_raw
        df1['cluster'] = labels
        out_dir = out_root / 'raw'
        out_dir.mkdir(parents=True, exist_ok=True)
        df1.to_csv(out_dir / f'df_{n}_clusters.csv', index = False)
        # xlsx writing
        tab_count = pd.crosstab(df1['ec_labels'], df1['cluster'])
        tab_count = tab_count.reindex(['XS', 'S', 'M', 'L'])
        tab_norm = tab_count.divide(tab_count.sum(axis=0), axis=1)
        sheet_name = f'{n}_clusters'
        current_row = 0
        pd.Series([f"Absolute distributions ec classes for cluster ({n} clusters)"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
        current_row += 1 
        tab_count.to_excel(writer, sheet_name=sheet_name, startrow=current_row)
        current_row += len(tab_count) + 2
        pd.Series([f"Percentage distributions (for Cluster)"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
        current_row += 1
        tab_norm.to_excel(writer, sheet_name=sheet_name, startrow=current_row)
