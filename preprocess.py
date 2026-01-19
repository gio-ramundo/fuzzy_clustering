from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Preprocessed:
    X: np.ndarray
    init_num_feat_names : List[str]
    fin_feat_names : List[str]
    df_kept_raw: pd.DataFrame  # rows aligned with X (after dropping/imputing)
    df_kept_norm: pd.DataFrame
    scaler_name: str

def _exclude_not_relevant_features(
    df: pd.DataFrame,
    exclude_columns: List[str],
) -> pd.DataFrame:
    df_work = df.copy()
    # Drop excluded columns if present
    to_drop = [c for c in exclude_columns if c in df_work.columns]
    if to_drop:
        df_work = df_work.drop(columns=to_drop)
    return df_work
    
def _select_numeric_features(
    df: pd.DataFrame,
    drop_non_numeric: bool,
) -> pd.DataFrame:
    df_work = df.copy()
    if drop_non_numeric:
        df_num = df_work.select_dtypes(include=[np.number]).copy()
    else:
        # attempt to coerce everything to numeric if possible
        df_num = df_work.apply(pd.to_numeric, errors="coerce")
    return df_num

def _no_id(
    df: pd.DataFrame,
    id_columns : List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    df_work = df.copy()

    # Drop id/excluded columns if present
    to_drop = [c for c in id_columns if c in df_work.columns]
    if to_drop:
        df_work = df_work.drop(columns=to_drop)

    feature_names = list(df_work.columns)
    return df_work, feature_names

def _islands_filter(df_num: pd.DataFrame, pop : List[int], area : List[int]) -> pd.DataFrame:
    df_filter = df_num[(df_num['IslandArea'] >= area[0]) & (df_num['IslandArea'] <= area[1])].copy()
    df_filter = df_filter[(df_filter['Population'] >= pop[0]) & (df_filter['Population'] <= pop[1])]
    return df_filter

def _handle_missing(df_num: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "drop":
        return df_num.dropna(axis=0)
    if strategy == "median":
        return df_num.fillna(df_num.median(numeric_only=True))
    if strategy == "mean":
        return df_num.fillna(df_num.mean(numeric_only=True))
    raise ValueError(f"Unknown missing strategy: {strategy}")

def _dimensions_reduction(df_num: pd.DataFrame) -> Tuple[pd.Dataframe, List[str]]:
    df_red = df_num.copy()

    # Correlated variables
    if set(['gdp_2019', 'Population', 'urban_area', 'ec_2019']).issubset(set(df_red.columns)):
        X=df_red[['gdp_2019', 'Population', 'urban_area', 'ec_2019']]
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X_scaled)
        df_red['gdp_cons_pop_urban_merged']=X_pca
        df_red = df_red.drop(columns=['gdp_2019', 'ec_2019', 'Population', 'urban_area', 'urban_area_rel'])
    
    # Variables merging
    if set(['res_area', 'IslandArea']).issubset(set(df_red.columns)):
        df_red['res_area']=(df_red['res_area']/100)*df_red['IslandArea']
        df_red = df_red.drop(columns=['IslandArea'])
    
    feature_names = list(df_red.columns)
    return df_red, feature_names

def _scale(df: pd.DataFrame, method: str) -> Tuple[np.ndarray, str]:

    df_scaled = df.copy()

    # Different techniques for different columns
    robust_features = ['solar_power', 'mean_temperature', 'wind_std', 'evi']
    scaler = RobustScaler()
    for col in robust_features:
        if col in df_scaled.columns:
            df_scaled[col] = scaler.fit_transform(df_scaled[[col]])

    yeo_features = ['gdp_cons_pop_urban_merged', 'wind_power']
    yeo_pipeline = Pipeline([
        ('yeojohnson', PowerTransformer(method='yeo-johnson', standardize= False)),
        ('robust_scaler', RobustScaler())
    ])
    for col in yeo_features:
        if col in df_scaled.columns:
            df_scaled[col] = yeo_pipeline.fit_transform(df_scaled[[col]])
    
    log_robust_features = ['res_area', 'Population_density', 'solar_seas_ind']
    log_pipeline = Pipeline([
            ('log_transformer', FunctionTransformer(np.log1p, validate=True)),
            ('robust_scaler', RobustScaler())
        ])
    for col in log_robust_features:
        if col in df_scaled.columns:
            df_scaled[col] = log_pipeline.fit_transform(df_scaled[[col]])

    zeros_log=['offshore_wind_potential', 'hydro_potential']
    standscaler = StandardScaler(with_mean=False)
    for col in zeros_log:
        if col in df_scaled.columns:
            zero_mask = df_scaled[col] <= 0
            df_scaled.loc[zero_mask, col] = np.nan
            df_scaled[col] = np.log1p(df_scaled[col])
            df_scaled[col] = standscaler.fit_transform(df_scaled[[col]])
            df_scaled.loc[zero_mask, col] = 0

    yeo_pipeline = Pipeline([
        ('yeojohnson', PowerTransformer(method='yeo-johnson', standardize= False)),
        ('standard_scaler', standscaler)
    ])
    if 'geothermal_potential' in df_scaled.columns:
        zero_mask = df_scaled['geothermal_potential'] <= 0
        df_scaled.loc[zero_mask, 'geothermal_potential'] = np.nan
        df_scaled['geothermal_potential'] = yeo_pipeline.fit_transform(df_scaled[['geothermal_potential']])
        df_scaled.loc[zero_mask, 'geothermal_potential'] = 0

    # Remaining columns, if present
    method = method.lower()
    if method == "none":
        return df_scaled, "none"
    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scale method: {method}")
    for col in df.columns:
        if col not in robust_features + yeo_features + log_robust_features + zeros_log + ['geothermal_potential'] :
            df_scaled[col] = scaler.fit_transform(df_scaled[[col]])

    return df_scaled, method

def preprocess_dataframe(
    df: pd.DataFrame,
    id_columns: List[str] = [],
    exclude_columns: List[str] = [],
    drop_non_numeric: bool = True,
    missing: str = "median",
    scale: str = "robust"
) -> Preprocessed:
    df_exclude = _exclude_not_relevant_features(df, exclude_columns=exclude_columns)
    df_num = _select_numeric_features(df_exclude, drop_non_numeric=drop_non_numeric)
    df_clust, init_num_feat_names = _no_id(df_num, id_columns)
    df_fin = _handle_missing(df_clust, strategy=missing)
    finite_mask = np.isfinite(df_fin.to_numpy()).all(axis=1)
    # keep only rows that are fully finite
    df_fin = df_fin.loc[finite_mask].copy()
    df_red, fin_feat_names = _dimensions_reduction(df_fin)
    df_kept_norm, scaler_name = _scale(df_red, method=scale)
    Xs = df_kept_norm.to_numpy(dtype=float)
    # df_kept should align to Xs
    df_kept_raw = df_exclude.loc[df_fin.index].copy()
    return Preprocessed(X=Xs, init_num_feat_names = init_num_feat_names, fin_feat_names = fin_feat_names, df_kept_raw=df_kept_raw, df_kept_norm=df_kept_norm, scaler_name=scaler_name)