from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from run_all_fs import FUZ_RANGE, FUZZINESS_INDICATORS

# Correlation matrixes functions
def matrix_computation(df : pd.DataFrame, cols1 : List[str], cols : List[str]):
    pearson_mat = pd.DataFrame(
        index=cols1,
        columns=cols,
        dtype=float
    )
    spearman_mat = pearson_mat.copy()
    for fuzz_col in cols1:
        for stab_col in cols:
            x = df[fuzz_col]
            y = df[stab_col]
            mask = x.notna() & y.notna()
            x_clean = x[mask]
            y_clean = y[mask]
            if len(x_clean) > 2  and x_clean.nunique() > 1 and y_clean.nunique() > 1:
                pearson_mat.loc[fuzz_col, stab_col] = pearsonr(x_clean, y_clean)[0]
                spearman_mat.loc[fuzz_col, stab_col] = spearmanr(x_clean, y_clean)[0]
            else:
                pearson_mat.loc[fuzz_col, stab_col] = np.nan
                spearman_mat.loc[fuzz_col, stab_col] = np.nan
    return pearson_mat, spearman_mat
# Fuzziness stability correlations matrix construction and exportation
def correlation_matrixes(df : pd.DataFrame, fuzzy_cols : List[str], stab_cols : List[str], output_dir : Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir, "correlations_matrixes.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        current_row = 0
        sheet_name = f"Pearson"
        current_row1 = 0
        sheet_name1 = f"Spearman"
        for fuz in FUZ_RANGE:
            fuzzy_cols1 = [col for col in fuzzy_cols if col.startswith(f"fuz_{round(fuz,1)}")]
            matrixes = matrix_computation(df, fuzzy_cols1, stab_cols)
            pd.Series([f"Fuzziness coefficient: {round(fuz,1)}"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
            current_row += 1 
            matrixes[0].round(3).to_excel(writer, sheet_name=sheet_name, startrow=current_row)
            current_row += len(matrixes[0]) + 2
            
            pd.Series([f"Fuzziness coefficient: {round(fuz,1)}"]).to_excel(writer, sheet_name=sheet_name1, startrow=current_row1, index=False, header=False)
            current_row1 += 1 
            matrixes[1].round(3).to_excel(writer, sheet_name=sheet_name1, startrow=current_row1)
            current_row1 += len(matrixes[1]) + 2
           
# Fuzziness interval stability means
def fuzziness_by_intervals(dataframe : pd.DataFrame, stab_cols : List[str], output_dir : Path, bin_width : float=0.1) -> None:
    bins = np.arange(0, 1 + bin_width, bin_width)
    labels = [f"{round(bins[i],2)}–{round(bins[i+1],2)}"
              for i in range(len(bins) - 1)]
    out_path = output_dir / "fuzziness_intervals.xlsx"
    ordered_fuzzy_cols = [f'fuz_{round(m,1)}_' + ind for ind in FUZZINESS_INDICATORS for m in FUZ_RANGE]
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for fcol in ordered_fuzzy_cols:
            temp = dataframe.copy()
            temp["fuzz_bin"] = pd.cut(
                temp[fcol],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            mean_stab = (
                temp
                .groupby("fuzz_bin")[stab_cols]
                .mean()
            )
            counts = (
                temp
                .groupby("fuzz_bin")
                .size()
                .rename("n_obs")
            )
            summary = mean_stab.join(counts)
            summary.round(3).to_excel(writer, sheet_name=fcol)

# Fuzziness quantiles stability
def fuzziness_by_quantiles(dataframe : pd.DataFrame, stab_cols : List[str], output_dir : Path, n_quantiles : int=10) -> None:    
    out_path = output_dir / "fuzziness_quantiles.xlsx"
    ordered_fuzzy_cols = [f'fuz_{round(m,1)}_' + ind for ind in FUZZINESS_INDICATORS for m in FUZ_RANGE]
    with pd.ExcelWriter(out_path) as writer:
        for fcol in ordered_fuzzy_cols:
            temp = dataframe.copy()
            temp["fuzz_quantile"] = pd.qcut(
                temp[fcol],
                q=n_quantiles,
                duplicates="drop"
            )
            mean_stab = (
                temp
                .groupby("fuzz_quantile")[stab_cols]
                .mean()
            )
            counts = (
                temp
                .groupby("fuzz_quantile")
                .size()
                .rename("n_obs")
            )
            summary = mean_stab.join(counts)
            summary.round(3).to_excel(writer, sheet_name=fcol)

def main() -> None:
    # Input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_path", type=str, default="", help="Input dataframes")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder")
    args = ap.parse_args()
    df_path = Path(args.df_path)
    df_cls = pd.read_csv(df_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stability columns
    fuzziness_cols = [c for c in df_cls.columns if c.startswith("fuz_")]
    stability_cols = [c for c in df_cls.columns if c.startswith("st_")]
    # Summary statistics
    class_name = df_path.name.replace('df_class_', '').replace('.csv','')
    class_dir = out_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    correlation_matrixes(df_cls, fuzziness_cols, stability_cols, class_dir)
    fuzziness_by_intervals(df_cls, stability_cols, class_dir)
    fuzziness_by_quantiles(df_cls, stability_cols, class_dir)
    


if __name__ == "__main__":
    main()