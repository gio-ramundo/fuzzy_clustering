from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from run_all_sd import FUZZINESS_INDICATORS, FUZ_RANGE

# Fuzziness and stability indicators means and quantiles in diferrent data types
def mean_indicators_computation(dataframe : pd.DataFrame, stab_cols : List[str], output_dir : Path, output_name : str, n_quantiles : int=10, n_intervals : int=10) -> None:    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir, output_name+".xlsx")
    # Different points type
    mask_dict = {
        'All' : True,
        'Normal' : True,
        'Overlap' : [-1],
        'Outliers' : [-2, -3],
        'Outliers1' : [-2],
        'Outliers2' : [-3],
    }
    ordered_fuzzy_cols = [f'fuz_{round(m,1)}_' + ind for ind in FUZZINESS_INDICATORS for m in FUZ_RANGE]
    with pd.ExcelWriter(out_path) as writer:
        means_fuz = pd.DataFrame(
            index = ordered_fuzzy_cols,
            columns = mask_dict.keys(),
            dtype = float
        )
        means_stab = pd.DataFrame(
            index = stab_cols,
            columns = mask_dict.keys(),
            dtype = float
        )
        for key in mask_dict:
            if key == 'All':
                df1 = dataframe.copy()
            elif key == 'Normal':
                df1 = dataframe[~dataframe['clusters'].isin([-1, -2, -3])].copy()
            else:
                df1 = dataframe[dataframe['clusters'].isin(mask_dict[key])].copy()
            means_fuz[key] = df1[ordered_fuzzy_cols].mean()
            means_stab[key] = df1[stab_cols].mean()
        sheet_name = f"Means"    
        current_row = 0
        pd.Series([f"Fuzziness indicators mean"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
        current_row += 1
        means_fuz.round(3).to_excel(writer, sheet_name=sheet_name, startrow=current_row)
        current_row += len(means_fuz) + 2
        pd.Series([f"Stability indicators mean"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
        current_row += 1
        means_stab.round(3).to_excel(writer, sheet_name=sheet_name, startrow=current_row)
        
        # Fuzziness quantiles and intervals
        sheet_name1 = f"Fuzzy quantiles"    
        current_row1 = 0
        sheet_name2 = f"Fuzzy intervals"    
        current_row2 = 0
        for fcol in ordered_fuzzy_cols:
            q = np.arange(0, n_quantiles + 1) / n_quantiles
            bin_width = 1/n_intervals
            bins = np.arange(0, 1 + bin_width, bin_width)
            labels = [f"{round(bins[i],2)}–{round(bins[i+1],2)}"
                      for i in range(len(bins) - 1)]
            quantiles = pd.DataFrame(
                index = q,
                columns = mask_dict.keys(),
                dtype = float
            )
            intervals = pd.DataFrame(
                columns = mask_dict.keys(),
                dtype = int
            )
            for key in mask_dict:
                if key == 'All':
                    df1 = dataframe.copy()
                elif key == 'Normal':
                    df1 = dataframe[~dataframe['clusters'].isin([-1, -2, -3])].copy()
                else:
                    df1 = dataframe[dataframe['clusters'].isin(mask_dict[key])].copy()
                quantiles[key] = df1[fcol].quantile(q)
                intervals[key] = pd.cut(df1[fcol], bins=bins, 
                                        include_lowest=True, right=True
                                    ).value_counts().sort_index()
            intervals.index = labels
            pd.Series([f"{fcol}"]).to_excel(writer, sheet_name=sheet_name1, startrow=current_row1, index=False, header=False)
            current_row1 += 1
            quantiles.round(3).to_excel(writer, sheet_name=sheet_name1, startrow=current_row1)
            current_row1 += len(quantiles) + 2
            pd.Series([f"{fcol}"]).to_excel(writer, sheet_name=sheet_name2, startrow=current_row2, index=False, header=False)
            current_row2 += 1
            intervals.round(3).to_excel(writer, sheet_name=sheet_name2, startrow=current_row2)
            current_row2 += len(intervals) + 2

        # Stability quantiles and intervals
        sheet_name1 = f"Stability quantiles"    
        current_row1 = 0
        sheet_name2 = f"Stability intervals"    
        current_row2 = 0
        for scol in stab_cols:
            q = np.arange(0, n_quantiles + 1) / n_quantiles
            bin_width = 1/n_intervals
            bins = np.arange(0, 1 + bin_width, bin_width)
            labels = [f"{round(bins[i],2)}–{round(bins[i+1],2)}"
                      for i in range(len(bins) - 1)]
            quantiles = pd.DataFrame(
                index = q,
                columns = mask_dict.keys(),
                dtype = float
            )
            intervals = pd.DataFrame(
                columns = mask_dict.keys(),
                dtype = int
            )
            for key in mask_dict:
                if key == 'All':
                    df1 = dataframe.copy()
                elif key == 'Normal':
                    df1 = dataframe[~dataframe['clusters'].isin([-1, -2, -3])].copy()
                else:
                    df1 = dataframe[dataframe['clusters'].isin(mask_dict[key])].copy()
                quantiles[key] = df1[scol].quantile(q)
                intervals[key] = pd.cut(df1[scol], bins=bins, 
                                        include_lowest=True, right=True
                                    ).value_counts().sort_index()
            intervals.index = labels
            pd.Series([f"{scol}"]).to_excel(writer, sheet_name=sheet_name1, startrow=current_row1, index=False, header=False)
            current_row1 += 1
            quantiles.to_excel(writer, sheet_name=sheet_name1, startrow=current_row1)
            current_row1 += len(quantiles) + 2
            pd.Series([f"{scol}"]).to_excel(writer, sheet_name=sheet_name2, startrow=current_row2, index=False, header=False)
            current_row2 += 1
            intervals.to_excel(writer, sheet_name=sheet_name2, startrow=current_row2)
            current_row2 += len(intervals) + 2

def main() -> None:
    # Input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp_dir", type=str, default="", help="Input dataframes")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder")
    args = ap.parse_args()
    inp_dir = Path(args.inp_dir)
    df = pd.read_csv(inp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stability_columns = [c for c in df.columns if c.startswith("st_")]
    out_name = inp_dir.name.replace('df_', '').replace('.csv','')
    mean_indicators_computation(df, stability_columns, out_dir, out_name)
    
if __name__ == "__main__":
    main()