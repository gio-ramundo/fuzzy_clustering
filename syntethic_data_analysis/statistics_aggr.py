from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
from collections import defaultdict
import re

import numpy as np
import pandas as pd

from run_all_sd import FUZZINESS_INDICATORS, FUZ_RANGE

# Fuzziness and stability indicators means and quantiles in diferrent data types fro single files
def mean_indicators_computation(dataframe : pd.DataFrame, stab_cols : List[str], output_dir : Path, output_name : str, n_quantiles : int=10, n_intervals : int=10) -> None:    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir, output_name+".xlsx")
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

# Dataset type means
def class_mean(class_name : str, input_dir : Path, output_dir : Path) -> None:
    # Different points type
    mask_dict = {
        'All' : True,
        'Normal' : True,
        'Overlap' : [-1],
        'Outliers' : [-2, -3],
        'Outliers1' : [-2],
        'Outliers2' : [-3],
    }
    output_path = Path(output_dir / f'{class_name}_statistic_means.xlsx')
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for dimensions_dir in sorted([p for p in (input_dir).iterdir() if p.is_dir()]):
            stat_dir = dimensions_dir/'statistics'
            # Datasets name extraction and iteration
            for class_dir in sorted([p for p in (stat_dir).iterdir() if p.is_dir()]):
                df_groups = defaultdict(list)
                if class_dir.name != class_name:
                    continue
                for csv_file in sorted(class_dir.iterdir()):
                    base_name = re.sub(r'_\d+\.csv$', '', csv_file.name)
                    df_groups[base_name].append(csv_file)
                # Iteration for datasets with all caractheristics equal
                for base_name, files in df_groups.items():
                    fuz_means_tot = []
                    stab_means_tot = []
                    for csv_file in files:
                        df = pd.read_csv(csv_file)
                        fuz_cols = [c for c in df.columns if c.startswith("fuz_")]
                        stab_cols = [c for c in df.columns if c.startswith("st_")]
                        means_fuz = pd.DataFrame(
                            index = fuz_cols,
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
                                df1 = df.copy()
                            elif key == 'Normal':
                                df1 = df[~df['clusters'].isin([-1, -2, -3])].copy()
                            else:
                                df1 = df[df['clusters'].isin(mask_dict[key])].copy()
                            means_fuz[key] = df1[fuz_cols].mean()
                            means_stab[key] = df1[stab_cols].mean()
                        fuz_means_tot.append(means_fuz)
                        stab_means_tot.append(means_stab)
                    # Concatenation for mean and std calculation
                    concat_fuz = pd.concat(fuz_means_tot, keys=range(len(fuz_means_tot)))
                    concat_stab = pd.concat(stab_means_tot, keys=range(len(stab_means_tot)))
                    mean_fuz_matrix = concat_fuz.groupby(level=1).mean()
                    mean_stab_matrix = concat_stab.groupby(level=1).mean()
                    # Exportation, one sheet each dataset type
                    sheet_name = f'{dimensions_dir.name}_{base_name.replace("df_", "")}'
                    current_row = 0
                    pd.Series([f"Fuzziness average values"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
                    current_row += 1
                    mean_fuz_matrix.to_excel(writer,sheet_name=sheet_name,startrow=current_row)
                    current_row += len(mean_fuz_matrix) + 2
                    pd.Series([f"Stability average values"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
                    current_row += 1
                    mean_stab_matrix.to_excel(writer,sheet_name=sheet_name,startrow=current_row)

def main() -> None:
    # Input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp_dir", type=str, default="", help="Input dataframes")
    ap.add_argument("--out_dir", type=str, default="", help="Output folder")
    args = ap.parse_args()
    inp_dir = Path(args.inp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_list = []
    # Single files statistics
    for dimensions_dir in sorted([p for p in inp_dir.iterdir() if p.is_dir()]):
        dimension = dimensions_dir.name
        for cls_dir in sorted([p for p in (dimensions_dir/'statistics').iterdir() if p.is_dir()]):
            cls_name = cls_dir.name
            class_list.append(cls_name)
            out_dir1 = out_dir/dimension/'statistics_aggregation'/cls_name
            for csv_file in sorted([p for p in (cls_dir).iterdir()]):
                df = pd.read_csv(csv_file)
                stability_columns = [c for c in df.columns if c.startswith("st_")]
                out_name = csv_file.name.replace('df_', '').replace('.csv','')
                mean_indicators_computation(df, stability_columns, out_dir1, out_name)
    # Class mean statistics
    for class_name in class_list:
        class_mean(class_name, inp_dir, out_dir)
    
    
if __name__ == "__main__":
    main()