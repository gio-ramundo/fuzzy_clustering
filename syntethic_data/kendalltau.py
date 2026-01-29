from __future__ import annotations

import argparse
from pathlib import Path
from scipy.stats import kendalltau
from itertools import product
from collections import defaultdict
import re

import numpy as np
import pandas as pd

def xlsx_function(class_name : str, input_dir : Path, output_dir : Path) -> None:
    output_path = Path(output_dir / f'{class_name}_taub_values.xlsx')
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
                    tau_matrices = []
                    for csv_file in files:
                        df = pd.read_csv(csv_file)
                        fuz_cols = [c for c in df.columns if c.startswith("fuz_")]
                        stab_cols = [c for c in df.columns if c.startswith("st_")]
                        # Kendall tau values
                        t_matrix = pd.DataFrame(index=fuz_cols, columns=stab_cols, dtype=float)
                        for col1, col2 in product(fuz_cols, stab_cols):
                            t, _ = kendalltau(df[col1], df[col2])
                            t_matrix.loc[col1, col2] = t
                        tau_matrices.append(t_matrix)
                    # Concatenation for mean and std calculation
                    concat_tau = pd.concat(tau_matrices, keys=range(len(tau_matrices)))
                    mean_tau_matrix = concat_tau.groupby(level=1).mean()
                    std_tau_matrix = concat_tau.groupby(level=1).std()
                    # Exportation, one sheet each dataset type
                    sheet_name = f'{dimensions_dir.name}_{base_name.replace("df_", "")}'
                    current_row = 0
                    pd.Series([f"Tau beta average values"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
                    current_row += 1
                    mean_tau_matrix.to_excel(writer,sheet_name=sheet_name,startrow=current_row)
                    current_row += len(mean_tau_matrix) + 2
                    pd.Series([f"Tau beta average std values"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
                    current_row += 1
                    std_tau_matrix.to_excel(writer,sheet_name=sheet_name,startrow=current_row)

def main() -> None:
    # Input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp_dir", type=str, default="", help="Input directory")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory")
    args = ap.parse_args()
    inp_dir = Path(args.inp_dir)
    out_dir = Path(args.out_dir)
    cls_list = []
    # Dataset types name extraction, one file for each class
    for dimensions_dir in sorted([p for p in (inp_dir).iterdir() if p.is_dir()]):
        stat_dir = dimensions_dir/'statistics'
        for class_dir in sorted([p for p in (stat_dir).iterdir() if p.is_dir()]):
            cls_list.append(class_dir.name)
        break
    for class_name in cls_list:
        xlsx_function(class_name, inp_dir, out_dir)
    
if __name__ == "__main__":
    main()