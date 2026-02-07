from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from itertools import product

def main() -> None:
    # Input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp_dir", type=str, default="", help="Input directory")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory")
    args = ap.parse_args()
    inp_dir = Path(args.inp_dir)
    out_dir = Path(args.out_dir)
    output_path = Path(out_dir / f'taub_values.xlsx')
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for csv_file in sorted([p for p in inp_dir.iterdir() if p.name.startswith('df')]):
            df = pd.read_csv(csv_file)
            fuz_cols = [c for c in df.columns if c.startswith("fuz_")]
            stab_cols = [c for c in df.columns if c.startswith("st_")]
            # Matrixes with kendalltau values and their p-values
            t_matrix = pd.DataFrame(index=fuz_cols, columns=stab_cols, dtype=float)
            p_matrix = t_matrix.copy()
            for col1, col2 in product(fuz_cols, stab_cols):
                t, p = kendalltau(df[col1], df[col2])
                t_matrix.loc[col1, col2] = t
                p_matrix.loc[col1, col2] = p
            # One sheet each consumption class
            sheet_name = f'{csv_file.name.replace("df_", "").replace(".csv", "")}'
            # Matrixes exportation
            current_row = 0
            pd.Series([f"Tau beta values"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
            current_row += 1
            t_matrix.to_excel(writer,sheet_name=sheet_name,startrow=current_row)
            current_row += len(t_matrix) + 2
            pd.Series([f"Tau beta p-values"]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, index=False, header=False)
            current_row += 1
            p_matrix.to_excel(writer,sheet_name=sheet_name,startrow=current_row)
                    
        
    
if __name__ == "__main__":
    main()