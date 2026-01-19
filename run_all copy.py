# run_all.py
from __future__ import annotations

import subprocess
from pathlib import Path
import sys
import numpy as np

PYTHON = sys.executable

# columns you want in ambiguous list (must exist in outputs to appear)
ID_COLS = ["ALL_Uniq"]
OUT_ROOT = Path("fcm_step2")

INSTABILITY = Path("instability_points.py")

def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main() -> None:
    if not INSTABILITY.exists():
        raise FileNotFoundError(f"Missing {INSTABILITY}")
    
    # 3) Run analysis for each class output folder
    if not OUT_ROOT.exists():
        raise FileNotFoundError(f"Expected output folder {OUT_ROOT} not found after step-2 run.")

    id_cols_arg = ",".join(ID_COLS)

    for cls_dir in sorted([p for p in OUT_ROOT.iterdir() if p.is_dir()]):
        run([PYTHON, str(INSTABILITY), "--out_dir", str(cls_dir), "--id_cols", id_cols_arg])

    print("\nAll step-2 runs + analyses completed.")
    print(f"Explore: {OUT_ROOT}/<CLASS>/analysis/plots and {OUT_ROOT}/<CLASS>/analysis/tables")


if __name__ == "__main__":
    main()
