from __future__ import annotations

import subprocess
from pathlib import Path
import sys
import numpy as np

PYTHON = sys.executable

# columns you want in ambiguous list (must exist in outputs to appear)
ID_COLS = ["ALL_Uniq"]

# Hyperparameter
HYPER_SCRIPT = Path("hyperparam_definition.py")

# Two step
OUT_ROOT = Path("fcm_step2")

STEP2_SCRIPT = Path("run_fcm_step2.py")

# tolerate either spelling
ANALYSIS_SCRIPT_CANDIDATES = [Path("analyse_fcm.py"), Path("analyze_fcm.py")]

def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main() -> None:
    if not HYPER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing {HYPER_SCRIPT}")
    
    if not STEP2_SCRIPT.exists():
        raise FileNotFoundError(f"Missing {STEP2_SCRIPT}")

    analysis_script = None
    for cand in ANALYSIS_SCRIPT_CANDIDATES:
        if cand.exists():
            analysis_script = cand
            break
    if analysis_script is None:
        raise FileNotFoundError(
            f"Missing analysis script. Expected one of: {', '.join(str(p) for p in ANALYSIS_SCRIPT_CANDIDATES)}"
        )

    # 1) Run preliminary script
    run([PYTHON, str(HYPER_SCRIPT)])

    # 2) Run step-2 clustering
    run([PYTHON, str(STEP2_SCRIPT)])

    # 3) Run analysis for each class output folder
    if not OUT_ROOT.exists():
        raise FileNotFoundError(f"Expected output folder {OUT_ROOT} not found after step-2 run.")

    id_cols_arg = ",".join(ID_COLS)

    for cls_dir in sorted([p for p in OUT_ROOT.iterdir() if p.is_dir()]):
        if not (cls_dir / "assignments_memberships_norm.csv").exists():
            print(f"Skipping {cls_dir}: missing assignments_memberships_norm.csv")
            continue

        run([PYTHON, str(analysis_script), "--out_dir", str(cls_dir), "--id_cols", id_cols_arg])

    print("\nAll step-2 runs + analyses completed.")
    print(f"Explore: {OUT_ROOT}/<CLASS>/analysis/plots and {OUT_ROOT}/<CLASS>/analysis/tables")

    # INSERIRE SCRIPT PER VEDERE PUNTI BORDERLINE

if __name__ == "__main__":
    main()
