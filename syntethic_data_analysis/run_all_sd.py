from __future__ import annotations

import subprocess
from pathlib import Path
import sys
import numpy as np

PYTHON = sys.executable

OUT_ROOT = Path(Path(__file__).resolve().parent, 'dataframes')
OUT_ROOT1 = Path(Path(__file__).resolve().parent, 'results_analysis')

# Fuzziness values
FUZ_RANGE = np.arange(1.5, 3.1, 0.3)
# Perturbation noise values
NOISE_VALUES = [0.4, 0.6, 0.8, 1]
# FUZZINESS STABILITY INDICATORS
FUZZINESS_INDICATORS = ['entropy', 'gap'] # 'index']
STABILITY_INDICATORS = ['const_ass', 'thr_0.5'] # , 'thr_0.7', 'thr_0.9']

# Dataset generation
GENERATION = Path(Path(__file__).resolve().parent, "dataset_generation.py")

# Clustering
CLUSTERING = Path(Path(__file__).resolve().parent, "clustering.py")

# Fuzziness-stability statistics computation
STATISTICS = Path(Path(__file__).resolve().parent, "statistics.py")

# Fuzziness-stability statistics aggregations
STATISTICS_AGGR = Path(Path(__file__).resolve().parent, "statistics_aggr.py")

# Analisys
FUZ_STAB_LINK = Path(Path(__file__).resolve().parent, "fuz_stab_link.py")

# Kendalltau analysis
KENDALL = Path(Path(__file__).resolve().parent, "kendalltau.py")

def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main() -> None:
    if not GENERATION.exists():
        raise FileNotFoundError(f"Missing {GENERATION}")
    
    if not CLUSTERING.exists():
        raise FileNotFoundError(f"Missing {CLUSTERING}")
    
    if not STATISTICS.exists():
        raise FileNotFoundError(f"Missing {STATISTICS}")
    
    if not STATISTICS_AGGR.exists():
        raise FileNotFoundError(f"Missing {STATISTICS_AGGR}")

    if not FUZ_STAB_LINK.exists():
        raise FileNotFoundError(f"Missing {FUZ_STAB_LINK}")
    
    if not KENDALL.exists():
        raise FileNotFoundError(f"Missing {KENDALL}")

    run([PYTHON, str(GENERATION)])
    
    for dimensions_dir in sorted([p for p in (OUT_ROOT).iterdir() if p.is_dir()]):
        for cls_dir in sorted([p for p in (dimensions_dir/'raw').iterdir() if p.is_dir()]):
            for csv_file in [p for p in cls_dir.iterdir() if p.suffix.lower() == '.csv']:
                run([PYTHON, str(CLUSTERING), "--df_path", str(csv_file), 
                                            "--out_dir", str(OUT_ROOT/str(dimensions_dir.name)/'clustering'/str(cls_dir.name))])
    
    rng = np.random.default_rng(420)
    for dimensions_dir in sorted([p for p in (OUT_ROOT).iterdir() if p.is_dir()]):
        for cls_dir in sorted([p for p in (dimensions_dir/'clustering').iterdir() if p.is_dir()]):
            for csv_file in [p for p in cls_dir.iterdir()]:
                run([PYTHON, str(STATISTICS), "--df_path", str(csv_file), 
                                            "--out_dir", str(OUT_ROOT/str(dimensions_dir.name)/'statistics'/str(cls_dir.name)),
                                            "--seed", str(rng.integers(0, 1_000_000))])

    run([PYTHON, str(STATISTICS_AGGR), "--inp_dir", str(OUT_ROOT), 
                "--out_dir", str(OUT_ROOT1)])
    
    for dimensions_dir in sorted([p for p in (OUT_ROOT).iterdir() if p.is_dir()]):
        for cls_dir in sorted([p for p in (dimensions_dir/'statistics').iterdir() if p.is_dir()]):
            for csv_file in [p for p in cls_dir.iterdir()]:
                run([PYTHON, str(FUZ_STAB_LINK), "--df_path", str(csv_file), 
                                            "--out_dir", str(OUT_ROOT1/str(dimensions_dir.name)/'fuz_stab_link'/str(cls_dir.name))])
    
    run([PYTHON, str(KENDALL), "--inp_dir", str(OUT_ROOT),
                                "--out_dir", str(OUT_ROOT1)])

if __name__ == "__main__":
    main()