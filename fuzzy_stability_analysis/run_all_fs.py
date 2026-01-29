from __future__ import annotations

import subprocess
from pathlib import Path
import sys
import numpy as np

PYTHON = sys.executable

DF_PATH = Path('data/df_complete.csv')
OUT_ROOT = Path(Path(__file__).resolve().parent, 'results')

ID_COLS = ['ALL_Uniq']
# First step classes
EC_LABELS = 'ec_labels'
# Resource-only feature set for Step 2
CLUSTERING_FEATURES = [
    "solar_power",
    "solar_seas_ind",
    "wind_power",
    "wind_std",
    "offshore_wind_potential",
    "hydro_potential",
    "geothermal_potential",
    "evi",
    "res_area"
]
# MAX_MIN LIMIT VALUES
AREA_VALUES = [3, 100000]
POP_VALUES = [200, 1000000]
# n_clusters
N_CLUSTERS = {
    "XS" : 3,
    "S" : 3,
    "M" : 4,
    "L" : 2
}
# Fuzziness values
FUZ_RANGE = np.arange(1.5, 3.1, 0.3)
# Perturbation noise values
NOISE_VALUES = [0.8, 1, 1.2, 1.4]
# FUZZINESS STABILITY INDICATORS
FUZZINESS_INDICATORS = ['entropy', 'gap'] # 'index']
STABILITY_INDICATORS = ['const_ass', 'thr_0.5'] # , 'thr_0.7', 'thr_0.9']

# PCA t-SNE projections
PLOT = Path(Path(__file__).resolve().parent, "plots.py")

# Clustering
CLUSTERING = Path(Path(__file__).resolve().parent, "clustering.py")

# Clustering analysis
CL_ANALYSIS = Path(Path(__file__).resolve().parent, "analyse_fcm.py")

# Fuzziness-stability statistics computation
STATISTICS = Path(Path(__file__).resolve().parent, "fuz_stab_statistics.py")

# Analisys
FUZ_STAB_LINK = Path(Path(__file__).resolve().parent, "fuz_stab_link.py")

# Kendalltau analysis
KENDALL = Path(Path(__file__).resolve().parent, "kendalltau.py")

def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main() -> None:

    if not PLOT.exists():
        raise FileNotFoundError(f"Missing {PLOT}")
    if not CLUSTERING.exists():
        raise FileNotFoundError(f"Missing {CLUSTERING}")
    
    if not STATISTICS.exists():
        raise FileNotFoundError(f"Missing {STATISTICS}")
    
    if not FUZ_STAB_LINK.exists():
        raise FileNotFoundError(f"Missing {FUZ_STAB_LINK}")
    
    if not KENDALL.exists():
        raise FileNotFoundError(f"Missing {KENDALL}")

    run([PYTHON, str(PLOT), "--df_path", str(DF_PATH), 
        "--out_dir", str(OUT_ROOT/'plots')])

    run([PYTHON, str(CLUSTERING), "--df_path", str(DF_PATH), 
        "--out_dir", str(OUT_ROOT/'clustering')])

    for csv_file in sorted([p for p in (OUT_ROOT/'clustering').iterdir() if p.suffix == '.csv']):
        run([PYTHON, str(CL_ANALYSIS), "--df_path", str(csv_file)])

    rng = np.random.default_rng(42)
    for csv_file in sorted([p for p in (OUT_ROOT/'clustering').iterdir() if p.suffix == '.csv']):
        run([PYTHON, str(STATISTICS), "--df_path", str(csv_file), 
             "--out_dir", str(OUT_ROOT/'statistics'),
             "--seed", str(rng.integers(0, 1_000_000))])
    
    for csv_file in sorted([p for p in (OUT_ROOT/'statistics').iterdir() if p.suffix == '.csv' and p.name.startswith('df')]):
        run([PYTHON, str(FUZ_STAB_LINK), "--df_path", str(csv_file), 
                                        "--out_dir", str(OUT_ROOT/'fuz_stability_link')])
        
    run([PYTHON, str(KENDALL), "--inp_dir", str(OUT_ROOT/'statistics'),
        "--out_dir", str(OUT_ROOT/'fuz_stability_link')])

if __name__ == "__main__":
    main()