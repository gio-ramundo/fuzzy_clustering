from __future__ import annotations

import subprocess
from pathlib import Path
import sys
import numpy as np

PYTHON = sys.executable

OUT_ROOT = Path(Path(__file__).resolve().parent, 'dataframes')

# Dataset generation
GENERATION = Path(Path(__file__).resolve().parent, "dataset_generation.py")

# Clustering
CLUSTERING = Path(Path(__file__).resolve().parent, "clustering.py")

# Fuzziness-stability statistics computation
STATISTICS = Path(Path(__file__).resolve().parent, "statistics.py")

# Analisys
ANALISYS = Path(Path(__file__).resolve().parent, "confrontation.py")

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

    run([PYTHON, str(GENERATION)])

    for cls_dir in sorted([p for p in (OUT_ROOT/'raw').iterdir() if p.is_dir()]):
        for csv_file in [p for p in cls_dir.iterdir() if p.suffix.lower() == '.csv']:
            run([PYTHON, str(CLUSTERING), "--inp_dir", str(csv_file), "--out_dir", str(OUT_ROOT/'clustering'/str(cls_dir.name))])
    rng = np.random.default_rng(420)
    for cls_dir in sorted([p for p in (OUT_ROOT/'clustering').iterdir() if p.is_dir()]):
        for m_value in sorted([p for p in cls_dir.iterdir() if p.is_dir()]):
            for csv_file in [p for p in m_value.iterdir() if p.suffix.lower() == '.csv']:
                run([PYTHON, str(STATISTICS), "--inp_dir", str(csv_file), "--out_dir", str(OUT_ROOT/'statistics'/str(cls_dir.name)/str(m_value.name)), "--seed", str(rng.integers(0, 1_000_000))])

if __name__ == "__main__":
    main()