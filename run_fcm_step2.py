from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np
import pandas as pd

from preprocess import preprocess_dataframe, _islands_filter
from fcm import FuzzyCMeans
from metrics import fuzzy_diagnostics, _silhouette, _xie_beni_index

# -------------------- USER SETTINGS --------------------
CSV_PATH = "data/df_complete.csv"
OUT_ROOT = "fcm_step2"

ID_COLS = ["ALL_Uniq"]

AREA_VALUES = [3, 100000]
POP_VALUES = [200, 1000000]

# Resource-only feature set for Step 2
RESOURCE_FEATURES = [
    "solar_power",
    "solar_seas_ind",
    "wind_power",
    "wind_std",
    "offshore_wind_potential",
    "hydro_potential",
    "geothermal_potential",
    "evi",
    "res_area",
]
# ------------------------------------------------------

def choose_hyperparams(cons_class: str) -> Tuple[int, float]:
    hype_dict = {
        "XS" : (3, 2.4),
        "S" : (3, 2.4),
        "M" : (4, 2.1),
        "L" : (2, 2.1)
    }
    return hype_dict[cons_class]

def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = _islands_filter(df, POP_VALUES, AREA_VALUES)

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # Keep only resource features that exist
    feat_cols_present = [c for c in RESOURCE_FEATURES if c in df.columns]
    if not feat_cols_present:
        raise ValueError(
            "None of RESOURCE_FEATURES exist in the dataframe. "
            "Check column names vs RESOURCE_FEATURES."
        )

    print(f"Using resource features: {feat_cols_present}")
    
    # Helpful sanity check
    print("Consumption class counts:")
    print(df["ec_labels"].value_counts(dropna=False).to_string())
    print("")

    for cls in ["XS", "S", "M", "L"]:
        df_cls = df[df["ec_labels"] == cls].copy()

        if df_cls.shape[0] < 30:
            print(f"[{cls}] skipped (too few rows: {df_cls.shape[0]})")
            continue

        X_in = df_cls[ID_COLS + feat_cols_present].copy()

        # Preprocess features only
        prep = preprocess_dataframe(
            X_in,
            drop_non_numeric=True,
            id_columns= ID_COLS,
            missing="median",
            scale="robust",
        )

        # Hyperparams
        n_clusters, m = choose_hyperparams(cls)

        model = FuzzyCMeans(
            n_clusters=n_clusters,
            m=m,
            max_iter=600,
            tol=1e-5,
            seed=42,
        )
        res = model.fit(prep.X)
        labels = model.hard_labels(res.memberships)
        diag = fuzzy_diagnostics(res.memberships, ambiguous_threshold=0.6)

        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save centers (scaled-space centers)
        centers_df = pd.DataFrame(res.centers, columns=prep.fin_feat_names)
        centers_df.to_csv(out_dir / "centers.csv", index=False)

        # Add raw (unscaled) resource feature values for interpretability
        out_df = prep.df_kept_raw.copy()
        for k in range(n_clusters):
            out_df[f"mu_{k}"] = res.memberships[:, k]
        out_df["cluster_hard"] = labels
        out_df.to_csv(out_dir / "assignments_memberships_raw.csv", index=False)

        out_df = prep.df_kept_norm.copy()
        for k in range(n_clusters):
            out_df[f"mu_{k}"] = res.memberships[:, k]
        out_df["cluster_hard"] = labels
        out_df.to_csv(out_dir / "assignments_memberships_norm.csv", index=False)

        sil = _silhouette(out_df, feat_cols_present, "cluster_hard", m, "mu")
        xie = _xie_beni_index(out_df, feat_cols_present, "cluster_hard", m, "mu")

        summary = {
            "class": cls,
            "n_rows_total_in_class": int(df_cls.shape[0]),
            "n_rows_used_after_preprocess": int(out_df.shape[0]),
            "features_used": RESOURCE_FEATURES,
            "n_clusters": int(n_clusters),
            "m": float(m),
            "n_iter": int(res.n_iter),
            "converged": bool(res.converged),
            "final_objective": float(res.objective_history[-1]),
            "fuzzy_diagnostics": asdict(diag),
            "cluster_sizes_hard": {str(k): int(np.sum(labels == k)) for k in range(n_clusters)},
            "silhouette_hard" : sil[0],
            "silhouette_fuzzy" : sil[1],
            "Xie-beni-hard" : xie[0],
            "Xie-beni" : xie[1]
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print(
            f"[{cls}] done -> {out_dir} | "
            f"n_used={out_df.shape[0]} | k={n_clusters} m={m} | "
            f"sizes={summary['cluster_sizes_hard']}"
        )

    print("\nAll step-2 class runs completed.")

if __name__ == "__main__":
    main()