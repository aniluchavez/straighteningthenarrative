"""
make_freq_surp_timescale_summary.py
=====================================
Summary figure + table for the frequency-vs-surprisal "timescale" finding:
frequency's unique contribution to curvature dominates at the single-word
level and decays under averaging, while surprisal's unique contribution is
flat-to-growing under the same averaging -- i.e. frequency is a local,
punctate lexical-access effect and surprisal is a longer-context,
integrative one. See compute_freq_surprisal_full_grid_test.py (chunk=1 vs
chunk=15 endpoints, patient-cluster bootstrap) for the original grid; this
script fills in the intermediate chunk sizes (no bootstrap, point estimates
only) to show the transition is smooth, not a step tied to one chunk size.

Saves:
  figures/fig_freq_surp_timescale_summary.{pdf,png,svg}
  tables/freq_surp_timescale_summary.csv

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u make_freq_surp_timescale_summary.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
TABLES = os.path.join(PARENT_DIR, "tables")
CHUNKS = [1, 2, 3, 5, 8, 10, 15, 20, 25]

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PARENT_DIR)
import compute_freq_surprisal_full_grid_test as G

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_FREQ, C_SURP = "#2980B9", "#C0392B"


def unique_r2_point(df, base_cols, add_col):
    y = df.curv_z.values
    Xa = np.column_stack([np.ones(len(df))] + [df[c].values for c in base_cols])
    Xb = np.column_stack([Xa, df[add_col].values])
    return G.r2_ols(y, Xb) - G.r2_ols(y, Xa)


def main():
    print("Loading patient data...")
    patient_data = {}
    for pid in G.PODCAST_PATIENTS:
        d = G.get_patient_base(pid)
        if d is None:
            print(f"  {pid}: SKIP")
            continue
        patient_data[pid] = d
        print(f"  {pid}: loaded", flush=True)

    rows = []
    for curv_key, curv_label in [("c_sem", "semantic"), ("c_neu", "neural")]:
        for chunk in CHUNKS:
            df = G.build_chunked_df(patient_data, curv_key, chunk)
            uf = unique_r2_point(df, ["surp_z", "dur_z", "pos_z"], "freq_z")
            us = unique_r2_point(df, ["freq_z", "dur_z", "pos_z"], "surp_z")
            rows.append(dict(curvature=curv_label, chunk=chunk, n_rows=len(df),
                              unique_r2_freq=uf, unique_r2_surp=us))
            print(f"  {curv_label} chunk={chunk:3d}: freq_R2={uf:+.5f}  surp_R2={us:+.5f}")

    df_out = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    csv_path = os.path.join(TABLES, "freq_surp_timescale_summary.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, curv_label, title in [
        (axes[0], "semantic", "A  Semantic curvature (gpt2-xl, layer 20)"),
        (axes[1], "neural", "B  Neural curvature (hippocampus spikes)"),
    ]:
        sub = df_out[df_out.curvature == curv_label].sort_values("chunk")
        ax.plot(sub.chunk, sub.unique_r2_freq, "o-", color=C_FREQ, lw=1.8, ms=4, label="Frequency (unique $R^2$)")
        ax.plot(sub.chunk, sub.unique_r2_surp, "o-", color=C_SURP, lw=1.8, ms=4, label="Surprisal (unique $R^2$)")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel("Averaging window (words)")
        ax.set_ylabel("Unique $R^2$ (beyond dur, pos, other predictor)")
        ax.set_title(title, loc="left", fontsize=10, pad=3)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Frequency's unique contribution is local (word-level); surprisal's is longer-context / integrative\n"
                 "Podcast, all 13 patients, hippocampus",
                 fontsize=9, fontweight="bold", y=1.04)
    fig.tight_layout()

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_freq_surp_timescale_summary"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
