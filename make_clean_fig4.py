"""
make_clean_fig4.py
==================
Cleaner Figure 4 from the curvature-vs-displacement analysis tables.

Panels:
  A. Cross-space geometry alignment: displacement vs curvature.
  B. Controlled neural curvature model: semantic displacement vs curvature.
  C. Semantic-space surprisal model: semantic displacement vs curvature.
  D. Optional neural-space surprisal model: neural displacement vs curvature.

Example:
    python make_clean_fig4.py --lag 0
    python make_clean_fig4.py --lag 1
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import sem as scipy_sem
from scipy.stats import ttest_rel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results")
FIGURES = os.path.join(BASE_DIR, "figures")
PRIMARY_MODEL = "llama-3.1-8b"
REGIONS = ["hippocampus"]

REGION_LABELS = {
    "hippocampus": "Hippocampus",
}

C_CURV = "#D95319"
C_DISP = "#5E7CE2"
C_PT = "#333333"
C_ZERO = "#777777"


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def _mean_sem(vals: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    se = float(scipy_sem(arr)) if len(arr) > 1 else np.nan
    return float(np.mean(arr)), se


def _pstar(pval: float) -> str:
    if not np.isfinite(pval):
        return "n.s."
    if pval < 1e-3:
        return "***"
    if pval < 1e-2:
        return "**"
    if pval < 5e-2:
        return "*"
    return "n.s."


def _load_patient_summary(model: str, lag: int) -> pd.DataFrame:
    path = os.path.join(
        RESULTS,
        "clean_fig4",
        f"clean_fig4_{model}_lag{lag}_patient_summary.csv",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run run_clean_fig4_tests.py --model {model} "
            f"--surprisal_lag {lag} first."
        )
    df = pd.read_csv(path)
    return df[df["region"].isin(REGIONS)].copy()


def _paired_panel(
    ax,
    df: pd.DataFrame,
    keys: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
    ylabel: str,
    *,
    zero_line: bool = True,
    delta_order: tuple[int, int] = (1, 0),
) -> None:
    width = 0.26
    offsets = np.linspace(-width / 1.4, width / 1.4, len(keys))
    x_base = np.arange(len(REGIONS), dtype=float)
    all_vals = []

    for xi, region in zip(x_base, REGIONS):
        sub = df[df["region"] == region].copy()
        if sub.empty:
            continue
        for ki, (key, color) in enumerate(zip(keys, colors)):
            vals = pd.to_numeric(sub[key], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            all_vals.extend(vals.tolist())
            mu, se = _mean_sem(vals)
            xpos = xi + offsets[ki]
            ax.bar(
                xpos,
                mu,
                width=width,
                yerr=se,
                color=color,
                alpha=0.86,
                edgecolor="white",
                linewidth=0.6,
                capsize=2.5,
                label=labels[ki] if xi == 0 else "_nolegend_",
                zorder=2,
            )

        pair = sub[list(keys)].apply(pd.to_numeric, errors="coerce").dropna()
        for _, row in pair.iterrows():
            ys = row.to_numpy(dtype=float)
            xs = xi + offsets
            ax.plot(xs, ys, color=C_PT, alpha=0.20, lw=0.6, zorder=3)
            ax.scatter(
                xs,
                ys,
                s=12,
                facecolor="white",
                edgecolor=C_PT,
                linewidth=0.45,
                alpha=0.72,
                zorder=4,
            )

        if len(pair) > 1:
            hi, lo = delta_order
            diff = pair.iloc[:, hi].to_numpy() - pair.iloc[:, lo].to_numpy()
            _, pval = ttest_rel(pair.iloc[:, hi], pair.iloc[:, lo], nan_policy="omit")
            vals = pair.to_numpy(dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if len(vals):
                y = np.nanmax(vals)
                yrange = max(np.nanmax(all_vals) - np.nanmin(all_vals), 0.02)
                ax.text(
                    xi,
                    y + 0.11 * yrange,
                    f"Δ={np.mean(diff):.3f} {_pstar(float(pval))}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#222222",
                )

    if zero_line:
        ax.axhline(0, color=C_ZERO, lw=0.7, ls="--", alpha=0.7, zorder=1)

    ax.set_xticks(x_base)
    ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS])
    ax.set_title(title, loc="left", pad=5)
    ax.set_ylabel(ylabel)
    vals = np.asarray(all_vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
        lo = min(0.0 if zero_line else np.nanmin(vals), np.nanmin(vals))
        hi = np.nanmax(vals)
        pad = max((hi - lo) * 0.28, 0.015)
        ax.set_ylim(lo - pad * 0.35, hi + pad)


def make_figure(model: str, lag: int, output_stem: str | None = None) -> None:
    df = _load_patient_summary(model, lag)
    os.makedirs(FIGURES, exist_ok=True)

    lag_text = "surprisal$_t$" if lag == 0 else f"surprisal$_{{t+{lag}}}$"
    lag_label = "event word" if lag == 0 else f"+{lag} word"

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.15))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.80,
                        hspace=0.52, wspace=0.38)

    _paired_panel(
        axes[0, 0],
        df,
        keys=["r_dsem_dneu", "r_csem_cneu"],
        labels=["Displacement", "Curvature"],
        colors=[C_DISP, C_CURV],
        title="A  Cross-space geometry alignment",
        ylabel="Semantic-neural Pearson $r$",
        zero_line=True,
        delta_order=(1, 0),
    )

    _paired_panel(
        axes[0, 1],
        df,
        keys=["beta_dsem_to_cneu", "beta_csem_to_cneu"],
        labels=["$d_{sem}$", "$c_{sem}$"],
        colors=[C_DISP, C_CURV],
        title="B  Controlled neural curvature model",
        ylabel="Std. beta predicting $c_{neu}$",
        zero_line=True,
        delta_order=(1, 0),
    )

    _paired_panel(
        axes[1, 0],
        df,
        keys=["beta_dsem_to_surp", "beta_csem_to_surp"],
        labels=["$d_{sem}$", "$c_{sem}$"],
        colors=[C_DISP, C_CURV],
        title=f"C  Semantic geometry → {lag_text}",
        ylabel="Std. beta predicting surprisal",
        zero_line=True,
        delta_order=(1, 0),
    )

    _paired_panel(
        axes[1, 1],
        df,
        keys=["beta_dneu_to_surp", "beta_cneu_to_surp"],
        labels=["$d_{neu}$", "$c_{neu}$"],
        colors=[C_DISP, C_CURV],
        title=f"D  Neural geometry → {lag_text}",
        ylabel="Std. beta predicting surprisal",
        zero_line=True,
        delta_order=(1, 0),
    )

    fig.legend(
        handles=[
            Patch(facecolor=C_DISP, edgecolor="white", label="Displacement / speed"),
            Patch(facecolor=C_CURV, edgecolor="white", label="Curvature / direction change"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.900),
        ncol=2,
        columnspacing=1.8,
        handlelength=1.6,
    )

    fig.suptitle(
        f"{model}  ·  Figure 4 clean regression summary ({lag_label})",
        fontsize=10,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.045,
        "Bars show patient mean ± SEM; points/lines show paired patient values. "
        "Beta panels use standardized OLS. Panel B controls surprisal, sentence position/start, "
        "turn boundary, speaker role, word length, log frequency, and distance to turn; "
        "Panels C-D use the same controls except surprisal.",
        ha="center",
        va="center",
        fontsize=6.4,
    )

    stem = output_stem or f"fig4_clean_lag{lag}"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        if ext == "svg":
            txt = open(path).read()
            txt = txt.replace("TeX Gyre Termes", "Times New Roman")
            open(path, "w").write(txt)
        print(f"  -> {path}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Make the cleaner Figure 4 from clean_fig4 patient summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=PRIMARY_MODEL)
    p.add_argument("--lag", type=int, default=0,
                   help="Surprisal lag used by run_clean_fig4_tests.py.")
    p.add_argument("--output_stem", default=None,
                   help="Optional output filename stem in the figures directory.")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    make_figure(args.model, args.lag, args.output_stem)


if __name__ == "__main__":
    main()
