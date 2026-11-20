"""
make_event_triggered_curvature_residual_ratecontrol.py
==========================================================
Extends make_event_triggered_curvature_residual.py: Panel C additionally
controls for raw total firing rate (summed spike count across all neurons
in the word's window, BEFORE PCA reduction) alongside gain (PCA-space step
magnitude, d_neu) and direction. Tests whether the curvature-surprisal
relationship survives controlling for raw population spike-count magnitude
specifically, not just the PCA-projected step size (which could differ if
firing-rate changes happen along dimensions PCA discards).

Three panels: A) raw curvature, B) residual after gain+direction (same as
the original figure), C) residual after gain+direction+raw_rate.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

from run_geometry_analysis import PATIENTS, load_patient, fit_neural_pca, trajectory_curvature

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES  = os.path.join(BASE_DIR, "figures")
REGION   = "hippocampus"
MODEL    = "llama-3.1-8b"
WINDOW   = 5
HI_PCTILE = 75
LO_PCTILE = 25

C_HI = "#C0392B"
C_LO = "#2980B9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi":  150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


def unit_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n < 1e-12] = np.nan
    return V / n


def ols_residual(y, X):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    resid = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < X.shape[1] + 3:
        return resid
    Xm = np.column_stack([np.ones(mask.sum()), X[mask]])
    b, _, _, _ = np.linalg.lstsq(Xm, y[mask], rcond=None)
    resid[mask] = y[mask] - Xm @ b
    return resid


def build_patient_curvature_table(pid, preg):
    data = load_patient(pid, preg, REGION, MODEL)
    if data is None or data["neu_vecs"] is None:
        return None
    n = data["n_words"]
    Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None

    c_neu = trajectory_curvature(Y_neu)
    steps = Y_neu[1:] - Y_neu[:-1]
    if len(steps) < 4:
        return None
    d_neu = np.linalg.norm(steps[1:], axis=1)

    unit_steps = unit_rows(steps)
    mu = np.nanmean(unit_steps, axis=0)
    mu_norm = np.linalg.norm(mu)
    if mu_norm < 1e-12:
        return None
    mu_hat = mu / mu_norm
    direction_align = unit_steps[1:] @ mu_hat

    # raw total spike count (pre-PCA), summed across all neurons, at each word's window
    raw_rate_full = data["neu_vecs"][:n].sum(axis=1)   # len n

    m = min(len(c_neu), len(d_neu), len(direction_align))
    c_neu, d_neu, direction_align = c_neu[:m], d_neu[:m], direction_align[:m]
    center_word_index = np.arange(2, 2 + m)
    raw_rate = raw_rate_full[center_word_index] if center_word_index.max() < n else np.full(m, np.nan)

    resid_gaindir = ols_residual(c_neu, np.column_stack([d_neu, direction_align]))
    resid_all = ols_residual(c_neu, np.column_stack([d_neu, direction_align, raw_rate]))

    surp = data["surp"]
    return dict(center_word_index=center_word_index,
                c_neu_raw=c_neu, c_neu_resid_gaindir=resid_gaindir, c_neu_resid_all=resid_all,
                surprisal=surp[center_word_index] if center_word_index.max() < n else None,
                n_words=n, surp_full=surp)


def event_triggered_patient(center_idx, values, surp_full, n_words):
    lut = dict(zip(center_idx, values))
    surp_at_center = surp_full[center_idx]
    finite = np.isfinite(surp_at_center)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(surp_at_center[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(surp_at_center[finite], LO_PCTILE)

    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    for j, sj in zip(center_idx, surp_at_center):
        if not np.isfinite(sj):
            continue
        is_hi, is_lo = sj >= hi_thr, sj <= lo_thr
        if not (is_hi or is_lo):
            continue
        row = [lut.get(j + k, np.nan) for k in lags]
        if np.sum(np.isfinite(row)) < WINDOW:
            continue
        (hi_rows if is_hi else lo_rows).append(row)
    if len(hi_rows) < 3 or len(lo_rows) < 3:
        return None, None
    return np.array(hi_rows), np.array(lo_rows)


def collect_all(value_key):
    pat_hi, pat_lo = [], []
    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        tbl = build_patient_curvature_table(pid, preg)
        if tbl is None or tbl["surprisal"] is None:
            continue
        hi_rows, lo_rows = event_triggered_patient(
            tbl["center_word_index"], tbl[value_key], tbl["surp_full"], tbl["n_words"])
        if hi_rows is None:
            continue
        pat_hi.append(np.nanmean(hi_rows, axis=0))
        pat_lo.append(np.nanmean(lo_rows, axis=0))
    return np.array(pat_hi), np.array(pat_lo)


def _panel(ax, hi, lo, title, ylabel):
    lags = np.arange(-WINDOW, WINDOW + 1)
    n_pat = hi.shape[0]

    hi_mu, hi_se = np.nanmean(hi, axis=0), scipy_sem(hi, axis=0, nan_policy="omit")
    lo_mu, lo_se = np.nanmean(lo, axis=0), scipy_sem(lo, axis=0, nan_policy="omit")

    ax.plot(lags, hi_mu, color=C_HI, lw=1.6, label="High surprisal (top 25%)")
    ax.fill_between(lags, hi_mu - hi_se, hi_mu + hi_se, color=C_HI, alpha=0.18)
    ax.plot(lags, lo_mu, color=C_LO, lw=1.6, label="Low surprisal (bottom 25%)")
    ax.fill_between(lags, lo_mu - lo_se, lo_mu + lo_se, color=C_LO, alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Words relative to event\nlag 0 ends at event word", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}\n$n$ = {n_pat} patients", loc="left", fontsize=10.5, pad=3)
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.75, edgecolor="#DDDDDD")
    ax.set_xticks(lags)


def main():
    hi_raw, lo_raw = collect_all("c_neu_raw")
    hi_gd, lo_gd = collect_all("c_neu_resid_gaindir")
    hi_all, lo_all = collect_all("c_neu_resid_all")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.23, top=0.76, wspace=0.45)

    _panel(axes[0], hi_raw, lo_raw, "A  Raw curvature (no control)", "Mean curvature (°)")
    _panel(axes[1], hi_gd, lo_gd, "B  Residual (gain + direction)", "Mean curvature residual (°)")
    _panel(axes[2], hi_all, lo_all, "C  Residual (gain + direction\n+ raw firing-rate magnitude)",
           "Mean curvature residual (°)")

    fig.suptitle(
        "llama-3.1-8b  ·  event-triggered neural curvature, hippocampus  ·  adding raw firing-rate control",
        fontsize=9, fontweight="bold", y=0.97)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig3_event_triggered_curvature_residual_ratecontrol"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)

    print(f"\nPanel C peak (lag 1): hi={np.nanmean(hi_all, axis=0)[WINDOW+1]:.3f}  lo={np.nanmean(lo_all, axis=0)[WINDOW+1]:.3f}")


if __name__ == "__main__":
    main()
