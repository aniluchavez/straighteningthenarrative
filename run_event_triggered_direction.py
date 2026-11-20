"""
run_event_triggered_direction.py
=================================
Directional companion to the event-triggered magnitude/curvature analyses:
tests whether the *direction* of neural population movement (not just its
size) differs between high- and low-surprisal words.

At each lag relative to the event word, for each patient:
  1. Take the neural PCA-trajectory step vectors (Y_neu[t] - Y_neu[t-1])
     for words flagged high-surprisal vs low-surprisal (top/bottom 25%).
  2. Unit-normalize each step vector (direction only, magnitude discarded).
  3. Compute the mean resultant direction for each condition.
  4. angle_between = angle (deg) between the two conditions' mean directions.
  5. Permutation null: shuffle the hi/lo condition label within patient
     (keeping which lag-window each event contributes), recompute the angle,
     repeat N_PERM times -> p-value for whether the observed angle exceeds
     chance separation.

Works in each patient's own neural-PCA space (dimensionality varies by
patient, ~90% variance threshold) -- comparisons are done per-patient, then
aggregated across patients, so cross-patient dimensionality differences
don't matter.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

from run_geometry_analysis import (
    PATIENTS, load_patient, fit_neural_pca, surprisal_quantile_event_masks,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES  = os.path.join(BASE_DIR, "figures")
REGION   = "hippocampus"
MODEL    = "llama-3.1-8b"
WINDOW   = 5
N_PERM   = 2000
SEED     = 0

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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def angle_between_means(hi_steps, lo_steps):
    hi_u = [unit(v) for v in hi_steps if unit(v) is not None]
    lo_u = [unit(v) for v in lo_steps if unit(v) is not None]
    if len(hi_u) < 2 or len(lo_u) < 2:
        return np.nan
    mu_hi = np.mean(hi_u, axis=0)
    mu_lo = np.mean(lo_u, axis=0)
    n_hi, n_lo = np.linalg.norm(mu_hi), np.linalg.norm(mu_lo)
    if n_hi < 1e-12 or n_lo < 1e-12:
        return np.nan
    cos = np.clip(np.dot(mu_hi, mu_lo) / (n_hi * n_lo), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def patient_lag_steps(data, Y_neu):
    """dict: lag -> {'hi': [...step vecs...], 'lo': [...step vecs...]}"""
    n = data["n_words"]
    surp = data["surp"]
    hi_mask, lo_mask, _, _ = surprisal_quantile_event_masks(surp)
    steps = Y_neu[1:] - Y_neu[:-1]  # steps[k] = Y_neu[k+1]-Y_neu[k]

    lags = list(range(-WINDOW, WINDOW + 1))
    out = {lag: {"hi": [], "lo": []} for lag in lags}
    for j in range(n):
        is_hi, is_lo = hi_mask[j], lo_mask[j]
        if not (is_hi or is_lo):
            continue
        for lag in lags:
            k = (j + lag) - 1
            if 0 <= k < len(steps):
                (out[lag]["hi"] if is_hi else out[lag]["lo"]).append(steps[k])
    return out


def main():
    lags = list(range(-WINDOW, WINDOW + 1))
    rng = np.random.default_rng(SEED)

    # ---- load everyone once, cache per-patient per-lag step vectors ----
    per_patient_lagsteps = {}
    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, MODEL)
        if data is None or data["neu_vecs"] is None:
            print(f"  {pid}: SKIP -- no neural data")
            continue
        Y_neu, n_pcs = fit_neural_pca(data["neu_vecs"], data["n_words"])
        if Y_neu is None:
            print(f"  {pid}: SKIP -- neural PCA failed")
            continue
        per_patient_lagsteps[pid] = (patient_lag_steps(data, Y_neu), n_pcs)

    pids_used = list(per_patient_lagsteps.keys())
    n_pat = len(pids_used)
    print(f"n_patients used: {n_pat}")

    # ---- observed angles ----
    patient_angles = np.full((n_pat, len(lags)), np.nan)
    for pi, pid in enumerate(pids_used):
        lag_steps, n_pcs = per_patient_lagsteps[pid]
        for li, lag in enumerate(lags):
            hi, lo = lag_steps[lag]["hi"], lag_steps[lag]["lo"]
            patient_angles[pi, li] = angle_between_means(hi, lo)
        print(f"  {pid}: n_pcs={n_pcs}  lag0 angle={patient_angles[pi, WINDOW]:.2f} deg  "
              f"n_hi={len(lag_steps[0]['hi'])} n_lo={len(lag_steps[0]['lo'])}")

    # ---- permutation null (shuffle hi/lo labels within patient, per lag) ----
    print("\nPer-lag: mean angle across patients, permutation p-value")
    perm_p = np.full(len(lags), np.nan)
    for li, lag in enumerate(lags):
        null_per_patient = np.full((n_pat, N_PERM), np.nan)
        for pi, pid in enumerate(pids_used):
            lag_steps, _ = per_patient_lagsteps[pid]
            hi, lo = lag_steps[lag]["hi"], lag_steps[lag]["lo"]
            combined = hi + lo
            n_hi = len(hi)
            if len(combined) < 4:
                continue
            for pn in range(N_PERM):
                idx = rng.permutation(len(combined))
                hi_p = [combined[i] for i in idx[:n_hi]]
                lo_p = [combined[i] for i in idx[n_hi:]]
                null_per_patient[pi, pn] = angle_between_means(hi_p, lo_p)
        null_mean_across_pat = np.nanmean(null_per_patient, axis=0)
        obs_mean = np.nanmean(patient_angles[:, li])
        perm_p[li] = float((null_mean_across_pat >= obs_mean).mean())
        print(f"  lag {lag:+d}: mean_angle={obs_mean:.3f} deg  perm_p={perm_p[li]:.4f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mu = np.nanmean(patient_angles, axis=0)
    se = scipy_sem(patient_angles, axis=0, nan_policy="omit")
    ax.plot(lags, mu, color="#8E44AD", lw=1.8, marker="o", ms=4)
    ax.fill_between(lags, mu - se, mu + se, color="#8E44AD", alpha=0.18)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    for li, lag in enumerate(lags):
        if perm_p[li] < 0.05:
            ax.text(lag, mu[li] + se[li] + 1.5, "*", ha="center", fontsize=13, color="#8E44AD")
    ax.set_xlabel("Words relative to event\n(angle between high- vs low-surprisal mean step direction)")
    ax.set_ylabel("Angle between mean directions (°)")
    ax.set_title(f"Neural population direction (hippocampus)\n$n$ = {n_pat} patients", loc="left")
    ax.set_xticks(lags)

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig3_event_triggered_direction"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
