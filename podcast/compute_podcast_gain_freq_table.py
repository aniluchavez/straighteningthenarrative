"""
compute_podcast_gain_freq_table.py
====================================
Podcast-native table (all 13 patients, LLaMA-2-7B layer 18, hippocampus)
testing whether the surprisal -> neural curvature relationship survives:

  - Base: no control, all words.
  - Additive gain control: subtract the mean population spike vector within
    each surprisal-quintile bin before computing curvature (removes
    common-mode, surprisal-locked additive gain).
  - Multiplicative gain control: normalize each population spike vector to
    unit norm before computing curvature (removes uniform multiplicative
    gain; preserves direction/cosine-angle structure, which is what
    curvature actually measures).
  - Frequency control: restrict to words in the middle 10% of SUBTLEX
    population frequency (same restriction used in
    make_podcast_curvature_allwords_vs_midfreq.py), no gain transform.

Reuses additive_gain_control / multiplicative_gain_control from
run_gain_control_surprisal_split.py (originally written for convo; the
transform functions themselves are corpus-agnostic, only the data loading
differs).

Statistic: per-patient Pearson r(surprisal, neural curvature), exit-aligned
(word j's surprisal paired with curvature AT word j+1, matching the
project's lag+1 convention -- see compute_surprisal_modulation_by_freq_decile.py
for the concrete verification of this alignment), then a one-sample
t-test across the 13 patients per condition.

Saves:
  tables/podcast_gain_freq_table.csv
  tables/podcast_gain_freq_table.tex

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u compute_podcast_gain_freq_table.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp
from wordfreq import zipf_frequency

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
P6_DIR = os.path.dirname(PARENT_DIR)
TABLES_DIR = os.path.join(PARENT_DIR, "tables")
MODEL = "llama-2-7b"
LAYER = 18
REGION = "hippocampus"
MIDFREQ_LO_PCTILE, MIDFREQ_HI_PCTILE = 45, 55

sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, P6_DIR)
from run_geometry_analysis import fit_neural_pca, trajectory_curvature
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient
# Inlined from run_gain_control_surprisal_split.py rather than imported --
# that module has no `if __name__ == "__main__":` guard and executes a full
# convo analysis (overwriting a shared results file) as a side effect of
# being imported at all.
GAIN_BINS = 5

def additive_gain_control(neu_arr, surp_arr, n_bins=GAIN_BINS):
    """Subtract the mean population vector within each surprisal quintile bin."""
    out = neu_arr.copy()
    valid = np.isfinite(surp_arr)
    if valid.sum() < n_bins * 3:
        return out - neu_arr.mean(axis=0)
    edges = np.percentile(surp_arr[valid], np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.full(len(surp_arr), -1)
    bin_idx[valid] = np.clip(np.digitize(surp_arr[valid], edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            out[m] = neu_arr[m] - neu_arr[m].mean(axis=0)
    unk = bin_idx == -1
    if unk.sum() > 0:
        out[unk] = neu_arr[unk] - neu_arr.mean(axis=0)
    return out


def multiplicative_gain_control(neu_arr):
    """Normalize each population vector to unit norm."""
    norms = np.clip(np.linalg.norm(neu_arr, axis=1, keepdims=True), 1e-12, None)
    return neu_arr / norms

_zipf_cache = {}
def subtlex_log_freq(words):
    out = np.empty(len(words))
    for i, w in enumerate(words):
        wl = str(w).lower()
        if wl not in _zipf_cache:
            _zipf_cache[wl] = zipf_frequency(wl, "en")
        out[i] = _zipf_cache[wl]
    return out


def midfreq_mask(log_freq):
    finite = np.isfinite(log_freq)
    lo_edge = np.nanpercentile(log_freq[finite], MIDFREQ_LO_PCTILE)
    hi_edge = np.nanpercentile(log_freq[finite], MIDFREQ_HI_PCTILE)
    return finite & (log_freq >= lo_edge) & (log_freq <= hi_edge)


def exit_aligned_pair(surp, c_full, eligible):
    """word j's surprisal paired with curvature AT word j+1 (c_full[j], since
    c_full[i] = curvature at word i+1), for j = 1..len(c_full)-1."""
    m = len(c_full)
    j = np.arange(1, m)
    s = surp[j]
    c = c_full[j]
    e = eligible[j] if eligible is not None else np.ones(len(j), dtype=bool)
    fin = np.isfinite(s) & np.isfinite(c) & e
    return s[fin], c[fin]


def exit_aligned_triple(surp, c_full, freq):
    """Same as exit_aligned_pair but also returns frequency, for ALL words
    (no restriction) -- used for the partial (covariate) frequency control,
    which keeps full statistical power instead of discarding ~90% of the
    data the way a hard middle-decile restriction does."""
    m = len(c_full)
    j = np.arange(1, m)
    s, c, f = surp[j], c_full[j], freq[j]
    fin = np.isfinite(s) & np.isfinite(c) & np.isfinite(f)
    return s[fin], c[fin], f[fin]


def partial_beta_surprisal(surp, curv, freq):
    """Standardized OLS beta on surprisal, controlling for frequency:
    curv ~ surprisal + freq, both z-scored. Returns (beta, n)."""
    n = len(surp)
    if n < 20:
        return np.nan, n
    X = np.column_stack([surp, freq])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    y = (curv - curv.mean()) / (curv.std() + 1e-12)
    Xm = np.column_stack([np.ones(n), X])
    b, _, _, _ = np.linalg.lstsq(Xm, y, rcond=None)
    return float(b[1]), n  # b[1] = beta on surprisal (b[0]=intercept, b[2]=beta on freq)


def main():
    rows_base, rows_add, rows_mult, rows_freq, rows_freq_partial = [], [], [], [], []

    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data is None or data.get("neu_vecs") is None:
            print(f"  {pid}: SKIP")
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        words = data["words"][:n]
        neu_vecs = data["neu_vecs"]
        log_freq = subtlex_log_freq(words)
        mid_elig = midfreq_mask(log_freq)

        # ---- Base ----
        Y_neu, _ = fit_neural_pca(neu_vecs, n)
        if Y_neu is None:
            continue
        c_base = trajectory_curvature(Y_neu)
        s, c = exit_aligned_pair(surp, c_base, None)
        if len(s) >= 20:
            r, _ = pearsonr(s, c)
            rows_base.append((pid, r, len(s)))

        # ---- Additive gain control ----
        neu_add = additive_gain_control(neu_vecs, surp)
        Y_add, _ = fit_neural_pca(neu_add, n)
        if Y_add is not None:
            c_add = trajectory_curvature(Y_add)
            s, c = exit_aligned_pair(surp, c_add, None)
            if len(s) >= 20:
                r, _ = pearsonr(s, c)
                rows_add.append((pid, r, len(s)))

        # ---- Multiplicative gain control ----
        neu_mult = multiplicative_gain_control(neu_vecs)
        Y_mult, _ = fit_neural_pca(neu_mult, n)
        if Y_mult is not None:
            c_mult = trajectory_curvature(Y_mult)
            s, c = exit_aligned_pair(surp, c_mult, None)
            if len(s) >= 20:
                r, _ = pearsonr(s, c)
                rows_mult.append((pid, r, len(s)))

        # ---- Frequency control, hard restriction (mid-freq only, base neural signal) ----
        s, c = exit_aligned_pair(surp, c_base, mid_elig)
        if len(s) >= 20:
            r, _ = pearsonr(s, c)
            rows_freq.append((pid, r, len(s)))

        # ---- Frequency control, partial/covariate (ALL words, full power) ----
        s_all, c_all, f_all = exit_aligned_triple(surp, c_base, log_freq)
        beta, n_used = partial_beta_surprisal(s_all, c_all, f_all)
        if np.isfinite(beta):
            rows_freq_partial.append((pid, beta, n_used))

        print(f"  {pid}: n_words={n}  done", flush=True)

    def summarize(rows, label):
        rs = np.array([r[1] for r in rows])
        n_pat = len(rs)
        t, p = ttest_1samp(rs, 0.0) if n_pat >= 2 else (np.nan, np.nan)
        n_pos = int((rs > 0).sum())
        return dict(condition=label, n_patients=n_pat, mean_r=rs.mean(), sem_r=rs.std(ddof=1) / np.sqrt(n_pat),
                    t=t, p=p, n_positive=n_pos)

    summary_rows = [
        summarize(rows_base, "Base (no control)"),
        summarize(rows_add, "+ Additive gain control"),
        summarize(rows_mult, "+ Multiplicative gain control"),
    ]
    df = pd.DataFrame(summary_rows)
    df["mean_r"] = df["mean_r"].round(4)
    df["sem_r"] = df["sem_r"].round(4)
    df["t"] = df["t"].round(2)
    df["p"] = df["p"].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")

    print("\n" + "=" * 90)
    print("PODCAST -- surprisal -> neural curvature (exit-aligned, lag+1) under gain/frequency controls")
    print(f"({MODEL} for word alignment/frequency only; neural signal is model-independent), hippocampus")
    print("=" * 90)
    print(df.to_string(index=False))

    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, "podcast_gain_freq_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    tex_path = os.path.join(TABLES_DIR, "podcast_gain_freq_table.tex")
    with open(tex_path, "w") as f:
        f.write(r"\begin{tabular}{lrrrrr}" + "\n")
        f.write(r"  \toprule" + "\n")
        f.write(r"  Control & $N$ & mean $r$ & SEM & $t$ & $p$ \\" + "\n")
        f.write(r"  \midrule" + "\n")
        for _, row in df.iterrows():
            f.write(f"  {row['condition']} & {row['n_patients']} & {row['mean_r']:+.4f} & "
                    f"{row['sem_r']:.4f} & {row['t']:.2f} & {row['p']} \\\\\n")
        f.write(r"  \bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Saved -> {tex_path}")


if __name__ == "__main__":
    main()
