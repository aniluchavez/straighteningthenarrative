"""
compute_surprisal_modulation_by_freq_decile.py
================================================
For each SUBTLEX population-frequency decile (1=rarest words ... 10=most
common), computes r(surprisal, curvature) two ways:

  - "Pooled": across all word TOKENS in the decile, mixing different word
    identities together (per-patient r, then patient-level t-test).
  - "Within-word": the same correlation computed on RESIDUALS after
    demeaning both surprisal and curvature within each distinct word type
    (pooled across patients), so a word's own occurrences are only ever
    compared against its own occurrences, never against a different word's
    baseline level. This removes a Simpson's-paradox-style confound where
    different word types have different average curvature levels that can
    make the pooled (between-word) correlation disagree in sign with the
    true within-word relationship.

...for BOTH semantic (LLM layer 18, raw embedding, no PCA) and neural
(hippocampal spikes) curvature.

IMPORTANT alignment note: trajectory_curvature(X) returns an array of
length len(X)-2, where element i is the curvature AT word index i+1 (the
vertex between steps i->i+1 and i+1->i+2), not word index i. This script
uses the "exit" alignment: word j's surprisal is paired with the curvature
AT word (j+1) -- i.e. the bend formed as the trajectory leaves word j and
arrives at word j+1, matching the lag+1 convention used in this project's
main event-triggered figures (the surprisal-locked curvature peak consistently
appears one word after the event, not at it).

Saves:
  results/surprisal_modulation_by_freq_decile.json
  figures/fig_surprisal_modulation_by_freq_decile.{pdf,png}

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u compute_surprisal_modulation_by_freq_decile.py
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp
from wordfreq import zipf_frequency

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(PARENT_DIR, "figures")
MODEL = "llama-2-7b"
LAYER = 18
REGION = "hippocampus"
N_DECILES = 10
MIN_WORD_N = 10  # minimum occurrences of a word type (pooled across patients) to include in within-word demeaning

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import fit_neural_pca, trajectory_curvature
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

_zipf_cache = {}
def subtlex_log_freq(words):
    out = np.empty(len(words))
    for i, w in enumerate(words):
        wl = str(w).lower()
        if wl not in _zipf_cache:
            _zipf_cache[wl] = zipf_frequency(wl, "en")
        out[i] = _zipf_cache[wl]
    return out


def main():
    per_patient = {}  # pid -> dict(word=, freq=, surp=, c_sem=, c_neu=) all aligned to word index 1..n-2
    all_freqs = []

    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data is None or data.get("neu_vecs") is None or data.get("emb_path") is None:
            print(f"  {pid}: SKIP")
            continue
        n = data["n_words"]
        surp_full = data["surp"][:n]
        words_full = data["words"][:n]
        freq_full = subtlex_log_freq(words_full)

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu_raw = trajectory_curvature(Y_neu)  # length n-2, index i -> word i+1

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if LAYER >= emb_mm.shape[0]:
            del emb_mm
            continue
        emb_L = np.array(emb_mm[LAYER, :n], dtype=np.float64)
        del emb_mm
        c_sem_raw = trajectory_curvature(emb_L)  # length n-2, index i -> word i+1

        m = len(c_sem_raw)  # = n-2; c_sem_raw[i] = curvature AT word (i+1)
        # "Exit" alignment: pair word j's surprisal with the bend AT word (j+1),
        # i.e. the curvature formed as the trajectory LEAVES word j and arrives
        # at word j+1 (one of that vertex's two constituent steps is exactly
        # word_j -> word_j+1). This is c_sem_raw[j] directly (no -1 shift),
        # valid for j = 1 .. m-1.
        word_idx = np.arange(1, m)
        per_patient[pid] = dict(
            word=words_full[word_idx],
            freq=freq_full[word_idx],
            surp=surp_full[word_idx],
            c_sem=c_sem_raw[word_idx],
            c_neu=c_neu_raw[word_idx],
        )
        all_freqs.append(freq_full[word_idx])
        print(f"  {pid}: n_aligned={m}  done", flush=True)

    all_freqs = np.concatenate(all_freqs)
    finite = np.isfinite(all_freqs)
    decile_edges = np.percentile(all_freqs[finite], np.linspace(0, 100, N_DECILES + 1))
    print(f"\nGlobal decile edges (zipf freq): {np.round(decile_edges, 2)}")

    results = {"decile": [], "freq_lo": [], "freq_hi": [],
               "r_sem_mean": [], "r_sem_sem": [], "r_sem_t": [], "r_sem_p": [], "r_sem_n": [],
               "r_neu_mean": [], "r_neu_sem": [], "r_neu_t": [], "r_neu_p": [], "r_neu_n": [],
               "r_sem_within_word": [], "r_sem_within_word_p": [], "r_sem_within_word_n": [],
               "r_neu_within_word": [], "r_neu_within_word_p": [], "r_neu_within_word_n": []}

    for d in range(N_DECILES):
        lo, hi = decile_edges[d], decile_edges[d + 1]

        # ---- pooled (naive, between-word-confounded) per-patient r, t-test ----
        r_sem_list, r_neu_list = [], []
        n_words_total = 0
        for pid, dd in per_patient.items():
            freq, surp, c_sem, c_neu = dd["freq"], dd["surp"], dd["c_sem"], dd["c_neu"]
            if d == N_DECILES - 1:
                in_bin = np.isfinite(freq) & (freq >= lo) & (freq <= hi)
            else:
                in_bin = np.isfinite(freq) & (freq >= lo) & (freq < hi)
            fin = in_bin & np.isfinite(surp) & np.isfinite(c_sem) & np.isfinite(c_neu)
            if fin.sum() < 20:
                continue
            r_s, _ = pearsonr(surp[fin], c_sem[fin])
            r_n, _ = pearsonr(surp[fin], c_neu[fin])
            r_sem_list.append(r_s)
            r_neu_list.append(r_n)
            n_words_total += fin.sum()

        r_sem_arr = np.array(r_sem_list)
        r_neu_arr = np.array(r_neu_list)
        t_s, p_s = ttest_1samp(r_sem_arr, 0.0) if len(r_sem_arr) >= 2 else (np.nan, np.nan)
        t_n, p_n = ttest_1samp(r_neu_arr, 0.0) if len(r_neu_arr) >= 2 else (np.nan, np.nan)

        results["decile"].append(d + 1)
        results["freq_lo"].append(float(lo)); results["freq_hi"].append(float(hi))
        results["r_sem_mean"].append(float(r_sem_arr.mean())); results["r_sem_sem"].append(float(r_sem_arr.std(ddof=1) / np.sqrt(len(r_sem_arr))))
        results["r_sem_t"].append(float(t_s)); results["r_sem_p"].append(float(p_s)); results["r_sem_n"].append(int(n_words_total))
        results["r_neu_mean"].append(float(r_neu_arr.mean())); results["r_neu_sem"].append(float(r_neu_arr.std(ddof=1) / np.sqrt(len(r_neu_arr))))
        results["r_neu_t"].append(float(t_n)); results["r_neu_p"].append(float(p_n)); results["r_neu_n"].append(int(n_words_total))

        # ---- within-word (word-identity-demeaned) pooled correlation ----
        by_word_surp = defaultdict(list)
        by_word_csem = defaultdict(list)
        by_word_cneu = defaultdict(list)
        for pid, dd in per_patient.items():
            freq, surp, word, c_sem, c_neu = dd["freq"], dd["surp"], dd["word"], dd["c_sem"], dd["c_neu"]
            if d == N_DECILES - 1:
                in_bin = np.isfinite(freq) & (freq >= lo) & (freq <= hi)
            else:
                in_bin = np.isfinite(freq) & (freq >= lo) & (freq < hi)
            fin = in_bin & np.isfinite(surp) & np.isfinite(c_sem) & np.isfinite(c_neu)
            for j in np.where(fin)[0]:
                w = word[j]
                by_word_surp[w].append(surp[j])
                by_word_csem[w].append(c_sem[j])
                by_word_cneu[w].append(c_neu[j])

        resid_surp, resid_csem, resid_cneu = [], [], []
        for w in by_word_surp:
            s = np.array(by_word_surp[w])
            if len(s) < MIN_WORD_N:
                continue
            cs = np.array(by_word_csem[w])
            cn = np.array(by_word_cneu[w])
            resid_surp.append(s - s.mean())
            resid_csem.append(cs - cs.mean())
            resid_cneu.append(cn - cn.mean())

        if resid_surp:
            resid_surp = np.concatenate(resid_surp)
            resid_csem = np.concatenate(resid_csem)
            resid_cneu = np.concatenate(resid_cneu)
            r_sw, p_sw = pearsonr(resid_surp, resid_csem) if len(resid_surp) >= 20 else (np.nan, np.nan)
            r_nw, p_nw = pearsonr(resid_surp, resid_cneu) if len(resid_surp) >= 20 else (np.nan, np.nan)
            n_w = len(resid_surp)
        else:
            r_sw = p_sw = r_nw = p_nw = np.nan
            n_w = 0

        results["r_sem_within_word"].append(float(r_sw)); results["r_sem_within_word_p"].append(float(p_sw)); results["r_sem_within_word_n"].append(int(n_w))
        results["r_neu_within_word"].append(float(r_nw)); results["r_neu_within_word_p"].append(float(p_nw)); results["r_neu_within_word_n"].append(int(n_w))

        print(f"  decile {d+1:2d} [zipf {lo:.2f}-{hi:.2f}]  n_words={n_words_total:6d}  "
              f"POOLED r_sem={r_sem_arr.mean():+.4f} (p={p_s:.3g})  r_neu={r_neu_arr.mean():+.4f} (p={p_n:.3g})   |   "
              f"WITHIN-WORD (n={n_w}) r_sem={r_sw:+.4f} (p={p_sw:.3g})  r_neu={r_nw:+.4f} (p={p_nw:.3g})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "surprisal_modulation_by_freq_decile.json")
    with open(out_json, "w") as f:
        json.dump(results, f)
    print(f"\nSaved -> {out_json}")

    # ── Figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    x = np.array(results["decile"])
    C_SEM, C_NEU = "#8E44AD", "#148F77"

    ax = axes[0]
    ax.errorbar(x - 0.08, results["r_sem_mean"], yerr=results["r_sem_sem"], fmt="-o",
                color=C_SEM, capsize=4, label="Semantic")
    ax.errorbar(x + 0.08, results["r_neu_mean"], yerr=results["r_neu_sem"], fmt="-s",
                color=C_NEU, capsize=4, label="Neural")
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xlabel("SUBTLEX frequency decile (1=rarest -> 10=most common)")
    ax.set_ylabel("r(surprisal, curvature)")
    ax.set_title("A  Pooled across word types\n(mixes different words' baseline curvature levels)",
                  loc="left", fontsize=10.5, pad=8)
    ax.legend(loc="best", fontsize=9, frameon=False)

    ax = axes[1]
    ax.plot(x, results["r_sem_within_word"], "-o", color=C_SEM, label="Semantic")
    ax.plot(x, results["r_neu_within_word"], "-s", color=C_NEU, label="Neural")
    for i, d in enumerate(x):
        if results["r_sem_within_word_p"][i] < 0.05:
            ax.text(d, results["r_sem_within_word"][i] + 0.01, "*", color=C_SEM, ha="center", fontsize=14)
        if results["r_neu_within_word_p"][i] < 0.05:
            ax.text(d, results["r_neu_within_word"][i] - 0.015, "*", color=C_NEU, ha="center", fontsize=14)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xlabel("SUBTLEX frequency decile (1=rarest -> 10=most common)")
    ax.set_ylabel("r(surprisal, curvature), within-word residuals")
    ax.set_title("B  Within-word (word identity controlled)\neach word type compared only to its own occurrences",
                  loc="left", fontsize=10.5, pad=8)
    ax.legend(loc="best", fontsize=9, frameon=False)

    fig.suptitle(f"Surprisal-curvature coupling by word frequency  ·  podcast, {MODEL} layer {LAYER}, "
                 f"hippocampus, n=13 patients", y=1.02, fontsize=11)
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"fig_surprisal_modulation_by_freq_decile.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
