"""
debug_unrestricted_live_recompute.py
=====================================
Isolates whether the lag-0-vs-lag+1 discrepancy comes from the mid-frequency
restriction itself, or from some difference between the CACHED JSON pipeline
(behind fig_convo_podcast_curvature_comparison.png) and the LIVE RECOMPUTE
pipeline (behind the _surp_midfreq scripts) that has nothing to do with
frequency restriction at all.

Runs the exact same live-recompute event_windows()-based code as
make_convo_podcast_curvature_comparison_surp_midfreq.py's load_convo_surp_midfreq(),
but WITHOUT the mid-frequency mask -- i.e. splits directly on the full,
unrestricted surp array, same as the reference/cached pipeline is supposed to
represent. If this still peaks at lag 0, the restriction is NOT the cause and
there's a real pipeline bug. If it peaks at lag +1 (matching the reference
figure), the restriction really is what's shifting the peak.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PARENT_DIR, "results")
WINDOW      = 5
CONVO_MODEL = "llama-3.1-8b"
REGION      = "hippocampus"
HI_PCTILE = 75
LO_PCTILE = 25

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import (
    fit_neural_pca, trajectory_curvature, load_patient, PATIENTS, build_paths,
)


def event_windows(values, split_var, n_words):
    finite = np.isfinite(split_var)
    if finite.sum() < 20:
        return None, None
    hi_thr = np.nanpercentile(split_var[finite], HI_PCTILE)
    lo_thr = np.nanpercentile(split_var[finite], LO_PCTILE)
    lags = list(range(-WINDOW, WINDOW + 1))
    hi_rows, lo_rows = [], []
    m = len(values)
    for j in range(n_words):
        sj = split_var[j]
        if not np.isfinite(sj):
            continue
        is_hi, is_lo = sj >= hi_thr, sj <= lo_thr
        if not (is_hi or is_lo):
            continue
        row = [values[(j + k) - 2] if 0 <= (j + k) - 2 < m else np.nan for k in lags]
        if np.sum(np.isfinite(row)) < WINDOW:
            continue
        (hi_rows if is_hi else lo_rows).append(row)
    if len(hi_rows) < 3 or len(lo_rows) < 3:
        return None, None
    return np.array(hi_rows), np.array(lo_rows)


def main():
    conv_path = os.path.join(RESULTS_DIR, f"geometry_{CONVO_MODEL}_{REGION}.json")
    cached = {r["pid"]: r["loo_layer_neu"] for r in json.load(open(conv_path))}

    pat_hi_neu, pat_lo_neu = [], []
    pat_hi_sem, pat_lo_sem = [], []
    for pid, preg in PATIENTS.items():
        if pid not in cached or REGION not in preg:
            continue
        layer = cached[pid]
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu_full = trajectory_curvature(Y_neu)

        paths = build_paths(pid, CONVO_MODEL)
        emb_mm = np.load(paths["cache"], mmap_mode="r")
        if layer >= emb_mm.shape[0]:
            del emb_mm
            continue
        emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
        del emb_mm
        c_sem_full = trajectory_curvature(emb_opt)

        hi_n, lo_n = event_windows(c_neu_full, surp, n)
        hi_s, lo_s = event_windows(c_sem_full, surp, n)
        if hi_n is None or hi_s is None:
            continue
        pat_hi_neu.append(np.nanmean(hi_n, axis=0)); pat_lo_neu.append(np.nanmean(lo_n, axis=0))
        pat_hi_sem.append(np.nanmean(hi_s, axis=0)); pat_lo_sem.append(np.nanmean(lo_s, axis=0))
        print(f"  {pid}: layer={layer}  done", flush=True)

    hi_neu = np.nanmean(np.array(pat_hi_neu), axis=0)
    lo_neu = np.nanmean(np.array(pat_lo_neu), axis=0)
    hi_sem = np.nanmean(np.array(pat_hi_sem), axis=0)
    lo_sem = np.nanmean(np.array(pat_lo_sem), axis=0)
    lags = list(range(-WINDOW, WINDOW + 1))

    print("\n=== NEURAL (live recompute, UNRESTRICTED surprisal split) ===")
    for k, h, l in zip(lags, hi_neu, lo_neu):
        print(f"  lag {k:+d}: hi={h:7.3f}  lo={l:7.3f}")
    print(f"  hi peak at lag {lags[int(np.argmax(hi_neu))]:+d} (value {np.max(hi_neu):.3f})")
    print(f"  lo trough at lag {lags[int(np.argmin(lo_neu))]:+d} (value {np.min(lo_neu):.3f})")

    print("\n=== SEMANTIC (live recompute, UNRESTRICTED surprisal split) ===")
    for k, h, l in zip(lags, hi_sem, lo_sem):
        print(f"  lag {k:+d}: hi={h:7.3f}  lo={l:7.3f}")
    print(f"  hi peak at lag {lags[int(np.argmax(hi_sem))]:+d} (value {np.max(hi_sem):.3f})")
    print(f"  lo trough at lag {lags[int(np.argmin(lo_sem))]:+d} (value {np.min(lo_sem):.3f})")


if __name__ == "__main__":
    main()
