"""
compute_curv_disp_pos_gpt2xl_layer29_factorial.py
====================================================
Why does the winning metric (curvature / displacement / position) change
between analyses? Per patient, gpt2-xl layer 29, chunk=3 on valid words,
crossing:

  window : "dur_count"  [onset+80, offset+40] raw spike counts
           "dur_rate"   same window, count / window length
           "fixed250"   [onset+80, onset+330] counts (same length for every word)
  K      : 8, 16, 32, 64 matched top PCs (standardized; semantic K = the neural
           K actually available for that patient), or "full" (semantic raw
           1600-d embedding, neural all units z-scored)

Position = coordinate on PC1 of each space. For each cell and metric:
  r         Pearson r of chunk means, semantic vs neural
  r_partial same after regressing chunk-mean log word duration out of both
  r_dur_sem / r_dur_neu  r of each side's chunk-mean metric with log duration
All 13 patients (YFX with the +574.9 s clock fix).

Saves: results/curv_disp_pos_gpt2xl_layer29_factorial.json
"""
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import trajectory_curvature, trajectory_displacement, build_valid_mask
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions, load_words_onset_offset, windowed_neu_vecs
from compute_curv_disp_pos_gpt2xl_layer29 import pca_fixed

MODEL, LAYER, CHUNK = "gpt2-xl", 29, 3
KS = [8, 16, 32, 64, "full"]
WINDOWS = ["dur_count", "dur_rate", "fixed250"]
METRICS = ["curv", "disp", "pos"]
OUT_PATH = os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_factorial.json")


def feats(Y, Ypos):
    return dict(curv=trajectory_curvature(Y), disp=trajectory_displacement(Y), pos=Ypos)


def masked(x, valid):
    m = min(len(x), len(valid))
    return x[:m][valid[:m]]


def chunks(x):
    n = len(x) // CHUNK * CHUNK
    return x[:n].reshape(-1, CHUNK).mean(axis=1)


def resid(y, x):
    X = np.c_[np.ones_like(x), x]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    out = dict(config=dict(KS=[str(k) for k in KS], WINDOWS=WINDOWS, LAYER=LAYER, CHUNK=CHUNK), patients={})
    for pid in PODCAST_PATIENTS:
        try:
            _, on, off = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
        if data is None:
            continue
        n = min(data["n_words"], len(on))
        on, off = on[:n], off[:n]
        preg = _convo_regions(pid)
        neu = {"dur_count": windowed_neu_vecs(pid, "hippocampus", preg, on, off, 80, 40, rate=False),
               "dur_rate": windowed_neu_vecs(pid, "hippocampus", preg, on, off, 80, 40, rate=True),
               "fixed250": windowed_neu_vecs(pid, "hippocampus", preg, on, on, 80, 330, rate=False)}
        if neu["fixed250"] is None:
            continue
        valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
        dur = off - on
        dur = np.where(np.isfinite(dur) & (dur > 0), dur, np.nanmedian(dur[dur > 0]))
        # duration of the word at the END of each step (index i+1 for disp, i+2 for curv); use the word itself for pos
        logd = np.log(dur)
        emb = np.array(np.load(data["emb_path"], mmap_mode="r")[LAYER, :n], dtype=np.float64)
        sem_pos = pca_fixed(emb, k=2)[:, 0]
        n_units = neu["fixed250"].shape[1]
        rec = {}
        for K in KS:
            k_used = n_units if K == "full" else min(K, n_units)
            Ys = emb if K == "full" else pca_fixed(emb, k=k_used)
            fs = feats(Ys, sem_pos)
            for w in WINDOWS:
                X = neu[w][:n]
                Yn = np.nan_to_num(StandardScaler().fit_transform(X)) if K == "full" else pca_fixed(X, k=k_used)
                fn = feats(Yn, pca_fixed(X, k=2)[:, 0])
                cell = dict(k_used=int(k_used))
                for m in METRICS:
                    d_al = {"curv": logd[1:-1], "disp": logd[1:], "pos": logd}[m]
                    a, b, dd = masked(fs[m], valid), masked(fn[m], valid), masked(d_al, valid)
                    L = min(len(a), len(b), len(dd))
                    a, b, dd = chunks(a[:L]), chunks(b[:L]), chunks(dd[:L])
                    cell[m] = dict(r=r(a, b), r_partial=r(resid(a, dd), resid(b, dd)),
                                   r_dur_sem=r(a, dd), r_dur_neu=r(b, dd))
                rec[f"{w}|{K}"] = cell
        out["patients"][pid] = rec
        print(f"  {pid}: units={n_units}  " + "  ".join(
            f"{w[:5]}|{K}: c{rec[f'{w}|{K}']['curv']['r']:+.2f} d{rec[f'{w}|{K}']['disp']['r']:+.2f}"
            for w in WINDOWS for K in (8, "full")), flush=True)
        json.dump(out, open(OUT_PATH, "w"))

    print("\n=== mean across patients: r  [duration-partial r] ===")
    for w in WINDOWS:
        for K in KS:
            line = f"  {w:9s} K={str(K):>4}"
            for m in METRICS:
                v = np.array([p[f"{w}|{K}"][m]["r"] for p in out["patients"].values()])
                vp = np.array([p[f"{w}|{K}"][m]["r_partial"] for p in out["patients"].values()])
                line += f"  {m}={v.mean():+.3f} [{vp.mean():+.3f}]"
            print(line)
    print("\n=== mean r(metric, log duration): semantic / neural ===")
    for w in WINDOWS:
        for K in (8, "full"):
            line = f"  {w:9s} K={str(K):>4}"
            for m in METRICS:
                s = np.mean([p[f"{w}|{K}"][m]["r_dur_sem"] for p in out["patients"].values()])
                nn = np.mean([p[f"{w}|{K}"][m]["r_dur_neu"] for p in out["patients"].values()])
                line += f"  {m}: {s:+.3f}/{nn:+.3f}"
            print(line)
    json.dump(out, open(OUT_PATH, "w"))
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
