"""
make_curvature_at_speaker_changes_convo_gpt2xl_layer29.py
============================================================
Does trajectory curvature change at speaker changes in conversation?
15 convo patients (run_geometry_analysis.PATIENTS), hippocampus.

Full space: semantic = raw gpt2-xl layer 29 embedding (convo cache);
neural = all hippocampal units, fixed 250 ms window [onset+80, onset+330]
spike counts for EVERY word (same alignment whoever speaks, so the window
does not jump at the turn), z-scored per unit.

Events = first word of a new speaker's turn, by type:
  "patient starts": previous word by someone else, this word by the patient
  "patient stops":  previous word by the patient, this word by someone else
  "other -> other": both non-patient speakers (different speakers)
Control = sentence starts WITHIN the same speaker's turn (sent_pos == 0, no
speaker change), since every turn change is also a sentence start.
Lag convention: lag 0 = angle whose second step ends at the event word;
lag +1 = turning angle AT the event word.
Per-patient mean curve (>= MIN_EV events), then mean +/- SEM across patients.

Saves: figures/fig_curvature_at_speaker_changes_convo_gpt2xl_layer29.{pdf,png,svg}
       results/curvature_at_speaker_changes_convo_gpt2xl_layer29.json
"""
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.stats import sem as scipy_sem, ttest_rel
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
FIGURES = os.path.join(BASE_DIR, "figures")
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)

from run_geometry_analysis import PATIENTS, ACCEPTED_QUAL, build_paths, load_patient, trajectory_curvature

MODEL, LAYER, REGION = "gpt2-xl", 29, "hippocampus"
PAD_ON, PAD_OFF = 80, 330
WINDOW, MIN_EV = 5, 5
TYPES = [("start", "Patient starts speaking"), ("stop", "Patient stops speaking"), ("oo", "Other → other speaker")]
C_EV, C_CTRL = "#8E44AD", "0.45"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def fixed_window_counts(pid, preg, onsets):
    paths = build_paths(pid, MODEL)
    with h5py.File(paths["spikes"], "r") as f:
        spikes = scipy.sparse.csr_matrix((f["spikes"]["data"][()], f["spikes"]["ir"][()],
                                          f["spikes"]["jc"][()])).toarray()
        qual = np.asarray(f["qual" if "qual" in f else "quality"][()]).ravel()
        chan = np.asarray(f["chan" if "chan" in f else "channelIds"][()]).ravel()
    idx = [i for lo, hi in preg[REGION] for i in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0]]
    sp = spikes[:, idx]
    cs = np.vstack([np.zeros((1, sp.shape[1])), np.cumsum(sp, axis=0)])
    T = sp.shape[0]
    t0 = np.clip(onsets.astype(int) + PAD_ON, 0, T)
    t1 = np.clip(onsets.astype(int) + PAD_OFF, 0, T)
    return cs[t1] - cs[t0]


def speakers(pid):
    paths = build_paths(pid, MODEL)
    xl = pd.ExcelFile(paths["xlsx"])
    df = pd.read_excel(paths["xlsx"], sheet_name="Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0],
                       keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).reset_index(drop=True)

    def _spk(row):
        for c in ["Speaker1", "Speaker2", "Speaker3", "Speaker4"]:
            if str(row.get(c, "")).strip():
                return c
        return None
    return df["onset"].values.astype(float), df.apply(_spk, axis=1).values


def main():
    lags = np.arange(-WINDOW, WINDOW + 1)
    per = {(s, t): [] for s in ("sem", "neu") for t in [k for k, _ in TYPES] + ["ctrl"]}
    counts = {}
    for pid, preg in PATIENTS.items():
        if REGION not in preg:
            continue
        d = load_patient(pid, preg, REGION, MODEL)
        if d is None:
            continue
        n = d["n_words"]
        onsets, spk = speakers(pid)
        onsets, spk = onsets[:n], spk[:n]
        neu = fixed_window_counts(pid, preg, onsets)
        if neu.shape[1] < 3:
            continue
        emb = np.array(np.load(d["emb_path"], mmap_mode="r")[LAYER, :n], dtype=np.float64)
        at = dict(sem=np.r_[np.nan, trajectory_curvature(emb), np.nan][:n],
                  neu=np.r_[np.nan, trajectory_curvature(np.nan_to_num(StandardScaler().fit_transform(neu))), np.nan][:n])
        is_p = spk == "Speaker1"
        prev_ok = np.r_[False, (spk[1:] != None) & (spk[:-1] != None)]  # noqa: E711
        change = prev_ok & np.r_[False, spk[1:] != spk[:-1]]
        ev = dict(start=change & is_p & ~np.r_[False, is_p[:-1]],
                  stop=change & ~is_p & np.r_[False, is_p[:-1]],
                  oo=change & ~is_p & ~np.r_[False, is_p[:-1]],
                  ctrl=(d["sent_pos"][:n] == 0) & ~change & prev_ok)
        counts[pid] = {}
        for t, m in ev.items():
            js = np.where(m)[0]
            js = js[(js - WINDOW - 1 >= 1) & (js + WINDOW - 1 < n - 1)]
            counts[pid][t] = int(len(js))
            if len(js) < MIN_EV:
                continue
            for s in ("sem", "neu"):
                per[(s, t)].append((pid, np.nanmean(at[s][js[:, None] + lags[None, :] - 1], axis=0)))
        print(f"  {pid}: units={neu.shape[1]} " + " ".join(f"{t}={c}" for t, c in counts[pid].items()), flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey="row")
    out = dict(counts=counts, results={})
    for r, (s, sname) in enumerate((("sem", f"Semantic ({MODEL} layer {LAYER}, full space)"),
                                    ("neu", "Neural (hippocampus, all units, fixed 250 ms window)"))):
        ctrl = dict(per[(s, "ctrl")])
        for c, (t, tname) in enumerate(TYPES):
            ax = axes[r, c]
            evd = dict(per[(s, t)])
            pids = [p for p in evd if p in ctrl]
            E = np.array([evd[p] for p in pids]); C = np.array([ctrl[p] for p in pids])
            for arr, col, lab in ((E, C_EV, f"{tname}"), (C, C_CTRL, "Sentence start within a turn (control)")):
                mu, se = np.nanmean(arr, axis=0), scipy_sem(arr, axis=0, nan_policy="omit")
                ax.plot(lags, mu, color=col, lw=1.8, marker="o", ms=3.5, label=lab)
                ax.fill_between(lags, mu - se, mu + se, color=col, alpha=0.2)
            i1 = WINDOW + 1
            tt = ttest_rel(E[:, i1], C[:, i1])
            ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
            ax.axvline(1, color="0.6", lw=0.6, ls=":")
            ax.set_xticks(lags)
            ax.set_title(f"{'ABCDEF'[3 * r + c]}  {sname}\n{tname}: n = {len(pids)} patients; "
                         f"lag +1 vs control p = {tt.pvalue:.3g}", loc="left", fontsize=9.5)
            if c == 0:
                ax.set_ylabel("Turning angle (°)")
            if r == 1:
                ax.set_xlabel("Words relative to the first word of the new turn\n(+1 = angle at that word)")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            em, cm = np.nanmean(E, axis=0), np.nanmean(C, axis=0)
            out["results"][f"{s}|{t}"] = dict(pids=pids, event_mean=em.tolist(), ctrl_mean=cm.tolist(),
                                              p_lag1=float(tt.pvalue), t_lag1=float(tt.statistic))
            print(f"  {s} {t:5s} n={len(pids):2d}  lag0 {em[WINDOW]:.1f}/{cm[WINDOW]:.1f}  lag+1 {em[i1]:.1f}/{cm[i1]:.1f}  "
                  f"lag+2 {em[i1 + 1]:.1f}/{cm[i1 + 1]:.1f}  base {em[:3].mean():.1f}/{cm[:3].mean():.1f}  "
                  f"t={tt.statistic:+.2f} p={tt.pvalue:.3g}")
    fig.suptitle("Curvature at speaker changes in conversation (event vs. within-turn sentence starts), "
                 "mean ± SEM across patients", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig_curvature_at_speaker_changes_convo_gpt2xl_layer29"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"))
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "curvature_at_speaker_changes_convo_gpt2xl_layer29.json"), "w"), indent=1)
    print(f"  -> {os.path.join(FIGURES, stem)}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
