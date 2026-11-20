"""
compute_podcast_magnitude_isc.py
====================================
Inter-subject correlation (ISC) of NEURAL curvature AND displacement
magnitude across the 9 podcast patients, who all listened to the same
stimulus. Extends compute_podcast_curvature_isc.py to also compute
displacement's ISC, reusing the same neural PCA fit per patient (cheap --
both metrics come off the same Y_neu), so the two ISC values are directly
comparable head-to-head: is curvature's cross-subject synchrony higher
than displacement's?

Patients are aligned to a common reference word sequence via text-based
sequence alignment (difflib), not by index or timestamp, since absolute
onset times differ session-to-session even though the word sequence
(stimulus) is identical.

Saves: results/podcast_magnitude_isc.json
"""
import json, os, sys, glob, difflib
import numpy as np
import pandas as pd
import h5py, scipy.sparse
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    ACCEPTED_QUAL, trajectory_curvature, trajectory_displacement, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, _convo_regions

ONSET_PAD_MS = 80
OFFSET_PAD_MS = 40
REGION = "hippocampus"
PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"
RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "podcast_magnitude_isc.json")


def find_spikes(pid):
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, "spikes", "*_spikes.mat"))
             if not os.path.basename(f).startswith(("._", "~$"))]
    return cands[0]


def find_transcript(pid):
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, "transcript", "*.xlsx"))
             if not os.path.basename(f).startswith(("._", "~$"))]
    return cands[0]


def load_words_df(pid):
    df = pd.read_excel(find_transcript(pid), sheet_name="Sheet1")
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    return df


def build_neu_metrics(pid, df):
    preg = _convo_regions(pid)
    with h5py.File(find_spikes(pid), "r") as f:
        qual_key = "qual" if "qual" in f else "quality"
        chan_key = "chan" if "chan" in f else "channelIds"
        spikes = scipy.sparse.csr_matrix((
            f["spikes"]["data"][()], f["spikes"]["ir"][()], f["spikes"]["jc"][()]
        )).toarray()
        qual = np.asarray(f[qual_key][()]).ravel()
        chan = np.asarray(f[chan_key][()]).ravel()
    T = spikes.shape[0]
    neuron_idx = [
        idx for lo, hi in preg[REGION]
        for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()
    ]
    reg_sp = spikes[:, neuron_idx]
    n = len(df)

    vecs = []
    for wi in range(n):
        onset_ms = df.iloc[wi]["onset"]
        offset_ms = df.iloc[wi]["offset"]
        onset_ms = int(onset_ms) if np.isfinite(onset_ms) else 0
        if np.isfinite(offset_ms) and offset_ms > onset_ms:
            offset_ms = int(offset_ms)
        else:
            offset_ms = onset_ms + 100
        t0 = max(0, onset_ms + ONSET_PAD_MS)
        t1 = min(T, offset_ms + OFFSET_PAD_MS)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    neu_vecs = np.array(vecs)

    Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        return None, None, None
    c_neu = trajectory_curvature(Y_neu)             # length n-2, index i ~ triple (i,i+1,i+2)
    d_neu = trajectory_displacement(Y_neu)[:n - 2]   # same length/indexing as curvature
    return c_neu, d_neu, n_pcs


def align_to_ref(ref_words, words):
    """Returns dict: ref_idx -> local_idx for exactly-matching aligned blocks."""
    sm = difflib.SequenceMatcher(None, ref_words, words, autojunk=False)
    mapping = {}
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def compute_isc(metric_by_pid, common_ref_idx, pids):
    mat = np.array([[metric_by_pid[pid][i] for i in common_ref_idx] for pid in pids])
    n_pat = len(pids)
    corr_mat = np.full((n_pat, n_pat), np.nan)
    p_mat = np.full((n_pat, n_pat), np.nan)
    for i in range(n_pat):
        for j in range(n_pat):
            if i == j:
                continue
            r, p = pearsonr(mat[i], mat[j])
            corr_mat[i, j] = r
            p_mat[i, j] = p
    loo_r, loo_p = {}, {}
    for i, pid in enumerate(pids):
        others_mean = np.mean(np.delete(mat, i, axis=0), axis=0)
        r, p = pearsonr(mat[i], others_mean)
        loo_r[pid] = float(r)
        loo_p[pid] = float(p)
    upper = [(pids[i], pids[j], float(corr_mat[i, j]), float(p_mat[i, j]))
             for i in range(n_pat) for j in range(i + 1, n_pat)]
    return upper, loo_r, loo_p


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    dfs = {pid: load_words_df(pid) for pid in PODCAST_PATIENTS}
    words = {pid: dfs[pid]["text"].astype(str).str.lower().tolist() for pid in PODCAST_PATIENTS}

    ref_pid = "YEZ"
    ref_words = words[ref_pid]
    print(f"Reference: {ref_pid}  n_words={len(ref_words)}", flush=True)

    mappings = {}
    for pid in PODCAST_PATIENTS:
        m = align_to_ref(ref_words, words[pid])
        mappings[pid] = m
        print(f"  {pid}: aligned {len(m)}/{len(ref_words)} words to reference", flush=True)

    curv, disp = {}, {}
    n_pcs_by_pid = {}
    for pid in PODCAST_PATIENTS:
        c_neu, d_neu, n_pcs = build_neu_metrics(pid, dfs[pid])
        if c_neu is None:
            print(f"  {pid}: SKIP (neural pca failed)")
            continue
        curv[pid] = c_neu
        disp[pid] = d_neu
        n_pcs_by_pid[pid] = n_pcs
        print(f"  {pid}: computed, n_pcs={n_pcs}, len={len(c_neu)}", flush=True)

    inv_mappings = {pid: {v: k for k, v in mappings[pid].items()} for pid in curv}

    ref_curv = {pid: {} for pid in curv}
    ref_disp = {pid: {} for pid in curv}
    for pid in curv:
        inv = inv_mappings[pid]
        for i in range(len(curv[pid])):
            local_word_idx = i + 2
            ref_idx = inv.get(local_word_idx)
            if ref_idx is not None:
                ref_curv[pid][ref_idx] = curv[pid][i]
                ref_disp[pid][ref_idx] = disp[pid][i]

    common_ref_idx = sorted(set.intersection(*[set(d.keys()) for d in ref_curv.values()]))
    print(f"\nCommon aligned+valid ref indices across all {len(curv)} patients: "
          f"{len(common_ref_idx)}", flush=True)

    pids = list(curv.keys())

    curv_pairwise, curv_loo_r, curv_loo_p = compute_isc(ref_curv, common_ref_idx, pids)
    disp_pairwise, disp_loo_r, disp_loo_p = compute_isc(ref_disp, common_ref_idx, pids)

    print("\n=== CURVATURE ISC ===")
    for a, b, r, p in sorted(curv_pairwise, key=lambda x: -x[2]):
        print(f"  {a}-{b}: r={r:+.4f}  p={p:.4g} {stars(p)}")
    curv_mean = float(np.nanmean([x[2] for x in curv_pairwise]))
    print(f"  Mean pairwise r: {curv_mean:+.4f}")

    print("\n=== DISPLACEMENT ISC ===")
    for a, b, r, p in sorted(disp_pairwise, key=lambda x: -x[2]):
        print(f"  {a}-{b}: r={r:+.4f}  p={p:.4g} {stars(p)}")
    disp_mean = float(np.nanmean([x[2] for x in disp_pairwise]))
    print(f"  Mean pairwise r: {disp_mean:+.4f}")

    print(f"\n=== HEAD-TO-HEAD ===")
    print(f"  Curvature mean pairwise ISC:    {curv_mean:+.4f}")
    print(f"  Displacement mean pairwise ISC: {disp_mean:+.4f}")
    print(f"  Curvature {'>' if curv_mean > disp_mean else '<'} Displacement "
          f"(diff = {curv_mean - disp_mean:+.4f})")

    print("\nLeave-one-out ISC (curvature vs displacement, subject vs mean of other 8):")
    for pid in pids:
        print(f"  {pid}: curv r={curv_loo_r[pid]:+.4f} {stars(curv_loo_p[pid])}   "
              f"disp r={disp_loo_r[pid]:+.4f} {stars(disp_loo_p[pid])}")

    out = dict(
        pids=pids,
        n_common_words=len(common_ref_idx),
        n_pcs_by_pid=n_pcs_by_pid,
        curvature=dict(
            pairwise=[dict(a=a, b=b, r=r, p=p) for a, b, r, p in curv_pairwise],
            loo_r=curv_loo_r, loo_p=curv_loo_p, mean_pairwise_r=curv_mean,
        ),
        displacement=dict(
            pairwise=[dict(a=a, b=b, r=r, p=p) for a, b, r, p in disp_pairwise],
            loo_r=disp_loo_r, loo_p=disp_loo_p, mean_pairwise_r=disp_mean,
        ),
    )
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
