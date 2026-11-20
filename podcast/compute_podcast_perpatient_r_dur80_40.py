"""
compute_podcast_perpatient_r_dur80_40.py
====================================
Per-patient (not pooled) semantic<->neural Pearson r for curvature and
displacement, at the correct duration window (onset+80/offset+40) and a
fixed K=20 (semantic PCA components, all 9 models averaged) -- answers
"is the curvature<->neural correlation similar across all patients, and
likewise for displacement" i.e. how consistent is the per-patient effect,
not whether raw magnitude synchronizes across subjects (that's ISC,
already computed separately).

Reuses cached semantic PCA embeddings (cache_pca_sweep_podcast/*.npz,
window-independent) and rebuilds neural vectors at the correct win.

Saves: results/podcast_perpatient_r_dur80_40.json
"""
import json, os, sys, glob
import numpy as np
import pandas as pd
import h5py, scipy.sparse
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    ACCEPTED_QUAL, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions

ONSET_PAD_MS = 80
OFFSET_PAD_MS = 40
CHUNK = 15
K = 20
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"
PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"

CACHE_DIR = os.path.join(BASE_DIR, "cache_pca_sweep_podcast")
RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "podcast_perpatient_r_dur80_40.json")


def find_spikes(pid):
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, "spikes", "*_spikes.mat"))
             if not os.path.basename(f).startswith(("._", "~$"))]
    return cands[0]


def find_transcript(pid):
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, "transcript", "*.xlsx"))
             if not os.path.basename(f).startswith(("._", "~$"))]
    return cands[0]


def build_neu_vecs(pid, data0):
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

    df = pd.read_excel(find_transcript(pid), sheet_name="Sheet1")
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    n = min(data0["n_words"], len(df))

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
    return np.array(vecs), n


def get_neural_data(pid):
    data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data0 is None:
        return None
    neu_vecs, n = build_neu_vecs(pid, data0)
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 50:
        return None
    Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]
    d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
    return n, valid, c_neu, d_neu, n_pcs


def load_cached_sem_k(pid, model, k):
    cache_path = os.path.join(CACHE_DIR, f"{pid}_{model}_allK.npz")
    if not os.path.exists(cache_path):
        return None
    z = np.load(cache_path, allow_pickle=True)
    return z[f"c_{k}"], z[f"d_{k}"]


def _chunk_pairs(x, y, cs, min_frac=0.5):
    n = len(x)
    out = []
    for i in range(0, n - cs + 1, cs):
        sl = slice(i, i + cs)
        fin = np.isfinite(x[sl]) & np.isfinite(y[sl])
        if fin.sum() < cs * min_frac:
            continue
        out.append((np.mean(x[sl][fin]), np.mean(y[sl][fin])))
    return out


def main():
    results = {}
    for pid in PODCAST_PATIENTS:
        res = get_neural_data(pid)
        if res is None:
            print(f"  {pid}: SKIP (neural)")
            continue
        n, valid, c_neu, d_neu, n_pcs = res

        c_sem_layers, d_sem_layers = [], []
        for model in MODELS:
            sem = load_cached_sem_k(pid, model, K)
            if sem is None:
                continue
            c_sem_layers.append(sem[0])
            d_sem_layers.append(sem[1])
        if not c_sem_layers:
            print(f"  {pid}: SKIP (no cached sem)")
            continue
        c_sem = np.mean(c_sem_layers, axis=0)
        d_sem = np.mean(d_sem_layers, axis=0)

        c_pairs = _chunk_pairs(c_sem, c_neu, CHUNK)
        d_pairs = _chunk_pairs(d_sem, d_neu, CHUNK)
        if len(c_pairs) < 5 or len(d_pairs) < 5:
            print(f"  {pid}: SKIP (too few chunks)")
            continue

        rc, pc = pearsonr([p[0] for p in c_pairs], [p[1] for p in c_pairs])
        rd, pd_ = pearsonr([p[0] for p in d_pairs], [p[1] for p in d_pairs])
        results[pid] = dict(rc=float(rc), pc=float(pc), rd=float(rd), pd=float(pd_),
                             n_pcs=n_pcs, n_chunks=len(c_pairs))
        print(f"  {pid}: n_pcs={n_pcs}  curv r={rc:+.4f} p={pc:.4g}   "
              f"disp r={rd:+.4f} p={pd_:.4g}   n_chunks={len(c_pairs)}", flush=True)

    rc_vals = np.array([v["rc"] for v in results.values()])
    rd_vals = np.array([v["rd"] for v in results.values()])
    print(f"\nCurvature per-patient r: mean={rc_vals.mean():+.4f}  sd={rc_vals.std():.4f}  "
          f"min={rc_vals.min():+.4f}  max={rc_vals.max():+.4f}")
    print(f"Displacement per-patient r: mean={rd_vals.mean():+.4f}  sd={rd_vals.std():.4f}  "
          f"min={rd_vals.min():+.4f}  max={rd_vals.max():+.4f}")
    n_pos_curv = int((rc_vals > 0).sum())
    n_pos_disp = int((rd_vals > 0).sum())
    print(f"Curvature: {n_pos_curv}/{len(rc_vals)} patients positive")
    print(f"Displacement: {n_pos_disp}/{len(rd_vals)} patients positive")

    json.dump(dict(per_patient=results,
                    curv_mean=float(rc_vals.mean()), curv_sd=float(rc_vals.std()),
                    disp_mean=float(rd_vals.mean()), disp_sd=float(rd_vals.std())),
               open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
