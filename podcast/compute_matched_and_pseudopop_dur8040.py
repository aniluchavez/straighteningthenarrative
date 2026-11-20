"""
compute_matched_and_pseudopop_dur8040.py
====================================
Two follow-ups to the peak-layer model-comparison result (curvature vs
curvature, dur80/40 rate-normalized neural window):

1. "matched" -- instead of a fixed K=20 semantic PCA, use each patient's own
   max variance-threshold neural dimensionality (fit_neural_pca's n_pcs) as
   K on the semantic side too, per patient. Bootstrapped across patients,
   same as compute_model_comparison_bootstrap_dur8040.py but K varies by
   patient instead of being fixed at 20.

2. "pseudopopulation" -- since all 13 patients heard the IDENTICAL podcast
   stimulus, their word sequences are difflib-aligned to a reference
   (YEY) and the ~99.4%-common word backbone (7317/7361 words) is used to
   CONCATENATE all patients' neural channels word-by-word into one much
   higher-dimensional combined neural trajectory (sum of all 13 patients'
   channel counts), instead of treating each patient separately. PCA'd to
   its own matched (variance-threshold) dimensionality, correlated against
   the reference patient's semantic curvature at the same word backbone.
   Single continuous series -> block-bootstrap over chunks (not patients)
   for a CI.

Neural side throughout: onset+80ms/offset+40ms, rate-normalized.

Saves: results/matched_and_pseudopop_dur8040.json
"""
import json, os, sys, difflib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import trajectory_curvature, build_valid_mask, fit_neural_pca, fit_neural_raw
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

REGION = "hippocampus"
CHUNK = 15
N_BOOT = 2000
SEED = 0
ONSET_PAD_MS, OFFSET_PAD_MS = 80, 40
REF_PID = "YEY"

RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "matched_and_pseudopop_dur8040.json")

PEAK_LAYER = {
    "roberta-base": 1, "llama-2-7b": 1, "llama-3.1-8b": 1, "opt-350m": 6,
    "gemma-2-9b": 0, "mistral-7b": 2, "gpt2-medium": 1, "bert-base": 1,
    "gpt2-xl": 3, "gpt2": 1,
}


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


def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    if sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def get_neural(pid):
    """Full patient record: neu_vecs (dur80/40 rate), words, surp, sent_pos, is_turn."""
    data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data0 is None:
        return None
    try:
        _, onset_ms, offset_ms = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        return None
    preg = _convo_regions(pid)
    n = min(data0["n_words"], len(onset_ms))
    neu_vecs = windowed_neu_vecs(pid, REGION, preg, onset_ms[:n], offset_ms[:n],
                                  ONSET_PAD_MS, OFFSET_PAD_MS, rate=True)
    if neu_vecs is None:
        return None
    return dict(n=n, neu_vecs=neu_vecs, words=data0["words"][:n], surp=data0["surp"][:n],
                sent_pos=data0["sent_pos"][:n], is_turn=data0["is_turn"][:n])


# ── Part 1: matched-K per patient ───────────────────────────────────────────
def run_matched(neural):
    results = {}
    for model, layer in PEAK_LAYER.items():
        rows = []
        for pid, rec in neural.items():
            n = rec["n"]
            valid = build_valid_mask(n, rec["surp"], rec["sent_pos"], rec["is_turn"])
            if valid.sum() < 50:
                continue
            Y_neu, n_pcs = fit_neural_pca(rec["neu_vecs"], n)
            if Y_neu is None:
                continue
            c_neu = trajectory_curvature(Y_neu)[valid]

            data_m = load_podcast_patient(pid, region=REGION, model_tag=model, context_words=200)
            if data_m is None:
                continue
            emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
            if layer >= emb_mm.shape[0]:
                del emb_mm
                continue
            emb_L = np.array(emb_mm[layer, :n], dtype=np.float64)
            del emb_mm
            Xs = StandardScaler().fit_transform(emb_L)
            max_pc = min(Xs.shape[1], Xs.shape[0] - 1, 300)
            if max_pc < n_pcs:
                continue
            pca = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit(Xs)
            c_sem = trajectory_curvature(pca.transform(Xs)[:, :n_pcs])[valid]

            pairs = _chunk_pairs(c_sem, c_neu, CHUNK)
            if len(pairs) < 3:
                continue
            s = zscore([p[0] for p in pairs]); nz = zscore([p[1] for p in pairs])
            rows += [(pid, si, nzi) for si, nzi in zip(s, nz)]

        if len(rows) < 10:
            results[model] = None
            continue
        df = pd.DataFrame(rows, columns=["patient", "sem_z", "neu_z"])
        rng = np.random.default_rng(SEED)
        patients = df.patient.unique()
        r_obs, _ = pearsonr(df.sem_z, df.neu_z)
        boot_r = []
        for _ in range(N_BOOT):
            samp = rng.choice(patients, size=len(patients), replace=True)
            bd = pd.concat([df[df.patient == p] for p in samp])
            if len(bd) < 3:
                continue
            r_b, _ = pearsonr(bd.sem_z, bd.neu_z)
            boot_r.append(r_b)
        boot_r = np.array(boot_r)
        lo, hi = np.percentile(boot_r, [2.5, 97.5])
        results[model] = dict(layer=layer, r_obs=float(r_obs), boot_mean=float(boot_r.mean()),
                               ci_lo=float(lo), ci_hi=float(hi), n_chunks=len(rows))
        print(f"  [matched] {model:15s} layer={layer:3d}  r={r_obs:+.4f}  boot_mean={boot_r.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]", flush=True)
    return results


# ── Part 2: pseudopopulation ─────────────────────────────────────────────────
def align_to_ref(ref_words, words):
    sm = difflib.SequenceMatcher(None, ref_words, words, autojunk=False)
    mapping = {}
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def run_pseudopop(neural):
    ref_words = neural[REF_PID]["words"].tolist()
    common = set(range(len(ref_words)))
    mappings = {}
    for pid, rec in neural.items():
        m = align_to_ref(ref_words, rec["words"].tolist())
        mappings[pid] = m
        common &= set(m.keys())
    common_sorted = sorted(common)
    print(f"  [pseudopop] common words across {len(neural)} patients: {len(common_sorted)}/{len(ref_words)}", flush=True)

    # Concatenate channels across patients at each common ref word
    pids = list(neural.keys())
    combined = []
    for pid in pids:
        m = mappings[pid]
        local_idx = [m[r] for r in common_sorted]
        combined.append(neural[pid]["neu_vecs"][local_idx])
    combined = np.concatenate(combined, axis=1)  # (n_common, sum_channels)
    print(f"  [pseudopop] combined neural shape: {combined.shape}", flush=True)

    n_common = len(common_sorted)
    Y_pseudo = fit_neural_raw(combined, n_common)  # standardized, NO PCA reduction -- keep all neurons
    if Y_pseudo is None:
        return None
    n_pcs_pseudo = Y_pseudo.shape[1]
    c_neu_pseudo = trajectory_curvature(Y_pseudo)
    print(f"  [pseudopop] raw dims (all neurons)={n_pcs_pseudo}", flush=True)

    ref_rec = neural[REF_PID]
    valid = build_valid_mask(n_common, ref_rec["surp"][common_sorted], ref_rec["sent_pos"][common_sorted],
                              ref_rec["is_turn"][common_sorted])
    c_neu_pseudo_v = c_neu_pseudo[valid]

    results = {}
    for model, layer in PEAK_LAYER.items():
        data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=model, context_words=200)
        if data_m is None:
            results[model] = None
            continue
        emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
        if layer >= emb_mm.shape[0]:
            del emb_mm
            results[model] = None
            continue
        emb_L = np.array(emb_mm[layer, common_sorted], dtype=np.float64)
        del emb_mm
        Xs = StandardScaler().fit_transform(emb_L)
        max_pc = min(Xs.shape[1], Xs.shape[0] - 1, max(300, n_pcs_pseudo))
        if max_pc < n_pcs_pseudo:
            results[model] = None
            continue
        pca = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit(Xs)
        c_sem = trajectory_curvature(pca.transform(Xs)[:, :n_pcs_pseudo])[valid]

        pairs = _chunk_pairs(c_sem, c_neu_pseudo_v, CHUNK)
        if len(pairs) < 10:
            results[model] = None
            continue
        s = zscore([p[0] for p in pairs]); nz = zscore([p[1] for p in pairs])
        r_obs, _ = pearsonr(s, nz)

        rng = np.random.default_rng(SEED)
        n_pairs = len(pairs)
        boot_r = []
        s_arr, nz_arr = np.array(s), np.array(nz)
        for _ in range(N_BOOT):
            idx = rng.integers(0, n_pairs, size=n_pairs)
            r_b, _ = pearsonr(s_arr[idx], nz_arr[idx])
            boot_r.append(r_b)
        boot_r = np.array(boot_r)
        lo, hi = np.percentile(boot_r, [2.5, 97.5])
        results[model] = dict(layer=layer, r_obs=float(r_obs), boot_mean=float(boot_r.mean()),
                               ci_lo=float(lo), ci_hi=float(hi), n_chunks=n_pairs,
                               n_pcs_pseudo=n_pcs_pseudo, combined_channels=int(combined.shape[1]))
        print(f"  [pseudopop] {model:15s} layer={layer:3d}  r={r_obs:+.4f}  boot_mean={boot_r.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]", flush=True)
    return results


def main():
    neural = {}
    for pid in PODCAST_PATIENTS:
        rec = get_neural(pid)
        if rec is not None:
            neural[pid] = rec
            print(f"  {pid}: n_words={rec['n']}", flush=True)
    print(f"\n{len(neural)} patients loaded\n", flush=True)

    print("=== Part 1: matched-K per patient ===", flush=True)
    matched_results = run_matched(neural)

    print("\n=== Part 2: pseudopopulation ===", flush=True)
    pseudopop_results = run_pseudopop(neural)

    out = dict(matched=matched_results, pseudopop=pseudopop_results)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
