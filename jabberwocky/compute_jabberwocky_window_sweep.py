"""
compute_jabberwocky_window_sweep.py
======================================
Full pipeline for the Jabberwocky dataset, correcting the mid-layer-shortcut
bug this session found in the convo/podcast investigation scripts:

  1. Per-model, per-patient LOO-neural layer selection: for each held-out
     patient, pick the layer maximizing |mean r(c_sem_layer, c_neu)| across
     the OTHER 6 patients (default WIN_MS=500), exactly mirroring
     run_geometry_analysis.py's Pass-1/LOO logic.
  2. WIN_MS sweep (100-2000ms) using each patient's own LOO layer, chunk=15
     permutation-null z-scored curvature/displacement correlation -- same
     methodology validated on convo/podcast this session.

Models: gpt2-medium, llama-3.1-8b (the two "top models" used throughout
this investigation).
"""
import sys, os, json, numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from scipy.stats import pearsonr
from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from jabberwocky_loader import JABBERWOCKY_PATIENTS, load_jabberwocky_patient

WIN_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000]
CHUNK = 15
N_PERM = 100
MODELS = ["gpt2-medium", "llama-3.1-8b"]
REGION = "hippocampus"

rng = np.random.default_rng(0)


def chunk_r(x, y, cs, min_chunks=3):
    n = len(x)
    xa, ya = [], []
    for i in range(0, n - cs + 1, cs):
        sl = slice(i, i + cs)
        fin = np.isfinite(x[sl]) & np.isfinite(y[sl])
        if fin.sum() < cs // 2:
            continue
        xa.append(np.mean(x[sl][fin])); ya.append(np.mean(y[sl][fin]))
    if len(xa) < min_chunks:
        return np.nan
    r, _ = pearsonr(xa, ya)
    return r


def select_loo_layers(model_tag):
    """Returns {pid: loo_layer} using default WIN_MS=500 neu_vecs (built by
    load_jabberwocky_patient), mirroring run_geometry_analysis.py's
    compute_profiles + LOO selection."""
    profiles = {}
    for pid in JABBERWOCKY_PATIENTS:
        data = load_jabberwocky_patient(pid, region=REGION, model_tag=model_tag, context_words=200)
        if data is None:
            print(f"  {pid}: SKIP (no data)")
            continue
        n = data["n_words"]
        valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])
        if valid.sum() < 50:
            print(f"  {pid}: SKIP (too few valid words: {valid.sum()})")
            continue
        Y_neu, n_pcs = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            print(f"  {pid}: SKIP (PCA failed)")
            continue
        c_neu = trajectory_curvature(Y_neu)[valid]

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        n_layers = emb_mm.shape[0]
        r_neural = []
        for li in range(n_layers):
            emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
            c_l = trajectory_curvature(emb_l)[valid]
            r_n, _ = pearsonr(c_l, c_neu)
            r_neural.append(float(r_n))
        del emb_mm
        profiles[pid] = dict(n=int(valid.sum()), n_pcs=n_pcs, r_neural=r_neural)
        print(f"  {pid}: n={profiles[pid]['n']}  n_pcs={n_pcs}  n_layers={n_layers}")

    pids = list(profiles.keys())
    loo_layer = {}
    for pid in pids:
        others = [p for p in pids if p != pid]
        rn_mat = np.array([profiles[p]["r_neural"] for p in others], dtype=float)
        rn_mean = np.nanmean(rn_mat, axis=0)
        loo_layer[pid] = int(np.nanargmax(np.abs(rn_mean))) if not np.all(np.isnan(rn_mean)) else 0
    return loo_layer


def run_window_sweep(model_tag, loo_layer):
    results = {}
    group_c = {w: [] for w in WIN_SIZES}
    group_d = {w: [] for w in WIN_SIZES}
    for pid, layer in loo_layer.items():
        results[pid] = {}
        row = []
        for win_ms in WIN_SIZES:
            # neu_vecs are rebuilt at WIN_MS=500 by the loader (fixed default);
            # to sweep WIN_MS we need to rebuild neu_vecs at each window size.
            data = load_jabberwocky_patient(pid, region=REGION, model_tag=model_tag, context_words=200)
            if data is None:
                row.append((win_ms, np.nan, np.nan)); continue
            n = data["n_words"]
            valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])

            # rebuild neu_vecs at this win_ms directly (loader default is WIN_MS=500)
            from jabberwocky_loader import find_spikes, _convo_regions
            import h5py, scipy.sparse, pandas as pd
            from run_geometry_analysis import ACCEPTED_QUAL, LISTEN_OFFSET_MS

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

            from jabberwocky_loader import find_transcript
            df = pd.read_excel(find_transcript(pid), sheet_name="Sheet1")
            df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
            df = df.dropna(subset=["onset"]).reset_index(drop=True)
            n2 = min(n, len(df))

            vecs = []
            for wi in range(n2):
                onset_ms = int(df.iloc[wi]["onset"])
                t0 = max(0, onset_ms + LISTEN_OFFSET_MS)
                t1 = min(T, t0 + win_ms)
                vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
            neu_vecs = np.array(vecs)

            valid2 = valid[:n2]
            Y_neu, _ = fit_neural_pca(neu_vecs, n2)
            if Y_neu is None:
                row.append((win_ms, np.nan, np.nan)); continue
            c_neu = trajectory_curvature(Y_neu)[valid2]
            d_neu = trajectory_displacement(Y_neu)[:n2 - 2][valid2]

            emb_mm = np.load(data["emb_path"], mmap_mode="r")
            emb_L = np.array(emb_mm[layer, :n2], dtype=np.float64)
            del emb_mm
            c_sem = trajectory_curvature(emb_L)[valid2]
            d_sem = trajectory_displacement(emb_L)[:n2 - 2][valid2]

            r_c = chunk_r(c_sem, c_neu, CHUNK)
            r_d = chunk_r(d_sem, d_neu, CHUNK)
            null_c, null_d = [], []
            for _ in range(N_PERM):
                perm = rng.permutation(len(c_neu))
                null_c.append(chunk_r(c_sem, c_neu[perm], CHUNK))
                perm2 = rng.permutation(len(d_neu))
                null_d.append(chunk_r(d_sem, d_neu[perm2], CHUNK))
            nc_mu, nc_sd = np.nanmean(null_c), np.nanstd(null_c)
            nd_mu, nd_sd = np.nanmean(null_d), np.nanstd(null_d)
            zc = (r_c - nc_mu) / nc_sd if nc_sd > 0 else np.nan
            zd = (r_d - nd_mu) / nd_sd if nd_sd > 0 else np.nan

            results[pid][win_ms] = dict(curv=r_c, disp=r_d, curv_z=zc, disp_z=zd, layer=layer)
            row.append((win_ms, zc, zd))
            group_c[win_ms].append(zc); group_d[win_ms].append(zd)
        print(pid, [f"w={w} c={zc:.2f} d={zd:.2f}" for w, zc, zd in row])

    print(f"\n{model_tag} GROUP MEANS:")
    for w in WIN_SIZES:
        print(f"  win={w:5d}  curv_z={np.nanmean(group_c[w]):6.2f}  disp_z={np.nanmean(group_d[w]):6.2f}")
    return results


def main():
    out_path = os.path.join(BASE_DIR, "results", "jabberwocky_window_length_sweep.json")
    all_results = {}
    for model_tag in MODELS:
        print(f"\n{'#'*70}\nMODEL: {model_tag} -- LOO layer selection\n{'#'*70}")
        loo_layer = select_loo_layers(model_tag)
        print(f"\n  LOO layers: {loo_layer}")
        print(f"\n{'#'*70}\nMODEL: {model_tag} -- WIN_MS sweep\n{'#'*70}")
        results = run_window_sweep(model_tag, loo_layer)
        all_results[model_tag] = dict(loo_layer=loo_layer, results=results)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
