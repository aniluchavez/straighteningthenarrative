"""
compute_podcast_layer_profile.py
===================================
Podcast version of ../compute_convo_layer_profile.py: per-layer mean
curvature/displacement and their correlation with neural curvature/
displacement, pooled across all 9 podcast patients, at a SMALL window
(WIN_MS=300ms), for both gpt2-medium and llama-3.1-8b. Matches the format
plotted in the Jabberwocky Layer Profile artifact.
"""
import sys, os, json, numpy as np, h5py, scipy.sparse, pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from scipy.stats import pearsonr
from run_geometry_analysis import (
    PATIENTS, ACCEPTED_QUAL, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions

WIN_MS = 300
MODELS = ["gpt2-medium", "llama-3.1-8b"]
REGION = "hippocampus"
PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"


def find_transcript(pid):
    import glob
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, "transcript", "*.xlsx"))
             if not os.path.basename(f).startswith(("._", "~$"))]
    return cands[0]


def find_spikes(pid):
    import glob
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, "spikes", "*_spikes.mat"))
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
    n = min(data0["n_words"], len(df))

    vecs = []
    for wi in range(n):
        onset_ms = df.iloc[wi]["onset"]
        onset_ms = int(onset_ms) if np.isfinite(onset_ms) else 0
        t0 = max(0, onset_ms + LISTEN_OFFSET_MS)
        t1 = min(T, t0 + WIN_MS)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n


def main():
    out_path = os.path.join(BASE_DIR, "results", "podcast_layer_profile_win300.json")
    all_results = {}
    for model_tag in MODELS:
        print(f"\n{'#'*70}\nMODEL: {model_tag}  WIN_MS={WIN_MS}\n{'#'*70}")
        per_patient_curv, per_patient_disp = {}, {}
        per_patient_rcurv, per_patient_rdisp = {}, {}
        n_layers = None
        for pid in PODCAST_PATIENTS:
            data0 = load_podcast_patient(pid, region=REGION, model_tag=model_tag, context_words=200)
            if data0 is None:
                print(f"  {pid}: SKIP (no data)")
                continue
            neu_vecs, n = build_neu_vecs(pid, data0)
            valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
            if valid.sum() < 50:
                print(f"  {pid}: SKIP (too few valid words)")
                continue
            Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
            if Y_neu is None:
                print(f"  {pid}: SKIP (PCA failed)")
                continue
            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

            emb_mm = np.load(data0["emb_path"], mmap_mode="r")
            n_layers = emb_mm.shape[0]
            mean_curv, mean_disp, r_curv, r_disp = [], [], [], []
            for li in range(n_layers):
                emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
                c_l = trajectory_curvature(emb_l)[valid]
                d_l = trajectory_displacement(emb_l)[:n - 2][valid]
                mean_curv.append(float(c_l.mean()))
                mean_disp.append(float(d_l.mean()))
                rc, _ = pearsonr(c_l, c_neu)
                rd, _ = pearsonr(d_l, d_neu)
                r_curv.append(float(rc))
                r_disp.append(float(rd))
            del emb_mm
            per_patient_curv[pid] = mean_curv
            per_patient_disp[pid] = mean_disp
            per_patient_rcurv[pid] = r_curv
            per_patient_rdisp[pid] = r_disp
            print(f"  {pid}: n={int(valid.sum())}  n_pcs={n_pcs}  n_layers={n_layers}")

        pids = list(per_patient_curv.keys())
        loo_layer = {}
        for pid in pids:
            others = [p for p in pids if p != pid]
            rc_mat = np.array([per_patient_rcurv[p] for p in others], dtype=float)
            rc_mean = np.nanmean(rc_mat, axis=0)
            loo_layer[pid] = int(np.nanargmax(np.abs(rc_mean))) if not np.all(np.isnan(rc_mean)) else 0
        print(f"\n  LOO layers: {loo_layer}")

        pooled_curv = np.nanmean(np.array([per_patient_curv[p] for p in pids]), axis=0)
        pooled_disp = np.nanmean(np.array([per_patient_disp[p] for p in pids]), axis=0)
        pooled_rcurv = np.nanmean(np.array([per_patient_rcurv[p] for p in pids]), axis=0)
        pooled_rdisp = np.nanmean(np.array([per_patient_rdisp[p] for p in pids]), axis=0)

        print("layer  mean_curv  mean_disp  r_curv   r_disp")
        for li in range(n_layers):
            print(f"  {li:3d}   {pooled_curv[li]:8.3f}  {pooled_disp[li]:9.3f}  {pooled_rcurv[li]:+.4f}  {pooled_rdisp[li]:+.4f}")

        all_results[model_tag] = dict(
            n_layers=n_layers, win_ms=WIN_MS, loo_layer=loo_layer,
            mean_curv=pooled_curv.tolist(), mean_disp=pooled_disp.tolist(),
            r_curv=pooled_rcurv.tolist(), r_disp=pooled_rdisp.tolist(),
        )
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
