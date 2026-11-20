"""
compute_podcast_curvature_wins_grid.py
=========================================
Podcast version of ../compute_curvature_wins_grid.py: for every archived
model, sweep WIN_MS and scan every layer, computing raw (unnulled) Pearson
r(c_sem_layer, c_neu) and r(d_sem_layer, d_neu) pooled across all 9 podcast
patients. Reports which (window, layer) cells have curvature beating
displacement.
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

WIN_SIZES = [100, 300, 500, 1000, 2000]
REGION = "hippocampus"
PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"

MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "bert-base", "roberta-base",
          "opt-350m", "llama-2-7b", "llama-3.1-8b", "mistral-7b", "gemma-2-9b"]


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


def build_neu_vecs(pid, win_ms, data0):
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
        t1 = min(T, t0 + win_ms)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n


def main():
    out_path = os.path.join(BASE_DIR, "results", "podcast_curvature_wins_grid.json")
    all_results = {}
    for model_tag in MODELS:
        print(f"\n{'#'*70}\nMODEL: {model_tag}\n{'#'*70}", flush=True)
        model_results = {}
        for win_ms in WIN_SIZES:
            per_patient_rcurv, per_patient_rdisp = {}, {}
            n_layers = None
            for pid in PODCAST_PATIENTS:
                data0 = load_podcast_patient(pid, region=REGION, model_tag=model_tag, context_words=200)
                if data0 is None:
                    continue
                neu_vecs, n = build_neu_vecs(pid, win_ms, data0)
                valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
                if valid.sum() < 50:
                    continue
                Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
                if Y_neu is None:
                    continue
                c_neu = trajectory_curvature(Y_neu)[valid]
                d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

                emb_mm = np.load(data0["emb_path"], mmap_mode="r")
                n_layers = emb_mm.shape[0]
                r_curv, r_disp = [], []
                for li in range(n_layers):
                    emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
                    c_l = trajectory_curvature(emb_l)[valid]
                    d_l = trajectory_displacement(emb_l)[:n - 2][valid]
                    rc, _ = pearsonr(c_l, c_neu)
                    rd, _ = pearsonr(d_l, d_neu)
                    r_curv.append(float(rc))
                    r_disp.append(float(rd))
                del emb_mm
                per_patient_rcurv[pid] = r_curv
                per_patient_rdisp[pid] = r_disp

            pids = list(per_patient_rcurv.keys())
            if not pids or n_layers is None:
                print(f"  win={win_ms}: SKIP (no usable patients)", flush=True)
                continue
            pooled_rcurv = np.nanmean(np.array([per_patient_rcurv[p] for p in pids]), axis=0)
            pooled_rdisp = np.nanmean(np.array([per_patient_rdisp[p] for p in pids]), axis=0)
            winning_layers = [li for li in range(n_layers) if pooled_rcurv[li] > pooled_rdisp[li]]
            model_results[win_ms] = dict(
                n_layers=n_layers, n_patients=len(pids),
                r_curv=pooled_rcurv.tolist(), r_disp=pooled_rdisp.tolist(),
                curv_wins_layers=winning_layers,
            )
            frac = len(winning_layers) / n_layers
            print(f"  win={win_ms:5d}  n_pat={len(pids):2d}  n_layers={n_layers:3d}  "
                  f"curv wins at {len(winning_layers)}/{n_layers} layers ({100*frac:.0f}%)  "
                  f"layers={winning_layers}", flush=True)

        all_results[model_tag] = model_results
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
