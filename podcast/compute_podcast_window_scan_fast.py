"""
compute_podcast_window_scan_fast.py
======================================
Podcast version of ../compute_new4_window_scan_fast.py: fast (no permutation
null) scan across spike-integration windows for all 9 podcast patients, at
chunk=15, all-layers-averaged, all 9 models -- to find whether ANY window
shows raw curvature beating raw displacement under the official methodology.
"""
import sys, os, numpy as np, h5py, scipy.sparse, pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from scipy.stats import pearsonr
from run_geometry_analysis import (
    ACCEPTED_QUAL, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions

WIN_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000]
CHUNK = 15
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"
PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"


def _chunk_r(x, y, cs, min_chunks=3):
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
    results = {win: {"curv": [], "disp": []} for win in WIN_SIZES}
    for win_ms in WIN_SIZES:
        print(f"\n=== win={win_ms}ms ===", flush=True)
        for model in MODELS:
            for pid in PODCAST_PATIENTS:
                data0 = load_podcast_patient(pid, region=REGION, model_tag=model, context_words=200)
                if data0 is None:
                    continue
                neu_vecs, n = build_neu_vecs(pid, win_ms, data0)
                valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
                if valid.sum() < 50:
                    continue
                Y_neu, _ = fit_neural_pca(neu_vecs, n)
                if Y_neu is None:
                    continue
                c_neu = trajectory_curvature(Y_neu)[valid]
                d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

                emb_mm = np.load(data0["emb_path"], mmap_mode="r")
                n_layers = emb_mm.shape[0]
                curv_layers, disp_layers = [], []
                for L in range(n_layers):
                    emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
                    c_l = trajectory_curvature(emb_L)[valid]
                    d_l = trajectory_displacement(emb_L)[:n - 2][valid]
                    curv_layers.append(_chunk_r(c_l, c_neu, CHUNK))
                    disp_layers.append(_chunk_r(d_l, d_neu, CHUNK))
                del emb_mm
                cv = np.nanmean(curv_layers); dv = np.nanmean(disp_layers)
                results[win_ms]["curv"].append(cv)
                results[win_ms]["disp"].append(dv)
        cmean = np.nanmean(results[win_ms]["curv"])
        dmean = np.nanmean(results[win_ms]["disp"])
        winner = "CURV WINS" if cmean > dmean else "disp wins"
        print(f"  win={win_ms:5d}  curv_mean={cmean:+.4f}  disp_mean={dmean:+.4f}  {winner}", flush=True)


if __name__ == "__main__":
    main()
