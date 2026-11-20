"""
compute_new4_window_scan_fast.py
===================================
Fast (no permutation null) scan across spike-integration windows for new4
patients only, at chunk=15, all-layers-averaged, all 9 models -- to find
whether ANY window shows raw curvature beating raw displacement under the
official methodology (not just the single "best" window already tested).
"""
import sys, os, numpy as np, h5py, scipy.sparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import pearsonr
from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)

WIN_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000]
CHUNK = 15
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"
NEW4 = ["PTYFM_task104", "PTYFP_task88", "PTYFR_task91", "PTYFS_task95"]


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


def build_neu_vecs(pid, model_for_meta, win_ms):
    paths = build_paths(pid, model_for_meta)
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, model_for_meta)
    if data0 is None:
        return None, None, None
    n = data0["n_words"]
    with h5py.File(paths["spikes"], "r") as f:
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

    xl = pd.ExcelFile(paths["xlsx"])
    sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(paths["xlsx"], sheet_name=sheet, keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).reset_index(drop=True)
    n = min(n, len(df))
    is_spk1 = df["Speaker1"].astype(str).str.strip() != ""

    vecs = []
    for wi in range(n):
        onset_ms = int(df.iloc[wi]["onset"])
        t0 = max(0, onset_ms - SPEAK_OFFSET_MS) if is_spk1.iloc[wi] else max(0, onset_ms + LISTEN_OFFSET_MS)
        t1 = min(T, t0 + win_ms)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n, data0


def main():
    results = {win: {"curv": [], "disp": []} for win in WIN_SIZES}
    for win_ms in WIN_SIZES:
        print(f"\n=== win={win_ms}ms ===", flush=True)
        for model in MODELS:
            for pid in NEW4:
                neu_vecs, n, data0 = build_neu_vecs(pid, model, win_ms)
                if neu_vecs is None:
                    continue
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
