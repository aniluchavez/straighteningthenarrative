"""
compute_cross_space_chunk_sweep_bestwin.py
=============================================
Same methodology as compute_cross_space_chunk_sweep.py (chunk-size sweep,
all-layers-averaged, 9 models, N_PERM=200), except neu_vecs are rebuilt at
each patient's GROUP-optimal spike-integration window for curvature
(established via the earlier corrected-layer WIN_MS sweep: orig10 peaks at
win=500ms, new4 peaks at win=300ms), instead of the fixed WIN_MS=500 default
baked into load_patient().

Checkpoints results/cross_space_chunk_sweep_bestwin.json after every model.
"""
import json, os, sys
import numpy as np
import h5py, scipy.sparse, pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N_PERM = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import (
    trajectory_curvature, trajectory_displacement,
    build_valid_mask, fit_neural_pca, load_patient,
    PATIENTS, MIN_PAIRS, build_paths, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
)

RESULTS     = os.path.join(BASE_DIR, "results")
REGION      = "hippocampus"
CHUNK_SIZES = [3, 5, 8, 10, 15, 20, 25, 30]

MODELS = [
    "gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
    "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
]

ORIG10 = {"PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60", "PTYFA_task25",
          "PTYFC_task28", "PTYFF_task17", "PTYFG_task18", "PTYFI_task81", "PTYFK_task40"}
NEW4 = {"PTYFM_task104", "PTYFP_task88", "PTYFR_task91", "PTYFS_task95"}

BEST_WIN = {}
for pid in ORIG10:
    BEST_WIN[pid] = 500
for pid in NEW4:
    BEST_WIN[pid] = 300
# any other patient (e.g. PTYFU if it ever loads): fall back to 500
DEFAULT_WIN = 500


def _pc1_series(X):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] < 2:
        return np.full(X.shape[0], np.nan)
    Xs = StandardScaler().fit_transform(X)
    pc1 = PCA(n_components=1).fit_transform(Xs)[:, 0]
    s = np.std(pc1)
    return (pc1 - pc1.mean()) / s if s > 1e-12 else np.full(len(pc1), np.nan)


def _chunk_r(x, y, chunk_size, min_chunks=3):
    n = len(x)
    xa, ya = [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        fin = np.isfinite(x[sl]) & np.isfinite(y[sl])
        if fin.sum() < chunk_size // 2:
            continue
        xa.append(np.mean(x[sl][fin]))
        ya.append(np.mean(y[sl][fin]))
    if len(xa) < min_chunks:
        return float("nan")
    r, _ = pearsonr(xa, ya)
    return float(r)


def build_neu_vecs_at_window(pid, model_for_meta, win_ms):
    paths = build_paths(pid, model_for_meta)
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, model_for_meta)
    if data0 is None:
        return None, None
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
    return np.array(vecs), data0


def main():
    out_path = os.path.join(RESULTS, "cross_space_chunk_sweep_bestwin.json")

    rng = np.random.default_rng(42)

    results = {cs: {m: {"disp": [], "curv": [], "pos": [],
                         "null_curv_mean": [], "null_curv_std": [],
                         "null_disp_mean": [], "null_disp_std": []}
                    for m in MODELS}
               for cs in CHUNK_SIZES}

    for model in MODELS:
        print(f"\n  {model}", flush=True)
        for pid, preg in PATIENTS.items():
            if REGION not in preg:
                continue
            win_ms = BEST_WIN.get(pid, DEFAULT_WIN)
            neu_vecs, data = build_neu_vecs_at_window(pid, model, win_ms)
            if neu_vecs is None:
                continue

            n       = data["n_words"]
            surp    = data["surp"]
            sp      = data["sent_pos"]
            is_turn = data["is_turn"]

            valid = build_valid_mask(n, surp, sp, is_turn)
            if valid.sum() < MIN_PAIRS:
                continue

            interior = np.arange(1, n - 1)

            Y_neu, _ = fit_neural_pca(neu_vecs, n)
            if Y_neu is None:
                continue

            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
            p_neu = _pc1_series(Y_neu)[interior][valid]

            emb_mm   = np.load(data["emb_path"], mmap_mode="r")
            n_layers = emb_mm.shape[0]

            layer_r  = {cs: {"disp": [], "curv": [], "pos": []} for cs in CHUNK_SIZES}
            c_sems, d_sems = [], []

            for L in range(n_layers):
                emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
                c_sem = trajectory_curvature(emb_L)[valid]
                d_sem = trajectory_displacement(emb_L)[:n - 2][valid]
                p_sem = _pc1_series(emb_L)[interior][valid]
                c_sems.append(c_sem)
                d_sems.append(d_sem)

                for cs in CHUNK_SIZES:
                    layer_r[cs]["curv"].append(_chunk_r(c_sem, c_neu, cs))
                    layer_r[cs]["disp"].append(_chunk_r(d_sem, d_neu, cs))
                    layer_r[cs]["pos"].append( _chunk_r(p_sem, p_neu, cs))

            del emb_mm

            for cs in CHUNK_SIZES:
                for metric in ("disp", "curv", "pos"):
                    vals = [v for v in layer_r[cs][metric] if not np.isnan(v)]
                    results[cs][model][metric].append(
                        float(np.mean(vals)) if vals else float("nan"))

            for cs in CHUNK_SIZES:
                null_curv_perms, null_disp_perms = [], []
                for _ in range(N_PERM):
                    perm = rng.permutation(len(c_neu))
                    rs_c = [_chunk_r(c_sems[L], c_neu[perm], cs) for L in range(n_layers)]
                    rs_d = [_chunk_r(d_sems[L], d_neu[perm], cs) for L in range(n_layers)]
                    null_curv_perms.append(float(np.nanmean(rs_c)))
                    null_disp_perms.append(float(np.nanmean(rs_d)))
                results[cs][model]["null_curv_mean"].append(float(np.nanmean(null_curv_perms)))
                results[cs][model]["null_curv_std"].append(float(np.nanstd(null_curv_perms)))
                results[cs][model]["null_disp_mean"].append(float(np.nanmean(null_disp_perms)))
                results[cs][model]["null_disp_std"].append(float(np.nanstd(null_disp_perms)))

            print(f"    {pid} (win={win_ms}ms): done ({n_layers} layers)", flush=True)

        out = {str(cs): results[cs] for cs in CHUNK_SIZES}
        json.dump(out, open(out_path, "w"))
        print(f"  checkpoint saved -> {out_path}", flush=True)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
