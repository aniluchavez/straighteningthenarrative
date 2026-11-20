"""
compute_cross_space_layers_convo_pca20_dur8040.py
====================================================
Conversation-dataset counterpart to
podcast/compute_cross_space_layers_podcast_pca20_dur8040.py: same dur80/40
word-duration-locked, rate-normalized neural window and PCA20-reduced LLM
embeddings, but on the free-conversation (ConvoDATAS) patients instead of
podcast-listening patients.

Output: results/cross_space_layers_convo_pca20_dur8040_{model}_hippocampus.json
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import h5py, scipy.sparse
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import (
    trajectory_curvature, trajectory_displacement, ACCEPTED_QUAL,
    build_valid_mask, fit_neural_pca, sem_pca_reduce, build_paths, PATIENTS,
    get_sent_positions, get_log_freq, MIN_PAIRS,
)

RESULTS = os.path.join(BASE_DIR, "results")
REGION = "hippocampus"
CHUNK_SIZE = 15
N_PERM = 200
ONSET_PAD_MS = 80
OFFSET_PAD_MS = 40

MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]


def _chunk_r(x, y, chunk_size, min_chunks=5):
    n = len(x)
    xa, ya = [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        fin = np.isfinite(x[sl]) & np.isfinite(y[sl])
        if fin.sum() < chunk_size // 2:
            continue
        xa.append(np.mean(x[sl][fin])); ya.append(np.mean(y[sl][fin]))
    xa, ya = np.array(xa), np.array(ya)
    if len(xa) < min_chunks:
        return float("nan"), float("nan")
    r, p = pearsonr(xa, ya)
    return float(r), float(p)


def _safe_r(x, y, min_n=15):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return float("nan"), float("nan")
    r, p = pearsonr(x[m], y[m])
    return float(r), float(p)


def load_words_onoff_convo(pid):
    paths = build_paths(pid, "gpt2")
    xl = pd.ExcelFile(paths["xlsx"])
    sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(paths["xlsx"], sheet_name=sheet, keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    df = df.dropna(subset=["onset"]).reset_index(drop=True)
    return df


def build_neu_vecs_dur8040(pid, chan_ranges, onset_ms, offset_ms):
    short = pid.split("_")[0].upper()[2:]
    task = pid.split("_", 1)[1]
    spikes_path = os.path.join("/scratch/aniluchavez/ConvoDATAS/SpikesMAT", short, f"pt{short}_{task}_new_spikes.mat")
    if not os.path.exists(spikes_path):
        return None
    with h5py.File(spikes_path, "r") as f:
        spikes = scipy.sparse.csr_matrix((f["spikes"]["data"][()], f["spikes"]["ir"][()], f["spikes"]["jc"][()])).toarray()
        chan_key = "chan" if "chan" in f else "channelIds"
        qual_key = "qual" if "qual" in f else "quality"
        chan = np.asarray(f[chan_key][()]).ravel()
        qual = np.asarray(f[qual_key][()]).ravel()
    T = spikes.shape[0]
    unit_idx = [idx for lo, hi in chan_ranges
                for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()]
    if len(unit_idx) < 1:
        return None
    reg_sp = spikes[:, unit_idx]
    n_words = len(onset_ms)
    vecs = []
    for wi in range(n_words):
        if not (np.isfinite(onset_ms[wi]) and np.isfinite(offset_ms[wi])):
            vecs.append(np.zeros(len(unit_idx))); continue
        t0 = max(0, int(onset_ms[wi] + ONSET_PAD_MS))
        t1 = min(T, max(t0, int(offset_ms[wi] + OFFSET_PAD_MS)))
        cnt = reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(unit_idx))
        vecs.append(cnt / max(1, t1 - t0) * 1000)
    return np.array(vecs)


def process_model(model):
    out_path = os.path.join(RESULTS, f"cross_space_layers_convo_pca20_dur8040_{model}_{REGION}.json")
    print(f"\n{'='*60}\n  Model: {model}\n{'='*60}")

    rng = np.random.default_rng(42)
    out = []
    for pid, regions in PATIENTS.items():
        if REGION not in regions:
            continue
        paths = build_paths(pid, model)
        if not os.path.exists(paths["cache"]) or not os.path.exists(paths["surprisal"]):
            print(f"  {pid}: missing cache/surprisal"); continue
        try:
            df = load_words_onoff_convo(pid)
        except Exception as e:
            print(f"  {pid}: xlsx load failed ({e})"); continue

        emb_shape = np.load(paths["cache"], mmap_mode="r").shape
        n = min(emb_shape[1], len(df))
        df = df.iloc[:n].reset_index(drop=True)
        onset_ms = df["onset"].values[:n]
        offset_ms = df["offset"].values[:n]

        surp = np.full(n, np.nan)
        for _, row in pd.read_csv(paths["surprisal"]).iterrows():
            wi = int(row["RowIndex"])
            if wi < n:
                surp[wi] = float(row["surprisal"])
        sent_pos = get_sent_positions(df)[:n]
        is_speaker = df["Speaker1"].astype(str).str.strip().str.len().gt(0).values[:n]

        valid = build_valid_mask(n, surp, sent_pos, ~is_speaker if is_speaker.dtype == bool else np.zeros(n, dtype=bool))
        if valid.sum() < MIN_PAIRS:
            print(f"  {pid}: too few valid ({valid.sum()})"); continue

        neu_vecs = build_neu_vecs_dur8040(pid, regions[REGION], onset_ms, offset_ms)
        if neu_vecs is None:
            print(f"  {pid}: neural windowing failed"); continue
        Y_neu, _ = fit_neural_pca(neu_vecs, n)
        if Y_neu is None:
            print(f"  {pid}: neural PCA failed"); continue

        c_neu_full = trajectory_curvature(Y_neu)
        d_neu_full = trajectory_displacement(Y_neu)
        m0 = min(len(c_neu_full), len(valid))
        c_neu = c_neu_full[:m0][valid[:m0]]
        d_neu = d_neu_full[:m0][valid[:m0]]

        emb_mm = np.load(paths["cache"], mmap_mode="r")
        n_layers = emb_mm.shape[0]

        r_curv_L, r_disp_L = [], []
        r_curv_ch, r_disp_ch = [], []
        c_sems = []
        for L in range(n_layers):
            emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
            emb_L = sem_pca_reduce(emb_L)
            c_sem_full = trajectory_curvature(emb_L)
            d_sem_full = trajectory_displacement(emb_L)
            c_sem = c_sem_full[:m0][valid[:m0]]
            d_sem = d_sem_full[:m0][valid[:m0]]
            c_sems.append(c_sem)

            rc, _ = _safe_r(c_sem, c_neu)
            rd, _ = _safe_r(d_sem, d_neu)
            r_curv_L.append(rc); r_disp_L.append(rd)

            rcc, _ = _chunk_r(c_sem, c_neu, CHUNK_SIZE)
            rdc, _ = _chunk_r(d_sem, d_neu, CHUNK_SIZE)
            r_curv_ch.append(rcc); r_disp_ch.append(rdc)
        del emb_mm

        null_curv_ch = np.full((N_PERM, n_layers), np.nan)
        for pi in range(N_PERM):
            perm = rng.permutation(len(c_neu))
            c_neu_shuf = c_neu[perm]
            for L in range(n_layers):
                null_curv_ch[pi, L] = _chunk_r(c_sems[L], c_neu_shuf, CHUNK_SIZE)[0]
        null_curv_ch_mean = np.nanmean(null_curv_ch, axis=0).tolist()
        null_curv_ch_std = np.nanstd(null_curv_ch, axis=0).tolist()

        def _peak(arr):
            vals = [r for r in arr if not np.isnan(r)]
            return max(vals) if vals else float("nan")

        print(f"  {pid}: {n_layers} layers  r_curv(word/chunk)={_peak(r_curv_L):+.4f}/{_peak(r_curv_ch):+.4f}")

        out.append(dict(
            pid=pid, n_layers=n_layers,
            r_curv_layers=r_curv_L, r_disp_layers=r_disp_L,
            r_curv_layers_ch=r_curv_ch, r_disp_layers_ch=r_disp_ch,
            null_curv_layers_ch_mean=null_curv_ch_mean, null_curv_layers_ch_std=null_curv_ch_std,
        ))

    if out:
        json.dump(out, open(out_path, "w"))
        print(f"\n  Saved {len(out)} patients -> {out_path}")
    else:
        print(f"\n  No data saved for {model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    models = [args.model] if args.model else MODELS
    for model in models:
        process_model(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
