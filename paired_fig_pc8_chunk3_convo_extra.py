"""
paired_fig_pc8_chunk3_convo.py
=================================
Conversation-dataset counterpart to podcast/paired_fig_pc8_chunk3.py:
fixed top-8 PCA (matched, both sides), dur80/40 window, raw spike counts,
3-word chunks, all convo patients, curvature/displacement/position per
model per patient (averaged across layers).
"""
import os, sys, json
import numpy as np
import pandas as pd
import h5py, scipy.sparse
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import (
    trajectory_curvature, trajectory_displacement, ACCEPTED_QUAL,
    build_valid_mask, build_paths, PATIENTS, get_sent_positions, MIN_PAIRS,
)

RESULTS = os.path.join(BASE_DIR, "results")
REGION = "hippocampus"
K = 8
CHUNK_SIZE = 3
ONSET_PAD_MS = 80
OFFSET_PAD_MS = 40

MODELS = ["gpt2-large", "gpt2-xl"]


def pca_fixed(X, k=K):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3 or X.shape[1] < 2: return None
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k)
    if max_pc < 2: return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def chunk_r(x, y, chunk_size=CHUNK_SIZE, min_chunks=3):
    fin = np.isfinite(x) & np.isfinite(y)
    n = len(x)
    xa, ya = [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        f = fin[sl]
        if f.sum() < chunk_size // 2: continue
        xa.append(np.mean(x[sl][f])); ya.append(np.mean(y[sl][f]))
    if len(xa) < min_chunks: return float("nan")
    r, _ = pearsonr(xa, ya); return float(r)


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
        vecs.append(cnt)  # raw count, matches podcast convention
    return np.array(vecs)


print("Building per-patient neural features (convo, dur80/40, count)...", flush=True)
neu_cache = {}
for pid, regions in PATIENTS.items():
    if REGION not in regions:
        continue
    try:
        df = load_words_onoff_convo(pid)
    except Exception:
        continue
    paths = build_paths(pid, MODELS[0])
    if not os.path.exists(paths["cache"]):
        continue
    emb_shape = np.load(paths["cache"], mmap_mode="r").shape
    n0 = min(emb_shape[1], len(df))
    df = df.iloc[:n0].reset_index(drop=True)
    onset_ms = df["onset"].values[:n0]
    offset_ms = df["offset"].values[:n0]
    neu_vecs = build_neu_vecs_dur8040(pid, regions[REGION], onset_ms, offset_ms)
    if neu_vecs is None:
        continue
    neu_cache[pid] = dict(neu_vecs=neu_vecs, n0=n0, onset_ms=onset_ms, offset_ms=offset_ms)
print(f"  {len(neu_cache)} patients", flush=True)

per_model_patient = {m: {} for m in MODELS}
for model in MODELS:
    print(f"=== {model} ===", flush=True)
    for pid in neu_cache:
        paths = build_paths(pid, model)
        if not os.path.exists(paths["cache"]) or not os.path.exists(paths["surprisal"]):
            continue
        n = min(neu_cache[pid]["n0"], np.load(paths["cache"], mmap_mode="r").shape[1])
        try:
            df = load_words_onoff_convo(pid)
        except Exception:
            continue
        df = df.iloc[:n].reset_index(drop=True)

        surp = np.full(n, np.nan)
        for _, row in pd.read_csv(paths["surprisal"]).iterrows():
            wi = int(row["RowIndex"])
            if wi < n:
                surp[wi] = float(row["surprisal"])
        sent_pos = get_sent_positions(df)[:n]
        is_speaker = df["Speaker1"].astype(str).str.strip().str.len().gt(0).values[:n]
        valid = build_valid_mask(n, surp, sent_pos, ~is_speaker if is_speaker.dtype == bool else np.zeros(n, dtype=bool))
        if valid.sum() < MIN_PAIRS:
            continue

        Y_neu = pca_fixed(neu_cache[pid]["neu_vecs"][:n])
        if Y_neu is None: continue
        c_neu_full = trajectory_curvature(Y_neu)
        d_neu_full = trajectory_displacement(Y_neu)
        p_neu_full = Y_neu[:, 0]
        mc = min(len(c_neu_full), len(valid)); c_neu = c_neu_full[:mc][valid[:mc]]
        md = min(len(d_neu_full), len(valid)); d_neu = d_neu_full[:md][valid[:md]]
        mp = min(len(p_neu_full), len(valid)); p_neu = p_neu_full[:mp][valid[:mp]]

        emb_mm = np.load(paths["cache"], mmap_mode="r")
        n_layers = emb_mm.shape[0]
        curv_L, disp_L, pos_L = [], [], []
        for li in range(n_layers):
            emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
            Y_sem = pca_fixed(emb_l)
            if Y_sem is None: continue
            c_sem_full = trajectory_curvature(Y_sem)
            d_sem_full = trajectory_displacement(Y_sem)
            p_sem_full = Y_sem[:, 0]

            mcc = min(len(c_sem_full), len(valid)); c_sem = c_sem_full[:mcc][valid[:mcc]]
            mdd = min(len(d_sem_full), len(valid)); d_sem = d_sem_full[:mdd][valid[:mdd]]
            mpp = min(len(p_sem_full), len(valid)); p_sem = p_sem_full[:mpp][valid[:mpp]]

            nc = min(len(c_sem), len(c_neu)); curv_L.append(chunk_r(c_neu[:nc], c_sem[:nc]))
            nd = min(len(d_sem), len(d_neu)); disp_L.append(chunk_r(d_neu[:nd], d_sem[:nd]))
            npz = min(len(p_sem), len(p_neu)); pos_L.append(chunk_r(p_neu[:npz], p_sem[:npz]))
        del emb_mm
        if curv_L:
            per_model_patient[model][pid] = dict(
                curv=float(np.nanmean(curv_L)), disp=float(np.nanmean(disp_L)), pos=float(np.nanmean(pos_L)))
        print(f"  {pid}: OK", flush=True)

json.dump(per_model_patient, open(os.path.join(RESULTS, "paired_fig_pc8_chunk3_convo_extra.json"), "w"))
print("\nSaved results/paired_fig_pc8_chunk3_convo_extra.json")

print("\n=== Per-model mean (convo, chunk=3, PC8) ===")
for model in MODELS:
    vals = per_model_patient[model]
    if not vals: continue
    curv = np.array([v["curv"] for v in vals.values()])
    disp = np.array([v["disp"] for v in vals.values()])
    pos = np.array([v["pos"] for v in vals.values()])
    print(f"  {model:14s} curv={np.nanmean(curv):+.4f}  disp={np.nanmean(disp):+.4f}  pos={np.nanmean(pos):+.4f}  (n={len(vals)})")
