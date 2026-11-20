"""
compute_orig10_win500_bymodel_pca20.py
====================================
Convo equivalent of podcast/compute_podcast_perchunk_sweep_win200_bymodel_pca20.py:
ORIG10 at their established best window (500ms), chunk=15, per-chunk bootstrap
test, each of 9 models reported SEPARATELY, semantic side PCA-reduced to a
FIXED 20 components (mirrors the podcast test that showed a dramatic reversal
toward curvature winning once semantic PCA was fixed at 20).

Prints both the no-PCA (raw, full-dim) and PCA20 result for each model, in
one pass (neural side computed once per patient, reused for both).
"""
import sys, os, numpy as np, h5py, scipy.sparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import pearsonr, ttest_rel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)

CHUNK = 15
WIN_MS = 500
PCA_N = 20
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"

ORIG10 = ["PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60", "PTYFA_task25",
          "PTYFC_task28", "PTYFF_task17", "PTYFG_task18", "PTYFI_task81", "PTYFK_task40"]


def fit_pca_fixed(X, n_components):
    X = np.asarray(X, dtype=np.float64)
    max_pc = min(X.shape[1], X.shape[0] - 1, n_components)
    if X.std() == 0 or max_pc < 2:
        return None
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit(Xs)
    return pca.transform(Xs)[:, :max_pc]


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


def build_neu_vecs_convo(pid, win_ms):
    paths = build_paths(pid, "gpt2")
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, "gpt2")
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


def get_neural_data(pid):
    neu_vecs, n, data0 = build_neu_vecs_convo(pid, WIN_MS)
    if neu_vecs is None:
        return None
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 50:
        return None
    Y_neu, _ = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        return None
    c_neu = trajectory_curvature(Y_neu)[valid]
    d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]
    return n, valid, c_neu, d_neu


def get_semantic_data(pid, model, n, valid):
    data_m = load_patient(pid, PATIENTS[pid], REGION, model)
    if data_m is None:
        return None
    emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
    n_layers = emb_mm.shape[0]
    c_raw_layers, d_raw_layers = [], []
    c_pca_layers, d_pca_layers = [], []
    for L in range(n_layers):
        emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
        c_raw_layers.append(trajectory_curvature(emb_L)[valid])
        d_raw_layers.append(trajectory_displacement(emb_L)[:n - 2][valid])
        emb_L_pca = fit_pca_fixed(emb_L, PCA_N)
        if emb_L_pca is not None:
            c_pca_layers.append(trajectory_curvature(emb_L_pca)[valid])
            d_pca_layers.append(trajectory_displacement(emb_L_pca)[:n - 2][valid])
    del emb_mm
    if not c_raw_layers or not c_pca_layers:
        return None
    return (np.mean(c_raw_layers, axis=0), np.mean(d_raw_layers, axis=0),
            np.mean(c_pca_layers, axis=0), np.mean(d_pca_layers, axis=0))


def bootstrap_test(df, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    patients = df.patient.unique()
    rc, _ = pearsonr(df[df.metric == "curv"].sem_z, df[df.metric == "curv"].neu_z)
    rd, _ = pearsonr(df[df.metric == "disp"].sem_z, df[df.metric == "disp"].neu_z)
    obs_diff = rc - rd
    boot_diffs = []
    for _ in range(n_boot):
        samp_pids = rng.choice(patients, size=len(patients), replace=True)
        boot_df = pd.concat([df[df.patient == p] for p in samp_pids])
        bc = boot_df[boot_df.metric == "curv"]; bd = boot_df[boot_df.metric == "disp"]
        if len(bc) < 3 or len(bd) < 3:
            continue
        rc_b, _ = pearsonr(bc.sem_z, bc.neu_z)
        rd_b, _ = pearsonr(bd.sem_z, bd.neu_z)
        boot_diffs.append(rc_b - rd_b)
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_boot = 2 * min(np.mean(boot_diffs > 0), np.mean(boot_diffs < 0))
    return rc, rd, obs_diff, ci_lo, ci_hi, p_boot


def build_df(c_pairs, d_pairs, pid):
    rows = []
    c_sem_c = zscore([p[0] for p in c_pairs]); c_neu_c = zscore([p[1] for p in c_pairs])
    d_sem_c = zscore([p[0] for p in d_pairs]); d_neu_c = zscore([p[1] for p in d_pairs])
    for i in range(len(c_pairs)):
        rows.append((pid, "curv", c_sem_c[i], c_neu_c[i]))
    for i in range(len(d_pairs)):
        rows.append((pid, "disp", d_sem_c[i], d_neu_c[i]))
    return rows


def main():
    neural = {}
    for pid in ORIG10:
        res = get_neural_data(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        neural[pid] = res
        print(f"  {pid} (win={WIN_MS}ms): n_words={res[0]}", flush=True)

    print(f"\nORIG10 (win={WIN_MS}ms, chunk={CHUNK}) results BY MODEL -- raw vs fixed-{PCA_N}-PC semantic PCA:")
    for model in MODELS:
        rows_raw, rows_pca = [], []
        for pid, (n, valid, c_neu, d_neu) in neural.items():
            sem = get_semantic_data(pid, model, n, valid)
            if sem is None:
                continue
            c_raw, d_raw, c_pca, d_pca = sem
            cp_raw = _chunk_pairs(c_raw, c_neu, CHUNK); dp_raw = _chunk_pairs(d_raw, d_neu, CHUNK)
            cp_pca = _chunk_pairs(c_pca, c_neu, CHUNK); dp_pca = _chunk_pairs(d_pca, d_neu, CHUNK)
            if len(cp_raw) < 3 or len(dp_raw) < 3 or len(cp_pca) < 3 or len(dp_pca) < 3:
                continue
            rows_raw += build_df(cp_raw, dp_raw, pid)
            rows_pca += build_df(cp_pca, dp_pca, pid)

        df_raw = pd.DataFrame(rows_raw, columns=["patient", "metric", "sem_z", "neu_z"])
        df_pca = pd.DataFrame(rows_pca, columns=["patient", "metric", "sem_z", "neu_z"])
        if len(df_raw) < 10 or len(df_pca) < 10:
            print(f"  {model:14s}  SKIP too few chunks")
            continue

        rc, rd, diff, lo, hi, p = bootstrap_test(df_raw)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {model:14s} [RAW]   n={len(df_raw[df_raw.metric=='curv']):5d}  r_curv={rc:+.4f}  r_disp={rd:+.4f}  "
              f"diff={diff:+.4f}  p={p:.4f} {sig}", flush=True)

        rc2, rd2, diff2, lo2, hi2, p2 = bootstrap_test(df_pca)
        sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "ns"
        print(f"  {model:14s} [PCA{PCA_N}] n={len(df_pca[df_pca.metric=='curv']):5d}  r_curv={rc2:+.4f}  r_disp={rd2:+.4f}  "
              f"diff={diff2:+.4f}  p={p2:.4f} {sig2}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
