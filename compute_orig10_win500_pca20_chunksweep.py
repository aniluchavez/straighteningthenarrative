"""
compute_orig10_win500_pca20_chunksweep.py
====================================
ORIG10 at win=500ms, semantic side PCA-reduced to a FIXED 20 components,
averaged across all 9 models (matching the "official" all-models-averaged
convention), sweeping CHUNK_SIZES with the per-chunk bootstrap test.

Tests whether the dramatic "PCA20 makes curvature win" pattern (confirmed
per-model at chunk=15) holds across chunk sizes 3-30, or is specific to
chunk=15.
"""
import sys, os, numpy as np, h5py, scipy.sparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)

CHUNK_SIZES = [3, 5, 8, 10, 15, 20, 25, 30]
WIN_MS = 500
PCA_N = 20
MODELS = ["gpt2", "gpt2-medium", "bert-base", "roberta-base", "opt-350m",
          "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
REGION = "hippocampus"

ORIG10 = ["PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60", "PTYFA_task25",
          "PTYFC_task28", "PTYFF_task17", "PTYFG_task18", "PTYFI_task81", "PTYFK_task40"]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_perword_orig10_pca20")
os.makedirs(CACHE_DIR, exist_ok=True)


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


def get_perword_data_pca20(pid):
    cache_path = os.path.join(CACHE_DIR, f"{pid}.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["c_sem"], z["d_sem"], z["c_neu"], z["d_neu"]

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

    c_sem_models, d_sem_models = [], []
    for model in MODELS:
        data_m = load_patient(pid, PATIENTS[pid], REGION, model)
        if data_m is None:
            continue
        emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
        n_layers = emb_mm.shape[0]
        c_layers, d_layers = [], []
        for L in range(n_layers):
            emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
            emb_L_pca = fit_pca_fixed(emb_L, PCA_N)
            if emb_L_pca is None:
                continue
            c_layers.append(trajectory_curvature(emb_L_pca)[valid])
            d_layers.append(trajectory_displacement(emb_L_pca)[:n - 2][valid])
        del emb_mm
        if not c_layers:
            continue
        c_sem_models.append(np.mean(c_layers, axis=0))
        d_sem_models.append(np.mean(d_layers, axis=0))
    if not c_sem_models:
        return None
    c_sem = np.mean(c_sem_models, axis=0)
    d_sem = np.mean(d_sem_models, axis=0)
    np.savez(cache_path, c_sem=c_sem, d_sem=d_sem, c_neu=c_neu, d_neu=d_neu)
    return c_sem, d_sem, c_neu, d_neu


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


def main():
    perword = {}
    for pid in ORIG10:
        res = get_perword_data_pca20(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        perword[pid] = res
        print(f"  {pid}: n_words={len(res[0])}", flush=True)

    print(f"\nORIG10 (win={WIN_MS}ms, ALL 9 MODELS avg, PCA{PCA_N}) results by chunk size:")
    for cs in CHUNK_SIZES:
        rows = []
        for pid, (c_sem, d_sem, c_neu, d_neu) in perword.items():
            c_pairs = _chunk_pairs(c_sem, c_neu, cs)
            d_pairs = _chunk_pairs(d_sem, d_neu, cs)
            if len(c_pairs) < 3 or len(d_pairs) < 3:
                continue
            c_sem_c = zscore([p[0] for p in c_pairs]); c_neu_c = zscore([p[1] for p in c_pairs])
            d_sem_c = zscore([p[0] for p in d_pairs]); d_neu_c = zscore([p[1] for p in d_pairs])
            for i in range(len(c_pairs)):
                rows.append((pid, "curv", c_sem_c[i], c_neu_c[i]))
            for i in range(len(d_pairs)):
                rows.append((pid, "disp", d_sem_c[i], d_neu_c[i]))
        df = pd.DataFrame(rows, columns=["patient", "metric", "sem_z", "neu_z"])
        if len(df) < 10:
            print(f"  chunk={cs:3d}  SKIP too few chunks")
            continue
        rc, rd, diff, lo, hi, p = bootstrap_test(df)
        n_c = (df.metric == "curv").sum()
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  chunk={cs:3d}  n_chunks={n_c:5d}  r_curv={rc:+.4f}  r_disp={rd:+.4f}  "
              f"diff={diff:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  p={p:.4f} {sig}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
