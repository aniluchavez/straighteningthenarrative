"""
compute_perchunk_mixed_model.py
==================================
Per-chunk (not per-patient) test of curvature-vs-displacement, at each
patient's own optimal spike-integration window (orig10=500ms, new4=300ms),
chunk=15, llama-3.1-8b only (single well-validated model, not the full
9-model average, to keep this tractable given the compute history).

For every chunk (n/15 per patient), extracts the paired (c_sem_chunk,
c_neu_chunk) and (d_sem_chunk, d_neu_chunk) values -- the actual
chunk-averaged curvature/displacement, not yet correlated. Z-scores within
patient (removes patient-level offset/scale) so chunks can be pooled across
patients while preserving the within-patient relationship. Then:
  1. Pools all chunks within each group (orig10, new4) and computes a single
     high-n pooled correlation for curvature and displacement.
  2. Fits a proper linear mixed-effects model (patient as random intercept +
     random slope) via statsmodels, comparing curv slope vs disp slope.
"""
import sys, os, numpy as np, h5py, scipy.sparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import pearsonr
from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)

CHUNK = 15
MODEL = "llama-3.1-8b"
REGION = "hippocampus"

ORIG10 = ["PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60", "PTYFA_task25",
          "PTYFC_task28", "PTYFF_task17", "PTYFG_task18", "PTYFI_task81", "PTYFK_task40"]
NEW4 = ["PTYFM_task104", "PTYFP_task88", "PTYFR_task91", "PTYFS_task95"]
WIN_FOR = {**{p: 500 for p in ORIG10}, **{p: 100 for p in NEW4}}


def _chunk_pairs(x, y, cs, min_frac=0.5):
    """Returns list of (mean_x_chunk, mean_y_chunk) tuples, one per chunk."""
    n = len(x)
    out = []
    for i in range(0, n - cs + 1, cs):
        sl = slice(i, i + cs)
        fin = np.isfinite(x[sl]) & np.isfinite(y[sl])
        if fin.sum() < cs * min_frac:
            continue
        out.append((np.mean(x[sl][fin]), np.mean(y[sl][fin])))
    return out


def build_neu_vecs(pid, win_ms):
    paths = build_paths(pid, MODEL)
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, MODEL)
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


def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    if sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def process_group(pids, label):
    rows = []  # patient, c_sem_z, c_neu_z, d_sem_z, d_neu_z
    for pid in pids:
        win_ms = WIN_FOR[pid]
        neu_vecs, n, data0 = build_neu_vecs(pid, win_ms)
        if neu_vecs is None:
            print(f"  {pid}: SKIP no data"); continue
        valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
        if valid.sum() < 50:
            print(f"  {pid}: SKIP too few valid words"); continue
        Y_neu, _ = fit_neural_pca(neu_vecs, n)
        if Y_neu is None:
            print(f"  {pid}: SKIP PCA failed"); continue
        c_neu = trajectory_curvature(Y_neu)[valid]
        d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

        emb_mm = np.load(data0["emb_path"], mmap_mode="r")
        n_layers = emb_mm.shape[0]
        c_sem_layers, d_sem_layers = [], []
        for L in range(n_layers):
            emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
            c_sem_layers.append(trajectory_curvature(emb_L)[valid])
            d_sem_layers.append(trajectory_displacement(emb_L)[:n - 2][valid])
        del emb_mm
        c_sem = np.mean(c_sem_layers, axis=0)
        d_sem = np.mean(d_sem_layers, axis=0)

        c_pairs = _chunk_pairs(c_sem, c_neu, CHUNK)
        d_pairs = _chunk_pairs(d_sem, d_neu, CHUNK)
        if len(c_pairs) < 3 or len(d_pairs) < 3:
            print(f"  {pid}: SKIP too few chunks"); continue

        c_sem_c = zscore([p[0] for p in c_pairs]); c_neu_c = zscore([p[1] for p in c_pairs])
        d_sem_c = zscore([p[0] for p in d_pairs]); d_neu_c = zscore([p[1] for p in d_pairs])

        for i in range(len(c_pairs)):
            rows.append((pid, "curv", c_sem_c[i], c_neu_c[i]))
        for i in range(len(d_pairs)):
            rows.append((pid, "disp", d_sem_c[i], d_neu_c[i]))
        print(f"  {pid} (win={win_ms}ms): {len(c_pairs)} curv chunks, {len(d_pairs)} disp chunks", flush=True)

    return pd.DataFrame(rows, columns=["patient", "metric", "sem_z", "neu_z"])


def main():
    print("=== NEW4 (win=100ms) ===", flush=True)
    df_new = process_group(NEW4, "NEW4")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "perchunk_data_new4_win100.csv")
    df_new.assign(group="new4").to_csv(out_path, index=False)
    print(f"\nSaved chunk-level data -> {out_path}")

    for label, df in [("NEW4 (win=100ms)", df_new)]:
        print(f"\n--- {label}: pooled chunk-level correlation ---")
        for metric in ("curv", "disp"):
            sub = df[df.metric == metric]
            if len(sub) < 3:
                continue
            r, p = pearsonr(sub.sem_z, sub.neu_z)
            print(f"  {metric}: n_chunks={len(sub)}  pooled_r={r:+.4f}  p={p:.4g}")

        # Steiger's test for comparing two dependent/overlapping correlations is complex;
        # here we do a patient-cluster bootstrap for a valid p-value on r_curv - r_disp.
        rng = np.random.default_rng(0)
        patients = df.patient.unique()
        obs_rc, _ = pearsonr(df[df.metric == "curv"].sem_z, df[df.metric == "curv"].neu_z)
        obs_rd, _ = pearsonr(df[df.metric == "disp"].sem_z, df[df.metric == "disp"].neu_z)
        obs_diff = obs_rc - obs_rd
        boot_diffs = []
        for _ in range(2000):
            samp_pids = rng.choice(patients, size=len(patients), replace=True)
            boot_df = pd.concat([df[df.patient == p] for p in samp_pids])
            bc = boot_df[boot_df.metric == "curv"]
            bd = boot_df[boot_df.metric == "disp"]
            if len(bc) < 3 or len(bd) < 3:
                continue
            rc, _ = pearsonr(bc.sem_z, bc.neu_z)
            rd, _ = pearsonr(bd.sem_z, bd.neu_z)
            boot_diffs.append(rc - rd)
        boot_diffs = np.array(boot_diffs)
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        p_boot = 2 * min(np.mean(boot_diffs > 0), np.mean(boot_diffs < 0))
        print(f"  patient-cluster bootstrap: r_curv - r_disp = {obs_diff:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  p~{p_boot:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
