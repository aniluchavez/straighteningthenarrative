"""
compute_podcast_perchunk_sweep.py
====================================
Podcast version of ../compute_perchunk_sweep.py: sweeps CHUNK_SIZES using
the proper per-chunk pooled-correlation + patient-cluster bootstrap test,
at podcast's own curvature-preferring window (win=100ms, established via
compute_podcast_window_scan_fast.py), single model llama-3.1-8b.
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

CHUNK_SIZES = [3, 5, 8, 10, 15, 20, 25, 30]
MODEL = "llama-3.1-8b"
REGION = "hippocampus"
WIN_MS = 200
PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"


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


def get_perword_data(pid):
    data0 = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
    if data0 is None:
        return None
    neu_vecs, n = build_neu_vecs(pid, WIN_MS, data0)
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 50:
        return None
    Y_neu, _ = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        return None
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
    for pid in PODCAST_PATIENTS:
        res = get_perword_data(pid)
        if res is None:
            print(f"  {pid}: SKIP")
            continue
        perword[pid] = res
        print(f"  {pid} (win={WIN_MS}ms): n_words={len(res[0])}", flush=True)

    print(f"\nPODCAST (win={WIN_MS}ms) results by chunk size:")
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
