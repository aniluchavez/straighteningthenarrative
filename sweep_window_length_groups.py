"""
sweep_window_length_groups.py
================================
Sweeps the spike-window length (WIN_MS) to see whether the curvature vs
displacement dominance pattern (original 10 = curvature, new 5 = displacement)
is specific to the 500ms default, or holds/reverses at other window lengths.

Rebuilds neu_vecs manually with variable WIN_MS (reusing load_patient's other
logic), computes chunk=15 curvature/displacement correlation with a quick
permutation null, for a few representative patients per group.
"""
import sys, os, numpy as np, h5py, scipy.sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.stats import pearsonr
from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)

WIN_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000]
CHUNK = 15
N_PERM = 100
MODEL = 'gpt2-medium'
REGION = 'hippocampus'

ORIG_REPS = []   # already computed: PTYFI_task81, PTYFA_task25
NEW_REPS  = ['PTYFS_task95']  # PTYFR_task91 already computed; PTYFU swapped for PTYFS

rng = np.random.default_rng(0)


def chunk_r(x, y, cs, min_chunks=3):
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


def build_neu_vecs(pid, win_ms):
    paths = build_paths(pid, MODEL)
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, MODEL)  # for words/onset/is_speaker alignment
    n = data0['n_words']

    with h5py.File(paths['spikes'], 'r') as f:
        qual_key = 'qual' if 'qual' in f else 'quality'
        chan_key = 'chan' if 'chan' in f else 'channelIds'
        spikes = scipy.sparse.csr_matrix((
            f['spikes']['data'][()], f['spikes']['ir'][()], f['spikes']['jc'][()]
        )).toarray()
        qual = np.asarray(f[qual_key][()]).ravel()
        chan = np.asarray(f[chan_key][()]).ravel()
    T = spikes.shape[0]
    neuron_idx = [
        idx for lo, hi in preg[REGION]
        for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()
    ]
    reg_sp = spikes[:, neuron_idx]

    # need onset + is_speaker from the xlsx directly (same as load_patient)
    import pandas as pd
    xl = pd.ExcelFile(paths['xlsx'])
    sheet = 'Sheet1' if 'Sheet1' in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(paths['xlsx'], sheet_name=sheet, keep_default_na=False)
    df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
    df = df.dropna(subset=['onset']).reset_index(drop=True)
    n = min(n, len(df))
    is_spk1 = df['Speaker1'].astype(str).str.strip() != ''

    vecs = []
    for wi in range(n):
        onset_ms = int(df.iloc[wi]['onset'])
        t0 = max(0, onset_ms - SPEAK_OFFSET_MS) if is_spk1.iloc[wi] else max(0, onset_ms + LISTEN_OFFSET_MS)
        t1 = min(T, t0 + win_ms)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n, data0


def main():
    print(f'{"pid":16s} {"WIN_MS":>6s} {"curv":>8s} {"c_z":>6s}  {"disp":>8s} {"d_z":>6s}')
    for pid in ORIG_REPS + NEW_REPS:
        for win_ms in WIN_SIZES:
            neu_vecs, n, data0 = build_neu_vecs(pid, win_ms)
            valid = build_valid_mask(n, data0['surp'][:n], data0['sent_pos'][:n], data0['is_turn'][:n])
            Y_neu, _ = fit_neural_pca(neu_vecs, n)
            if Y_neu is None:
                print(f'{pid:16s} {win_ms:6d}   -- no PCA --')
                continue
            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

            emb_mm = np.load(data0['emb_path'], mmap_mode='r')
            emb_L = np.array(emb_mm[6, :n], dtype=np.float64)
            del emb_mm
            c_sem = trajectory_curvature(emb_L)[valid]
            d_sem = trajectory_displacement(emb_L)[:n - 2][valid]

            r_c = chunk_r(c_sem, c_neu, CHUNK)
            r_d = chunk_r(d_sem, d_neu, CHUNK)

            null_c, null_d = [], []
            for _ in range(N_PERM):
                perm = rng.permutation(len(c_neu))
                null_c.append(chunk_r(c_sem, c_neu[perm], CHUNK))
                perm2 = rng.permutation(len(d_neu))
                null_d.append(chunk_r(d_sem, d_neu[perm2], CHUNK))
            nc_mu, nc_sd = np.nanmean(null_c), np.nanstd(null_c)
            nd_mu, nd_sd = np.nanmean(null_d), np.nanstd(null_d)
            zc = (r_c - nc_mu) / nc_sd if nc_sd > 0 else np.nan
            zd = (r_d - nd_mu) / nd_sd if nd_sd > 0 else np.nan

            print(f'{pid:16s} {win_ms:6d} {r_c:8.4f} {zc:6.2f}  {r_d:8.4f} {zd:6.2f}')


if __name__ == '__main__':
    main()
