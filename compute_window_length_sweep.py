"""
compute_window_length_sweep.py
=================================
Full window-length sweep: for every patient (original 10 + new 5, minus any
with currently-broken source data), sweep WIN_MS and compute chunk=15
curvature/displacement correlation with a real permutation null, for a
couple of models.

Rebuilds neu_vecs manually per window length (reusing load_patient's other
logic for words/onset/surprisal/embeddings).

Saves: results/window_length_sweep.json
Structure: { pid: { win_ms: { model: {curv, disp, curv_z, disp_z} } } }

Run:
    conda run -n gpt2_embed python3 -u compute_window_length_sweep.py
"""
import sys, os, json, numpy as np, h5py, scipy.sparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.stats import pearsonr
from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
WIN_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000]
CHUNK = 15
N_PERM = 200
MODELS = ['gpt2-medium', 'llama-3.1-8b']
REGION = 'hippocampus'

ORIG10 = ['PTYEU_task147', 'PTYEV_task37', 'PTYEY_task86', 'PTYEZ_task60', 'PTYFA_task25',
          'PTYFC_task28', 'PTYFF_task17', 'PTYFG_task18', 'PTYFI_task81', 'PTYFK_task40']
NEW5 = ['PTYFM_task104', 'PTYFP_task88', 'PTYFR_task91', 'PTYFS_task95']  # PTYFU excluded: transcript currently broken

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


def build_neu_vecs(pid, win_ms, model_for_meta):
    paths = build_paths(pid, model_for_meta)
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, model_for_meta)
    if data0 is None:
        return None, None, None
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
    out_path = os.path.join(RESULTS_DIR, 'window_length_sweep.json')
    results = {}

    for pid in ORIG10 + NEW5:
        results[pid] = {}
        print(f'\n{pid}')
        for win_ms in WIN_SIZES:
            results[pid][win_ms] = {}
            for model in MODELS:
                neu_vecs, n, data0 = build_neu_vecs(pid, win_ms, model)
                if neu_vecs is None:
                    continue
                valid = build_valid_mask(n, data0['surp'][:n], data0['sent_pos'][:n], data0['is_turn'][:n])
                if valid.sum() < 20:
                    continue
                Y_neu, _ = fit_neural_pca(neu_vecs, n)
                if Y_neu is None:
                    continue
                c_neu = trajectory_curvature(Y_neu)[valid]
                d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

                emb_mm = np.load(data0['emb_path'], mmap_mode='r')
                emb_L = np.array(emb_mm[emb_mm.shape[0] // 2, :n], dtype=np.float64)  # mid layer
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

                results[pid][win_ms][model] = dict(curv=r_c, disp=r_d, curv_z=zc, disp_z=zd)
                print(f'  win={win_ms:5d} {model:14s} curv={r_c:+.4f}(z={zc:5.2f})  disp={r_d:+.4f}(z={zd:5.2f})')

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
