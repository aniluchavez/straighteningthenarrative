"""
compute_podcast_window_sweep.py
==================================
Same window-length sweep as compute_window_length_sweep.py (convo), but for
podcast patients: sweep WIN_MS, compute chunk=15 curvature/displacement
correlation with a real permutation null, gpt2-large (cleaned embeddings).

Saves: results/podcast_window_length_sweep.json
Structure: { pid: { win_ms: {curv, disp, curv_z, disp_z} } }
"""
import sys, os, json, numpy as np, h5py, scipy.sparse, pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from scipy.stats import pearsonr
from run_geometry_analysis import (
    PATIENTS, ACCEPTED_QUAL, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions

WIN_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000]
CHUNK = 15
N_PERM = 200
MODEL = 'llama-3.1-8b'
REGION = 'hippocampus'
PODCAST_ROOT = '/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison'

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


def find_transcript(pid):
    import glob
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, 'transcript', '*.xlsx'))
             if not os.path.basename(f).startswith(('._', '~$'))]
    return cands[0]


def find_spikes(pid):
    import glob
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, 'spikes', '*_spikes.mat'))
             if not os.path.basename(f).startswith(('._', '~$'))]
    return cands[0]


def build_neu_vecs(pid, win_ms, data0):
    preg = _convo_regions(pid)
    mat_path = find_spikes(pid)
    with h5py.File(mat_path, 'r') as f:
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

    tx_path = find_transcript(pid)
    df = pd.read_excel(tx_path, sheet_name='Sheet1')
    df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
    n = min(data0['n_words'], len(df))

    vecs = []
    for wi in range(n):
        onset_ms = df.iloc[wi]['onset']
        onset_ms = int(onset_ms) if np.isfinite(onset_ms) else 0
        t0 = max(0, onset_ms + LISTEN_OFFSET_MS)
        t1 = min(T, t0 + win_ms)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n


def main():
    out_path = os.path.join(BASE_DIR, 'results', 'podcast_window_length_sweep.json')
    results = {}
    for pid in PODCAST_PATIENTS:
        data0 = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data0 is None:
            print(f'{pid}: SKIP -- no data')
            continue
        results[pid] = {}
        print(f'\n{pid}')
        for win_ms in WIN_SIZES:
            neu_vecs, n = build_neu_vecs(pid, win_ms, data0)
            valid = build_valid_mask(n, data0['surp'][:n], data0['sent_pos'][:n], data0['is_turn'][:n])
            if valid.sum() < 20:
                continue
            Y_neu, _ = fit_neural_pca(neu_vecs, n)
            if Y_neu is None:
                continue
            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

            emb_mm = np.load(data0['emb_path'], mmap_mode='r')
            emb_L = np.array(emb_mm[emb_mm.shape[0] // 2, :n], dtype=np.float64)
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

            results[pid][win_ms] = dict(curv=r_c, disp=r_d, curv_z=zc, disp_z=zd)
            print(f'  win={win_ms:5d}  curv={r_c:+.4f}(z={zc:5.2f})  disp={r_d:+.4f}(z={zd:5.2f})')

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
