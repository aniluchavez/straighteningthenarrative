"""
test_window_overlap_smoothing.py
===================================
Tests the mechanism behind the curvature-vs-displacement window-length
crossover: does growing window length increasingly "smooth" the neural
trajectory's direction (consecutive steps become more self-similar, since
they share more of the same underlying spikes), and does this smoothing
kick in at a shorter absolute window for new patients than original ones?

For each window length: mean cosine similarity between consecutive
unit-normalized neural step vectors (steps[t] . steps[t+1] / |.||.|),
averaged across words, per patient. Higher = more directional smoothing/blur.
"""
import sys, os, numpy as np, h5py, scipy.sparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    build_valid_mask, fit_neural_pca, load_patient,
)

WIN_SIZES = [100, 200, 300, 500, 700, 1000, 1500, 2000]
REGION = 'hippocampus'

ORIG10 = ['PTYEU_task147', 'PTYEV_task37', 'PTYEY_task86', 'PTYEZ_task60', 'PTYFA_task25',
          'PTYFC_task28', 'PTYFF_task17', 'PTYFG_task18', 'PTYFI_task81', 'PTYFK_task40']
NEW4 = ['PTYFM_task104', 'PTYFP_task88', 'PTYFR_task91', 'PTYFS_task95']


def build_neu_vecs(pid, win_ms):
    paths = build_paths(pid, 'gpt2')
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, 'gpt2')
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


def mean_step_cosine_similarity(Y_neu, valid_mask):
    steps = Y_neu[1:] - Y_neu[:-1]
    norms = np.linalg.norm(steps, axis=1)
    ok = norms > 1e-9
    unit_steps = np.zeros_like(steps)
    unit_steps[ok] = steps[ok] / norms[ok, None]
    cos = np.sum(unit_steps[:-1] * unit_steps[1:], axis=1)
    # align to same indexing as curvature (valid mask over n-2 length)
    cos = cos[:len(valid_mask)]
    return np.nanmean(cos[valid_mask])


def main():
    print(f'{"pid":16s} ' + '  '.join(f'{w:>6d}' for w in WIN_SIZES))
    group_results = {'orig': [], 'new': []}
    for pid in ORIG10 + NEW4:
        row = []
        for win_ms in WIN_SIZES:
            neu_vecs, n, data0 = build_neu_vecs(pid, win_ms)
            if neu_vecs is None:
                row.append(np.nan); continue
            valid = build_valid_mask(n, data0['surp'][:n], data0['sent_pos'][:n], data0['is_turn'][:n])
            Y_neu, _ = fit_neural_pca(neu_vecs, n)
            if Y_neu is None:
                row.append(np.nan); continue
            cos_sim = mean_step_cosine_similarity(Y_neu, valid)
            row.append(cos_sim)
        tag = 'new' if pid in NEW4 else 'orig'
        group_results[tag].append(row)
        print(f'{pid:16s} ' + '  '.join(f'{v:6.3f}' if not np.isnan(v) else '   nan' for v in row))

    print()
    print('GROUP MEANS (mean cosine similarity between consecutive step directions):')
    print(f'{"win_ms":>7s} {"orig10":>8s} {"new4":>8s}')
    orig_arr = np.array(group_results['orig'])
    new_arr = np.array(group_results['new'])
    for i, w in enumerate(WIN_SIZES):
        print(f'{w:7d} {np.nanmean(orig_arr[:,i]):8.4f} {np.nanmean(new_arr[:,i]):8.4f}')


if __name__ == '__main__':
    main()
