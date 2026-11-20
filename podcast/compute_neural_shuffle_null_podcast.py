import sys, os
sys.path.insert(0, "/scratch/aniluchavez/straighteningthenarrative")
sys.path.insert(0, "/scratch/aniluchavez/straighteningthenarrative/podcast")
import numpy as np
import h5py, scipy.sparse
from run_geometry_analysis import trajectory_curvature, build_valid_mask, fit_neural_pca
from compute_podcast_magnitude_isc import load_words_df, find_spikes, ONSET_PAD_MS, OFFSET_PAD_MS, REGION
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, _convo_regions

N_PERM = 200
rng = np.random.default_rng(0)

def build_neu_vecs(pid, df):
    preg = _convo_regions(pid)
    with h5py.File(find_spikes(pid), "r") as f:
        qual_key = "qual" if "qual" in f else "quality"
        chan_key = "chan" if "chan" in f else "channelIds"
        spikes = scipy.sparse.csr_matrix((
            f["spikes"]["data"][()], f["spikes"]["ir"][()], f["spikes"]["jc"][()]
        )).toarray()
        qual = np.asarray(f[qual_key][()]).ravel()
        chan = np.asarray(f[chan_key][()]).ravel()
    ACCEPTED_QUAL = (4, 5)
    T = spikes.shape[0]
    neuron_idx = [
        idx for lo, hi in preg[REGION]
        for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()
    ]
    reg_sp = spikes[:, neuron_idx]
    n = len(df)
    vecs = []
    for wi in range(n):
        onset_ms = df.iloc[wi]["onset"]; offset_ms = df.iloc[wi]["offset"]
        onset_ms = int(onset_ms) if np.isfinite(onset_ms) else 0
        if np.isfinite(offset_ms) and offset_ms > onset_ms:
            offset_ms = int(offset_ms)
        else:
            offset_ms = onset_ms + 100
        t0 = max(0, onset_ms + ONSET_PAD_MS)
        t1 = min(T, offset_ms + OFFSET_PAD_MS)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs)


curv_means, null_means = [], []
for pid in PODCAST_PATIENTS:
    data0 = load_podcast_patient(pid, region="hippocampus", model_tag="gpt2", context_words=200)
    if data0 is None:
        print(f"  {pid}: SKIP")
        continue
    df = load_words_df(pid)
    neu_vecs = build_neu_vecs(pid, df)
    n = data0["n_words"]
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        continue
    c = trajectory_curvature(Y_neu)[valid]
    cm = np.mean(c)
    perm_means = []
    for _ in range(N_PERM):
        Y_shuf = Y_neu[rng.permutation(n)]
        perm_means.append(np.mean(trajectory_curvature(Y_shuf)[valid]))
    nm = np.mean(perm_means)
    curv_means.append(cm); null_means.append(nm)
    print(f"  {pid}: curv={cm:.2f}  null={nm:.2f}  diff={cm-nm:+.2f}  n_pcs={n_pcs}", flush=True)

print(f"\nPodcast neural pooled: curv={np.mean(curv_means):.2f}  null={np.mean(null_means):.2f}  diff={np.mean(curv_means)-np.mean(null_means):+.2f}")
