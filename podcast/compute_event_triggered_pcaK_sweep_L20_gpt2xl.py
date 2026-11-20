"""
compute_event_triggered_pcaK_sweep_L20_gpt2xl.py
====================================
At gpt2-xl layer 20 (the raw-dim layer whose event-triggered curvature
shape best matched neural: r=+0.559, peak_lag=+1), sweep the semantic-side
PCA K (top-K components, standard convention) from near-full-rank down to
K=3, to find where the "neural-like" shape (peak AFTER the event, at
lag +1) breaks down into the K=20 pipeline's peak-AT-lag-0 pattern.

Efficiency: fit ONE master PCA per patient at K_MAX=1000, slice to each K
(nested/hierarchical PCA components) instead of refitting per K.
K=1000 (full rank -- rotation-invariant, curvature identical to no-PCA raw
dims) anchors the top end.

Saves: results/event_triggered_pcaK_sweep_L20_gpt2xl.json
"""
import json, os, sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis_pca20 import trajectory_curvature, fit_neural_pca, EVENT_WINDOW
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS, "event_triggered_pcaK_sweep_L20_gpt2xl.json")

MODEL = "gpt2-xl"
REGION = "hippocampus"
LAYER = 20
PCTILE_HI, PCTILE_LO = 75, 25
K_MAX = 1000
KS = [1000, 500, 300, 200, 150, 100, 75, 50, 30, 20, 15, 10, 5, 3]


def quantile_event_masks(x, hi_pctile=PCTILE_HI, lo_pctile=PCTILE_LO):
    x = np.asarray(x, dtype=float)
    x_valid = x[np.isfinite(x)]
    hi_thr = float(np.percentile(x_valid, hi_pctile))
    lo_thr = float(np.percentile(x_valid, lo_pctile))
    return x >= hi_thr, x <= lo_thr


def event_triggered_curv(c_full, n, hi_mask, lo_mask, W=EVENT_WINDOW):
    lags = list(range(-W, W + 1))
    hi = [[] for _ in lags]; lo = [[] for _ in lags]
    for j in range(n):
        is_hi, is_lo = hi_mask[j], lo_mask[j]
        if not (is_hi or is_lo):
            continue
        for ki, k in enumerate(lags):
            ci = (j + k) - 2
            if 0 <= ci < len(c_full):
                (hi if is_hi else lo)[ki].append(c_full[ci])
    def _m(lst):
        return float(np.mean(lst)) if lst else float("nan")
    return [_m(x) for x in hi], [_m(x) for x in lo]


def main():
    per_patient_neu = {}
    per_patient_masks = {}
    per_patient_master = {}

    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
        if data is None or data["neu_vecs"] is None:
            continue
        n = data["n_words"]
        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu_full = trajectory_curvature(Y_neu)[:n - 2]
        hi_mask, lo_mask = quantile_event_masks(data["surp"][:n])
        hi_neu, lo_neu = event_triggered_curv(c_neu_full, n, hi_mask, lo_mask)
        per_patient_neu[pid] = (hi_neu, lo_neu)
        per_patient_masks[pid] = (hi_mask, lo_mask, n)

        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if LAYER >= emb_mm.shape[0]:
            continue
        emb_L = np.array(emb_mm[LAYER, :n], dtype=np.float64)
        del emb_mm
        Xs = StandardScaler().fit_transform(emb_L)
        max_pc = min(Xs.shape[1], Xs.shape[0] - 1, K_MAX)
        pca = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit(Xs)
        per_patient_master[pid] = pca.transform(Xs)
        print(f"  {pid}: master PCA fit, max_pc={max_pc}", flush=True)

    pids = [p for p in per_patient_neu if p in per_patient_master]
    hi_neu_mu = np.nanmean([per_patient_neu[p][0] for p in pids], axis=0)
    lo_neu_mu = np.nanmean([per_patient_neu[p][1] for p in pids], axis=0)
    neu_diff = np.array(hi_neu_mu) - np.array(lo_neu_mu)
    print(f"\nneural diff curve (hi-lo) by lag: {np.round(neu_diff, 3).tolist()}", flush=True)

    k_results = {}
    for K in KS:
        hi_list, lo_list = [], []
        for pid in pids:
            master = per_patient_master[pid]
            kk = min(K, master.shape[1])
            hi_mask, lo_mask, n = per_patient_masks[pid]
            c_sem_full = trajectory_curvature(master[:, :kk])
            hi_sem, lo_sem = event_triggered_curv(c_sem_full, n, hi_mask, lo_mask)
            hi_list.append(hi_sem); lo_list.append(lo_sem)
        hi_mu = np.nanmean(hi_list, axis=0)
        lo_mu = np.nanmean(lo_list, axis=0)
        sem_diff = hi_mu - lo_mu
        r_shape, _ = pearsonr(sem_diff, neu_diff)
        peak_lag = int(np.argmax(hi_mu)) - EVENT_WINDOW
        k_results[K] = dict(hi_sem=hi_mu.tolist(), lo_sem=lo_mu.tolist(),
                             shape_r_vs_neural=float(r_shape), peak_lag=peak_lag)
        print(f"  K={K:5d}: shape_r_vs_neural={r_shape:+.4f}  peak_lag={peak_lag:+d}", flush=True)

    out = dict(neu_diff=neu_diff.tolist(), layer=LAYER, ks=k_results, n_patients=len(pids))
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
