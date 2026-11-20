"""
compute_event_triggered_layersweep_raw_gpt2xl.py
====================================
Panel A of fig3_event_triggered_gpt2xl_freq_surp.py (semantic curvature,
surprisal-split, PCA20-reduced, layer 5) peaks AT lag 0 (the bend ENDING at
the surprising word) instead of AFTER it like the neural panel does (peaks
at lag +1). Two changes to test whether a different layer/representation
gives a neural-like ("rises after the event") shape instead:

  1. No PCA reduction on the semantic side -- raw standardized embedding
     dims (StandardScaler only, same "raw" convention used in
     compute_podcast_randomdim_bootstrap.py) instead of sem_pca_reduce's
     fixed top-20 PCA.
  2. Sweep ALL layers (0..48) instead of the single fig2-identified layer 5,
     and rank each by how closely its (hi_sem - lo_sem) event-triggered
     difference curve, across the 11 lags, matches the neural
     (hi_neu - lo_neu) difference curve's SHAPE (Pearson r across lags) --
     specifically checking whether the semantic peak lands at lag +1/+2
     (post-event, like neural) rather than lag 0.

Neural side unchanged: pipeline-default fixed window, PCA-reduced
(fit_neural_pca), surprisal-split -- same as before, layer-independent.

Saves: results/event_triggered_layersweep_raw_gpt2xl.json
"""
import json, os, sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis_pca20 import trajectory_curvature, fit_neural_pca, EVENT_WINDOW
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS, "event_triggered_layersweep_raw_gpt2xl.json")

MODEL = "gpt2-xl"
REGION = "hippocampus"
PCTILE_HI, PCTILE_LO = 75, 25
N_LAYERS = 49


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
    per_patient_emb = {}

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
        per_patient_emb[pid] = np.load(data["emb_path"], mmap_mode="r")
        print(f"  {pid}: neural done, n_layers_available={per_patient_emb[pid].shape[0]}", flush=True)

    pids = list(per_patient_neu.keys())
    hi_neu_mu = np.nanmean([per_patient_neu[p][0] for p in pids], axis=0)
    lo_neu_mu = np.nanmean([per_patient_neu[p][1] for p in pids], axis=0)
    neu_diff = np.array(hi_neu_mu) - np.array(lo_neu_mu)
    print(f"\nneural diff curve (hi-lo) by lag: {np.round(neu_diff, 3).tolist()}", flush=True)

    layer_results = {}
    for L in range(N_LAYERS):
        hi_list, lo_list = [], []
        for pid in pids:
            emb_mm = per_patient_emb[pid]
            if L >= emb_mm.shape[0]:
                continue
            hi_mask, lo_mask, n = per_patient_masks[pid]
            emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
            if emb_L.std() == 0:
                continue
            Xs = StandardScaler().fit_transform(emb_L)  # raw dims, NO PCA
            c_sem_full = trajectory_curvature(Xs)
            hi_sem, lo_sem = event_triggered_curv(c_sem_full, n, hi_mask, lo_mask)
            hi_list.append(hi_sem); lo_list.append(lo_sem)
        if not hi_list:
            continue
        hi_mu = np.nanmean(hi_list, axis=0)
        lo_mu = np.nanmean(lo_list, axis=0)
        sem_diff = hi_mu - lo_mu
        r_shape, _ = pearsonr(sem_diff, neu_diff)
        peak_lag = int(np.argmax(hi_mu)) - EVENT_WINDOW
        layer_results[L] = dict(hi_sem=hi_mu.tolist(), lo_sem=lo_mu.tolist(),
                                 shape_r_vs_neural=float(r_shape), peak_lag=peak_lag)
        print(f"  layer {L:3d}: shape_r_vs_neural={r_shape:+.4f}  peak_lag={peak_lag:+d}", flush=True)

    out = dict(neu_diff=neu_diff.tolist(), hi_neu_mu=hi_neu_mu.tolist(), lo_neu_mu=lo_neu_mu.tolist(),
               layers=layer_results, n_patients=len(pids))
    json.dump(out, open(OUT_PATH, "w"), indent=2)

    ranked = sorted(layer_results.items(), key=lambda kv: -kv[1]["shape_r_vs_neural"])
    print("\nTop 10 layers by shape similarity to neural diff curve:")
    for L, v in ranked[:10]:
        print(f"  layer {L:3d}: r={v['shape_r_vs_neural']:+.4f}  peak_lag={v['peak_lag']:+d}")

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
