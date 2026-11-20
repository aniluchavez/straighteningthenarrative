"""
compute_podcast_top3_check.py
====================================
One-off check: fix K=3 (top-3 PCs) on BOTH sides -- neural (hippocampus PCA)
and semantic (gpt2, PCA'd, layer-averaged) -- instead of each patient's own
matched n_pcs, and see what the neural<->semantic curvature/displacement/
position correlation looks like. Reuses compute_podcast_randomdim_bootstrap's
position()/metrics_from_X()/prep_patient_layers()/topk_metrics()/pool_and_corr().
"""
import sys, os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_podcast_randomdim_bootstrap as rb
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient
from run_geometry_analysis import build_valid_mask

K = 3


def get_neural_data_k3(pid):
    data0 = load_podcast_patient(pid, region=rb.REGION, model_tag=rb.MODEL, context_words=200)
    if data0 is None or data0["neu_vecs"] is None:
        return None
    n = data0["n_words"]
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 50:
        return None
    nv = data0["neu_vecs"][:n]
    max_pc = min(nv.shape[1], n - 1)
    if nv.std() == 0 or max_pc < K:
        return None
    Xs = StandardScaler().fit_transform(nv)
    pca = PCA(n_components=max_pc).fit(Xs)
    Y3 = pca.transform(Xs)[:, :K]
    c, d, p = rb.metrics_from_X(Y3, valid, n)
    return dict(n=n, valid=valid, c_neu=c, d_neu=d, p_neu=p, n_pcs=K)


def main():
    neural = {}
    for pid in PODCAST_PATIENTS:
        res = get_neural_data_k3(pid)
        if res is None:
            print(f"  {pid}: SKIP (no neural data / <3 PCs)")
            continue
        neural[pid] = res
        print(f"  {pid}: n_words={res['n']}", flush=True)

    layer_cache = {}
    for pid in neural:
        prep = rb.prep_patient_layers(pid, neural[pid]["n"], K)
        if prep is None:
            print(f"  {pid}: SKIP (no semantic embedding)")
            continue
        layer_cache[pid] = prep

    pids = [p for p in neural if p in layer_cache]
    print(f"\n{len(pids)} patients with both neural + semantic data (K={K} both sides): {pids}")

    for scheme in ("raw", "pca"):
        sem = {}
        for pid in pids:
            layers = layer_cache[pid][scheme]
            r = rb.topk_metrics(layers, K, neural[pid]["valid"], neural[pid]["n"])
            if r is not None:
                sem[pid] = r
        r = rb.pool_and_corr(neural, sem)
        print(f"\nscheme={scheme}  (semantic K={K}, {rb.CHUNK}-word chunks)")
        for key in ("curv", "disp", "pos"):
            v = r[key]
            print(f"  {key}: r={v:+.4f}" if v is not None else f"  {key}: insufficient data")


if __name__ == "__main__":
    main()
