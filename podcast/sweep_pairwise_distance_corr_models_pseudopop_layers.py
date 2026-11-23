"""
sweep_pairwise_distance_corr_models_pseudopop_layers.py
=======================================================
Pairwise-distance (RSA-style) correlation between semantic and neural space,
same recipe as make_pairwise_distance_scatter_gpt2xl_pseudopop.py /
sweep_pairwise_distance_corr_gpt2xl_pseudopop_layers.py, for several LLMs
and every layer:

  neural   : hippocampal pseudopop, dur80/40 firing RATES, difflib-aligned to
             YEY, common words, per-unit z-score, then valid words
             (build_valid_mask index, as in the original)
  distance : cosine distance between all pairs of valid words, each space
  stat     : Pearson r over all word pairs (pairs are not independent: no p)

Neural cache rebuilt 2026-11-23 (the Sep 18 pseudopop_neural_cache.npz
predates the YFX clock fix and includes YEX's wrong-clock words):
12 patients, YEX excluded. -> results/pseudopop_neural_cache_rate_12pt.npz

argv: model tags (EmbedCache *_ctx200), default gpt2-xl llama-3.1-8b
Saves: results/pairwise_distance_corr_{model}_pseudopop12_layersweep.json

    conda run -n gpt2_embed --no-capture-output python3 -u sweep_pairwise_distance_corr_models_pseudopop_layers.py [models...]
"""
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)
from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import (PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
                            load_words_onset_offset, windowed_neu_vecs)
from compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop import align_to_ref

REF_PID, REGION = "YEY", "hippocampus"
EXCLUDE = {"YEX"}
RESULTS = os.path.join(BASE_DIR, "results")
CACHE = os.path.join(RESULTS, "pseudopop_neural_cache_rate_12pt.npz")


def build_neural():
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        return c["neu_z"], c["common_valid"], list(c["patients"])
    neural = {}
    for pid in PODCAST_PATIENTS:
        if pid in EXCLUDE:
            continue
        d = load_podcast_patient(pid, region=REGION, model_tag="gpt2-xl", context_words=200)
        try:
            _, on, off = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        if d is None:
            continue
        n = min(d["n_words"], len(on))
        v = windowed_neu_vecs(pid, REGION, _convo_regions(pid), on[:n], off[:n], 80, 40, rate=True)
        if v is None:
            continue
        neural[pid] = dict(v=np.asarray(v, float), words=list(d["words"][:n]), d=d)
        print(f"  {pid}: words={n} units={v.shape[1]}", flush=True)
    ref = neural[REF_PID]
    common = set(range(len(ref["words"])))
    maps = {}
    for pid, r in neural.items():
        maps[pid] = align_to_ref(ref["words"], r["words"])
        common &= set(maps[pid])
    common = np.array(sorted(common))
    X = np.concatenate([neural[p]["v"][[maps[p][c] for c in common]] for p in neural], axis=1)
    rd = ref["d"]
    valid = build_valid_mask(len(common), np.asarray(rd["surp"])[common],
                             np.asarray(rd["sent_pos"])[common], np.asarray(rd["is_turn"])[common])
    vidx = np.where(valid)[0]
    neu_z = StandardScaler().fit_transform(X)[vidx]
    assert np.isfinite(neu_z).all()
    np.savez_compressed(CACHE, neu_z=neu_z, common_valid=common[vidx], patients=np.array(list(neural)))
    return neu_z, common[vidx], list(neural)


def cos_dist_upper(X, iu):
    X = np.asarray(X, np.float64)
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    S = Xn @ Xn.T
    return 1.0 - np.clip(S[iu], -1, 1)


def main():
    models = sys.argv[1:] or ["gpt2-xl", "llama-3.1-8b"]
    neu_z, cv, patients = build_neural()
    n = len(cv)
    print(f"pseudopop: {len(patients)} patients, {neu_z.shape[1]} units, {n} valid words", flush=True)
    iu = np.triu_indices(n, k=1)
    dn = cos_dist_upper(neu_z, iu)
    dn = (dn - dn.mean()) / dn.std()
    for model in models:
        d = load_podcast_patient(REF_PID, region=REGION, model_tag=model, context_words=200)
        emb = np.load(d["emb_path"], mmap_mode="r")
        res = {}
        for L in range(emb.shape[0]):
            ds = cos_dist_upper(np.array(emb[L, cv], np.float64), iu)
            r = float(np.mean(dn * (ds - ds.mean()) / ds.std()))
            res[L] = dict(r=r)
            print(f"  {model} layer {L:2d}/{emb.shape[0]-1}: r={r:+.4f}", flush=True)
        b = max(res, key=lambda k: res[k]["r"])
        out = dict(model=model, n_words=int(n), n_units=int(neu_z.shape[1]), patients=patients,
                   dim=int(emb.shape[2]), best_layer=int(b), best_r=res[b]["r"], layers=res)
        json.dump(out, open(os.path.join(RESULTS, f"pairwise_distance_corr_{model}_pseudopop12_layersweep.json"), "w"),
                  indent=2)
        print(f"{model}: best layer {b}/{emb.shape[0]-1}  r={res[b]['r']:+.4f}\n", flush=True)


if __name__ == "__main__":
    main()
