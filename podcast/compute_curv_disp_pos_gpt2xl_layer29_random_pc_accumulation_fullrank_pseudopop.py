"""
compute_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_fullrank_pseudopop.py
====================================================================================
Same as compute_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop.py
but with full-rank PC pools: semantic 1600 (gpt2-xl width), neural 475 (units).
K runs on both spaces up to 475; beyond that the semantic side keeps growing
to 1600 with the neural side fixed at all 475 PCs (K_neu = min(K, 475)).
Also stores, per random ordering, the step at which semantic PC1 / neural PC1
enters (pc1_pos_sem / pc1_pos_neu, 1-based).

Schemes: both / sem / neu random orderings (see the 200-PC version); top = top-K
reference with shuffle null.

Saves: results/curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_fullrank_pseudopop.json
"""
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from podcast_loader import load_podcast_patient
from compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop import (
    CACHE_PATH, REF_PID, REGION, MODEL, LAYER, CHUNK_SIZE, METRICS, pca_pool, corr3, Nuller,
)

KS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 475, 640, 800, 1024, 1280, 1600]
N_ORDER = 200
SEED = 0
OUT_PATH = os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_fullrank_pseudopop.json")


def main():
    c = np.load(CACHE_PATH)
    neu_z, common, valid = c["neu_z"], c["common"], c["valid"]
    data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
    emb = np.array(np.load(data_m["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
    Ys, evr_s = pca_pool(emb, k=emb.shape[1])
    Yn, evr_n = pca_pool(neu_z, k=neu_z.shape[1])
    ns, nn = Ys.shape[1], Yn.shape[1]
    ks = [K for K in KS if K <= ns]
    print(f"words={len(common)} valid={int(valid.sum())} pools sem={ns} neu={nn}", flush=True)

    rng = np.random.default_rng(SEED)
    nuller = Nuller(int(valid.sum()) // CHUNK_SIZE, rng, mode="shuffle")
    out = dict(config=dict(KS=ks, N_ORDER=N_ORDER, SEED=SEED, LAYER=LAYER, null="shuffle", n_sem=ns, n_neu=nn),
               evr_sem=evr_s.tolist(), evr_neu=evr_n.tolist(), top={}, random={})

    for K in ks:
        out["top"][str(K)] = corr3(Ys[:, :K], Yn[:, :min(K, nn)], valid, nuller)
    print("top curv: " + " ".join(f"K{K}:{out['top'][str(K)]['curv']['r']:+.3f}" for K in ks), flush=True)

    for scheme in ("both", "sem", "neu"):
        arr = np.full((N_ORDER, len(ks), len(METRICS)), np.nan)
        p1s, p1n = [], []
        for i in range(N_ORDER):
            os_ = rng.permutation(ns) if scheme in ("both", "sem") else np.arange(ns)
            on_ = rng.permutation(nn) if scheme in ("both", "neu") else np.arange(nn)
            p1s.append(int(np.where(os_ == 0)[0][0]) + 1)
            p1n.append(int(np.where(on_ == 0)[0][0]) + 1)
            for j, K in enumerate(ks):
                r = corr3(Ys[:, os_[:K]], Yn[:, on_[:min(K, nn)]], valid)
                arr[i, j] = [r[m]["r"] for m in METRICS]
        out["random"][scheme] = dict(r=arr.tolist(), pc1_pos_sem=p1s, pc1_pos_neu=p1n)
        med = np.median(arr, axis=0)
        print(f"{scheme:5s} median curv/disp: " + "  ".join(
            f"K{K}:{med[j, 0]:+.3f}/{med[j, 1]:+.3f}" for j, K in enumerate(ks)), flush=True)
        json.dump(out, open(OUT_PATH, "w"))

    json.dump(out, open(OUT_PATH, "w"))
    print(f"  -> {OUT_PATH}  (metrics order {METRICS})")


if __name__ == "__main__":
    main()
