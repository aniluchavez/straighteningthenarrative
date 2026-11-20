"""
compute_curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop.py
===========================================================================
Nested random PC accumulation, pseudopopulation, gpt2-xl layer 29.
For each of N_ORDER random orderings of the PC pool (K_POOL=200 per side),
PCs are added one at a time in that order and the chunked semantic-neural
correlation of curvature / displacement / position is computed at each K in
KS (position = coordinate on the first PC in the ordering).

Schemes:
  both : random order on both sides (independent permutations)
  sem  : random order on semantic side, neural top-K
  neu  : random order on neural side, semantic top-K
  top  : top-K on both (reference), with shuffle null (chunk-order permutation)

Inputs: results/pseudopop_full_sequence_dur8040_rate.npz (includes YFX).
Saves:  results/curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop.json
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

KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 200]
N_ORDER = 200
SEED = 0
OUT_PATH = os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_random_pc_accumulation_pseudopop.json")


def main():
    c = np.load(CACHE_PATH)
    neu_z, common, valid = c["neu_z"], c["common"], c["valid"]
    data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
    emb = np.array(np.load(data_m["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
    Ys, _ = pca_pool(emb)
    Yn, _ = pca_pool(neu_z)
    ns, nn = Ys.shape[1], Yn.shape[1]
    print(f"words={len(common)} valid={int(valid.sum())} pools sem={ns} neu={nn}", flush=True)

    rng = np.random.default_rng(SEED)
    nuller = Nuller(int(valid.sum()) // CHUNK_SIZE, rng, mode="shuffle")
    out = dict(config=dict(KS=KS, N_ORDER=N_ORDER, SEED=SEED, LAYER=LAYER, null="shuffle"), top={}, random={})

    for K in KS:
        out["top"][str(K)] = corr3(Ys[:, :K], Yn[:, :K], valid, nuller)
    print("top:  " + " ".join(f"K{K}:{out['top'][str(K)]['curv']['r']:+.3f}" for K in KS), flush=True)

    top_s, top_n = np.arange(ns), np.arange(nn)
    for scheme in ("both", "sem", "neu"):
        arr = np.full((N_ORDER, len(KS), len(METRICS)), np.nan)
        for i in range(N_ORDER):
            os_ = rng.permutation(ns) if scheme in ("both", "sem") else top_s
            on_ = rng.permutation(nn) if scheme in ("both", "neu") else top_n
            for j, K in enumerate(KS):
                r = corr3(Ys[:, os_[:K]], Yn[:, on_[:K]], valid)
                arr[i, j] = [r[m]["r"] for m in METRICS]
        out["random"][scheme] = arr.tolist()
        med = np.median(arr, axis=0)
        print(f"{scheme:5s} median " + "  ".join(
            f"K{K}:" + "/".join(f"{med[j, k]:+.3f}" for k in range(len(METRICS))) for j, K in enumerate(KS)), flush=True)

    json.dump(out, open(OUT_PATH, "w"))
    print(f"  -> {OUT_PATH}  (metrics order {METRICS})")


if __name__ == "__main__":
    main()
