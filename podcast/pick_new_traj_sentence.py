"""
pick_new_traj_sentence.py
==========================
Ranks ALL 6-12 word candidate sentences (across ALL 13 podcast patients, not
just each patient's single best) by surprisal range, and computes/saves
traj_viz for the sentence at a given rank (0 = the one already used, 1 =
next-best, etc.) into the existing scatter_data_podcast_pca20_{model}
_hippocampus.json's traj_viz_best_layer / traj_viz_best_disp_layer fields.

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u pick_new_traj_sentence.py --model llama-3.1-8b --rank 1
    conda run -n gpt2_embed python3 -u pick_new_traj_sentence.py --model llama-2-7b --rank 1
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import procrustes as scipy_procrustes
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis_pca20 import sem_pca_reduce
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REGION = "hippocampus"
MIN_WORDS, MAX_WORDS = 6, 12


def build_traj(data, wis, layer):
    n = data["n_words"]
    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    emb_full = sem_pca_reduce(np.array(emb_mm[layer, :n], dtype=np.float64))
    emb_sent = emb_full[wis]
    del emb_mm
    neu_sent = data["neu_vecs"][wis].astype(np.float64)

    def _pca2d(X):
        k = min(2, X.shape[1], X.shape[0] - 1)
        if k < 2:
            return None
        return PCA(n_components=k).fit_transform(X - X.mean(0))

    llm_2d, neu_2d = _pca2d(emb_sent), _pca2d(neu_sent)
    if llm_2d is None or neu_2d is None:
        return None
    try:
        _, neu_aligned, disparity = scipy_procrustes(llm_2d, neu_2d)
    except Exception:
        neu_aligned, disparity = neu_2d, float("nan")
    return dict(
        words=data["words"][wis].tolist(),
        surprisal=data["surp"][wis].tolist(),
        llm_pca2d=llm_2d.tolist(),
        neu_pca2d=neu_aligned.tolist(),
        procrustes_disparity=float(disparity),
        layer=int(layer),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=1, help="0=best (already used), 1=next best, ...")
    args = ap.parse_args()

    scatter_path = os.path.join(RESULTS_DIR, f"scatter_data_podcast_pca20_{args.model}_{REGION}.json")
    with open(scatter_path) as f:
        sdata = json.load(f)
    best_layer = sdata["best_layer"]
    best_disp_layer = sdata["best_disp_layer"]

    candidates = []  # (surp_range, pid, wis)
    for pid in PODCAST_PATIENTS:
        data = load_podcast_patient(pid, region=REGION, model_tag=args.model, context_words=200)
        if data is None or data.get("neu_vecs") is None:
            continue
        surp, sent_id = data["surp"], data["sent_id"]
        for sid in np.unique(sent_id):
            wis = np.where(sent_id == sid)[0]
            if len(wis) < MIN_WORDS or len(wis) > MAX_WORDS:
                continue
            sv = surp[wis]
            if np.isnan(sv).sum() > len(wis) // 2:
                continue
            sr = float(np.nanmax(sv) - np.nanmin(sv))
            candidates.append((sr, pid, wis))

    candidates.sort(key=lambda x: -x[0])
    print(f"Top 10 unique-text candidates:")
    # Dedupe on the actual word text, not (pid, indices) -- every patient
    # heard the same podcast, so the same sentence tops every patient's own
    # ranking.
    words_cache = {}
    deduped = []
    seen_text = set()
    for sr, pid, wis in candidates:
        if pid not in words_cache:
            d = load_podcast_patient(pid, region=REGION, model_tag=args.model, context_words=200)
            words_cache[pid] = d["words"] if d is not None else None
        words = words_cache[pid]
        if words is None:
            continue
        text_key = tuple(words[wis].tolist())
        if text_key in seen_text:
            continue
        seen_text.add(text_key)
        deduped.append((sr, pid, wis))
    ranked_unique = deduped
    for i, (sr, pid, wis) in enumerate(ranked_unique[:10]):
        print(f"  [{i}] {pid} surp_range={sr:.2f} n_words={len(wis)}  words={words_cache[pid][wis].tolist()}")

    sr, pid, wis = ranked_unique[args.rank]
    print(f"\nUsing rank {args.rank}: {pid}  surp_range={sr:.2f}  n_words={len(wis)}")

    data = load_podcast_patient(pid, region=REGION, model_tag=args.model, context_words=200)
    traj_best = build_traj(data, wis, best_layer)
    traj_disp_best = build_traj(data, wis, best_disp_layer)
    print(f"Words: {traj_best['words']}")

    sdata["traj_viz_best_layer"] = traj_best
    sdata["traj_viz_best_disp_layer"] = traj_disp_best
    with open(scatter_path, "w") as f:
        json.dump(sdata, f)
    print(f"Patched -> {scatter_path}")


if __name__ == "__main__":
    main()
