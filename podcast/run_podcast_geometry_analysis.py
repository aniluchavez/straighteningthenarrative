"""
run_podcast_geometry_analysis.py
==================================
Podcast equivalent of run_geometry_analysis.py's run_one(): Pass 1 (LOO
layer selection across all patients) + Pass 2 (regression, event-triggered,
best-sentence trajectory) -- same helper functions, reused directly, just
sourced from podcast_loader.load_podcast_patient() instead of convo's
load_patient().

Usage:
    conda run -n gpt2_embed python3 -u run_podcast_geometry_analysis.py \
        --models llama-3.1-8b --overwrite
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis import (
    compute_profiles, run_regression, compute_event_triggered,
    find_best_sentence_traj, fit_neural_pca, MIN_PAIRS,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient, STATIC_MODELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REGION = "hippocampus"

MODELS = {
    "gpt2":         13,
    "gpt2-medium":  25,
    "gpt2-large":   37,
    "gpt2-xl":      49,
    "bert-base":    13,
    "roberta-base": 13,
    "opt-350m":     24,
    "llama-2-7b":   33,
    "llama-3.1-8b": 33,
    "mistral-7b":   33,
    "gemma-2-9b":   43,
    "fasttext-wiki": 1,
}


def run_one_podcast(model_tag, n_layers, region=REGION, context_words=200):
    print(f"\n{'='*65}\n  PODCAST MODEL: {model_tag}  ctx{context_words}   REGION: {region}\n{'='*65}")

    all_data, all_profs = {}, {}
    for short in PODCAST_PATIENTS:
        data = load_podcast_patient(short, region=region, model_tag=model_tag, context_words=context_words)
        if data is None:
            print(f"  {short}: SKIP -- no data")
            continue
        all_data[short] = data
        prof = compute_profiles(short, data, n_layers)
        if prof is not None:
            all_profs[short] = prof
            print(f"  {short}  n={prof['n']}  n_pcs={prof['n_pcs']}")
        else:
            print(f"  {short}: SKIP -- too few valid word pairs (< {MIN_PAIRS})")

    pids = list(all_profs.keys())
    if len(pids) < 2:
        print(f"  Too few patients ({len(pids)}) -- skip.")
        return None

    # LOO layer selection (same logic as convo run_one)
    loo_surp, loo_neural = {}, {}
    for pid in pids:
        others = [p for p in pids if p != pid]
        rs_mat = np.array([all_profs[p]["r_surp"] for p in others])
        rn_mat = np.array([all_profs[p]["r_neural"] for p in others], dtype=float)
        loo_surp[pid] = int(np.argmax(np.abs(rs_mat.mean(axis=0))))
        rn_mean = np.nanmean(rn_mat, axis=0)
        loo_neural[pid] = (int(np.nanargmax(np.abs(rn_mean)))
                           if not np.all(np.isnan(rn_mean))
                           else loo_surp[pid])

    print(f"\n  LOO-neural layers: { {p: loo_neural[p] for p in pids} }")

    results = []
    for pid in pids:
        data = all_data[pid]
        layer_n = loo_neural[pid]
        Y_neu, n_pcs = fit_neural_pca(data["neu_vecs"], data["n_words"])
        reg = run_regression(data, layer_n, Y_neu, n_pcs)
        evt = compute_event_triggered(data, layer_n, Y_neu)
        traj = find_best_sentence_traj(data, layer_n)
        results.append(dict(pid=pid,
                            loo_layer_surp=loo_surp[pid],
                            loo_layer_neu=loo_neural[pid],
                            profile=all_profs[pid],
                            regression=reg,
                            event_triggered=evt,
                            traj_viz=traj))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--regions", nargs="+", default=[REGION])
    parser.add_argument("--context_words", nargs="+", type=int, default=[200])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for model_tag in args.models:
        if model_tag not in MODELS:
            print(f"Unknown model {model_tag}, skipping")
            continue
        n_layers = MODELS[model_tag]
        ctx_list = [None] if model_tag in STATIC_MODELS else args.context_words
        for region in args.regions:
            for ctx in ctx_list:
                ctx_tag = "" if ctx is None else f"_ctx{ctx}"
                out_path = os.path.join(RESULTS_DIR, f"geometry_podcast_{model_tag}{ctx_tag}_{region}.json")
                if os.path.exists(out_path) and not args.overwrite:
                    print(f"{out_path} exists, skipping (use --overwrite)")
                    continue
                results = run_one_podcast(model_tag, n_layers, region, context_words=ctx or 200)
                if results is None:
                    continue
                with open(out_path, "w") as f:
                    json.dump(results, f)
                print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
