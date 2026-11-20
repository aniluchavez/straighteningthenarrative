"""
patch_scatter_traj_viz.py
==========================
Re-selects the trajectory-example candidate patient (using the just-patched
geometry JSON's traj_viz.surp_range) and recomputes traj_viz_best_layer /
traj_viz_best_disp_layer in an existing scatter_data_podcast_pca20_{model}
_hippocampus.json, WITHOUT rerunning the expensive per-layer chunk/word-level
pass (best_layer, all_layer_chunks, word_level_at_best_layer are untouched).

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u patch_scatter_traj_viz.py --model llama-3.1-8b
    conda run -n gpt2_embed python3 -u patch_scatter_traj_viz.py --model llama-2-7b
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis_pca20 import find_best_sentence_traj
from podcast_loader import load_podcast_patient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REGION = "hippocampus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    geom_path = os.path.join(RESULTS_DIR, f"geometry_podcast_pca20_{args.model}_ctx200_{REGION}.json")
    with open(geom_path) as f:
        geom_records = {r["pid"]: r for r in json.load(f)}

    scatter_path = os.path.join(RESULTS_DIR, f"scatter_data_podcast_pca20_{args.model}_{REGION}.json")
    with open(scatter_path) as f:
        sdata = json.load(f)

    best_layer = sdata["best_layer"]
    best_disp_layer = sdata["best_disp_layer"]

    candidates = []
    for pid, rec in geom_records.items():
        traj_surp_range = (rec.get("traj_viz") or {}).get("surp_range", -1.0)
        candidates.append((pid, float(traj_surp_range)))
    best_pid, best_range = max(candidates, key=lambda x: x[1])
    print(f"Best traj patient: {best_pid}  surp_range={best_range:.2f}")

    data = load_podcast_patient(best_pid, region=REGION, model_tag=args.model, context_words=200)
    print(f"Computing traj_viz at layer {best_layer}...")
    traj_best = find_best_sentence_traj(data, best_layer)
    print(f"Computing displacement traj_viz at layer {best_disp_layer}...")
    traj_disp_best = find_best_sentence_traj(data, best_disp_layer)

    n_words = len(traj_best["words"]) if traj_best else 0
    print(f"New trajectory example: {n_words} words -- {traj_best['words'] if traj_best else None}")

    sdata["traj_viz_best_layer"] = traj_best
    sdata["traj_viz_best_disp_layer"] = traj_disp_best

    with open(scatter_path, "w") as f:
        json.dump(sdata, f)
    print(f"Patched -> {scatter_path}")


if __name__ == "__main__":
    main()
