"""
patch_traj_viz_sentence_cap.py
================================
Recomputes just the `traj_viz` field in an existing
geometry_podcast_pca20_{model}_ctx200_{region}.json using the now-fixed
find_best_sentence_traj() (which caps sentence length at 40 words), without
redoing the expensive compute_profiles() permutation-null pass.

Fixes: patients with broken sentence-boundary detection (the 4 "extra"
patients YEX/YFB/YFX/YFU, whose whole session was previously counted as one
giant "sentence") were trivially winning the widest-surprisal-range
selection just by being huge, producing unreadable trajectory-example plots
in fig2 panels C/D instead of one real sentence.

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u patch_traj_viz_sentence_cap.py --model llama-3.1-8b
    conda run -n gpt2_embed python3 -u patch_traj_viz_sentence_cap.py --model llama-2-7b
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--region", default="hippocampus")
    args = ap.parse_args()

    path = os.path.join(RESULTS_DIR, f"geometry_podcast_pca20_{args.model}_ctx200_{args.region}.json")
    with open(path) as f:
        records = json.load(f)

    for rec in records:
        pid = rec["pid"]
        layer_n = rec["loo_layer_neu"]
        data = load_podcast_patient(pid, region=args.region, model_tag=args.model, context_words=200)
        if data is None:
            print(f"  {pid}: SKIP (no data)")
            continue
        old_range = (rec.get("traj_viz") or {}).get("surp_range", None)
        traj = find_best_sentence_traj(data, layer_n)
        rec["traj_viz"] = traj
        new_range = (traj or {}).get("surp_range", None)
        n_words = len(traj["words"]) if traj else 0
        print(f"  {pid}: old surp_range={old_range}  ->  new surp_range={new_range}  (n_words={n_words})")

    with open(path, "w") as f:
        json.dump(records, f)
    print(f"\nPatched -> {path}")


if __name__ == "__main__":
    main()
