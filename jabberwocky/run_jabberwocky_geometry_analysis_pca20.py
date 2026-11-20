"""
run_jabberwocky_geometry_analysis_pca20.py
===========================================
Jabberwocky version of ../podcast/run_podcast_geometry_analysis_pca20.py:
runs compute_profiles() (from run_geometry_analysis_pca20.py, unchanged) on
all 7 JABBERWOCKY_PATIENTS, producing the same per-layer profile structure
(mean_curv, r_neural, etc. by normalized layer position) used to build
fig1_layer_profiles_pca20_podcast.png, so the two can be overlaid directly.

Outputs: results/geometry_jabberwocky_pca20_{model}_ctx200_{region}.json

Run:
    cd .../geometry_paper/jabberwocky
    conda run -n gpt2_embed python3 -u run_jabberwocky_geometry_analysis_pca20.py \
        --model llama-3.1-8b
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis_pca20 import compute_profiles
from jabberwocky_loader import JABBERWOCKY_PATIENTS, load_jabberwocky_patient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REGION = "hippocampus"

N_LAYERS = {
    "gpt2": 13, "gpt2-medium": 25, "gpt2-large": 37, "gpt2-xl": 49,
    "bert-base": 13, "roberta-base": 13, "opt-350m": 24,
    "llama-2-7b": 33, "llama-3.1-8b": 33, "mistral-7b": 33, "gemma-2-9b": 43,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--region", default=REGION)
    args = ap.parse_args()

    n_layers = N_LAYERS[args.model]
    records = []
    for pid in JABBERWOCKY_PATIENTS:
        data = load_jabberwocky_patient(pid, region=args.region, model_tag=args.model, context_words=200)
        if data is None or data.get("neu_vecs") is None or data.get("emb_path") is None:
            print(f"  {pid}: SKIP (missing data)")
            continue
        prof = compute_profiles(pid, data, n_layers)
        if prof is None:
            print(f"  {pid}: SKIP (compute_profiles returned None)")
            continue
        records.append(prof)
        print(f"  {pid}: n={prof['n']}  n_pcs={prof['n_pcs']}  "
              f"peak_r_neural={max(prof['r_neural']):.4f} at layer {int(prof['r_neural'].index(max(prof['r_neural'])))}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"geometry_jabberwocky_pca20_{args.model}_ctx200_{args.region}.json")
    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"\nSaved {len(records)} patient profiles -> {out_path}")


if __name__ == "__main__":
    main()
