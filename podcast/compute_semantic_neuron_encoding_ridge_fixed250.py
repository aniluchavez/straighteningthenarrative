"""
compute_semantic_neuron_encoding_ridge_fixed250.py
=====================================================
Per-neuron semantic encoding with the SAME Gaussian ridge as the main
reported podcast encoding (ConvoData/run_gaussian_durint.py, run_unit_set:
100 z-scored embedding PCs + z log(dur) (unpenalized) + PC x log(dur),
5-fold shuffled KFold, per-neuron lambda from logspace(-3, 5, 15), Gaussian
CV log-likelihood; 500 random-shuffle + 100 circular-shift nulls with
rates + duration moved together against the embedding; sig = p < 0.05).

Inputs here (from compute_semantic_neuron_encoding_fixed250.py's cache,
results/semantic_neuron_pseudopop_noYFX.npz): 12 patients (no YFX), words
common to all; response = firing rate in fixed [onset+80, onset+330] ms;
embedding = gpt2-xl layer 29 (YEY embeddings at common words); duration =
YEY word duration at common words. Fit per patient, as in run_gaussian_durint.

Saves: results/semantic_neuron_encoding_ridge_fixed250.json

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed --no-capture-output python3 -u compute_semantic_neuron_encoding_ridge_fixed250.py
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
sys.path.insert(0, "/scratch/aniluchavez/semantic_syntactic_emb_project_for_Chavez/ConvoData")

from run_gaussian_durint import run_unit_set, SEED as RIDGE_SEED
from podcast_loader import load_podcast_patient

REF_PID = "YEY"
REGION = "hippocampus"
MODEL = "gpt2-xl"
LAYER = 29
WIN_S = 0.25
N_SHUFFLE, N_SHIFT = 500, 100
CACHE_PATH = os.path.join(RESULTS, "semantic_neuron_pseudopop_noYFX.npz")
OUT_PATH = os.path.join(RESULTS, "semantic_neuron_encoding_ridge_fixed250.json")


def main():
    c = np.load(CACHE_PATH, allow_pickle=True)
    rate = c["cnt"].astype(float) / WIN_S
    unit_pid, common, dur = c["unit_pid"], c["common"], c["dur"].astype(float)
    data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
    X = np.array(np.load(data_m["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
    print(f"words={len(common)}  units={rate.shape[1]}  emb={X.shape}", flush=True)

    rng = np.random.default_rng(RIDGE_SEED)
    units = []
    for pid in dict.fromkeys(unit_pid.tolist()):
        cols = np.where(unit_pid == pid)[0]
        r = run_unit_set(X, rate[:, cols], dur, rng, N_SHUFFLE, N_SHIFT)
        for j, u in enumerate(cols):
            units.append(dict(unit=int(u), pid=pid, cvLL=float(r["cvLL"][j]),
                              p_shuffle=float(r["pval_shuffle"][j]), p_shift=float(r["pval_shift"][j])))
        print(f"  {pid}: {len(cols)} units, {r['n_words']} words | sig shuffle {int(r['is_sig_shuffle'].sum())}, "
              f"shift {int(r['is_sig_shift'].sum())}", flush=True)

    n = len(units)
    for k in ("shuffle", "shift"):
        s = sum(u[f"p_{k}"] < 0.05 for u in units)
        print(f"  total sig ({k}): {s}/{n} ({100 * s / n:.1f}%)")
    json.dump(dict(config=dict(LAYER=LAYER, MODEL=MODEL, window_ms=[80, 330], N_SHUFFLE=N_SHUFFLE,
                               N_SHIFT=N_SHIFT, model="run_gaussian_durint.run_unit_set", n_words=int(len(common))),
                   units=units), open(OUT_PATH, "w"), indent=1)
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
