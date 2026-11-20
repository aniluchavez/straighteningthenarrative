"""
run_clean_fig4_tests_podcast_pca20.py
====================================
Podcast + PCA(20)-reduced-trajectory variant of ../run_clean_fig4_tests.py
(clean_fig4 word/patient/group tables that feed fig3_event_triggered_magnitude).
Reuses all the shared helpers (safe_pearson, centered_displacement,
causal_displacement, distance_to_turn, ols_clean, summarize_patient,
group_summary) from run_clean_fig4_tests_pca20.py unchanged; only the data
loading is podcast-sourced (podcast_loader.load_podcast_patient instead of
convo's geom.load_patient) and each layer's embedding is PCA(20)-reduced
via geom.sem_pca_reduce before curvature/displacement.

This is a genuinely NEW table for podcast (no raw-embedding podcast
version of clean_fig4 existed before this).

Outputs (in results/clean_fig4_podcast_pca20/):
  clean_fig4_podcast_pca20_{model}_lag{lag}{_causal}_patient_summary.csv
  clean_fig4_podcast_pca20_{model}_lag{lag}{_causal}_word_table.csv
  clean_fig4_podcast_pca20_{model}_lag{lag}{_causal}_group_summary.csv

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 run_clean_fig4_tests_podcast_pca20.py \
        --model llama-3.1-8b --surprisal_lag 0 --causal
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

import run_geometry_analysis_pca20 as geom
from run_clean_fig4_tests_pca20 import (
    safe_pearson, centered_displacement, causal_displacement, distance_to_turn,
    ols_clean, summarize_patient, group_summary,
)
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient


def load_existing_layers_podcast(model: str, region: str) -> Dict[str, int]:
    path = os.path.join(BASE_DIR, "results", f"geometry_podcast_pca20_{model}_ctx200_{region}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run run_podcast_geometry_analysis_pca20.py first.")
    with open(path) as f:
        records = json.load(f)
    return {rec["pid"]: int(rec["loo_layer_neu"]) for rec in records}


def build_patient_table_podcast(pid: str, region: str, model: str, layer: int,
                                surprisal_lag: int, causal: bool = False) -> Optional[pd.DataFrame]:
    data = load_podcast_patient(pid, region=region, model_tag=model, context_words=200)
    if data is None or data.get("neu_vecs") is None:
        return None

    n = data["n_words"]
    Y_neu, n_pcs = geom.fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    E = np.asarray(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm
    E = geom.sem_pca_reduce(E)

    c_sem = geom.trajectory_curvature(E)
    c_neu = geom.trajectory_curvature(Y_neu)

    if causal:
        d_sem = causal_displacement(E)
        d_neu = causal_displacement(Y_neu)
        m = min(len(c_sem), len(d_sem), len(c_neu), len(d_neu))
        center_idx = np.arange(2, 2 + m)
    else:
        d_sem = centered_displacement(E)
        d_neu = centered_displacement(Y_neu)
        m = min(len(c_sem), len(d_sem), len(c_neu), len(d_neu))
        center_idx = np.arange(1, 1 + m)

    target_idx = center_idx + int(surprisal_lag)
    valid_target = (target_idx >= 0) & (target_idx < n)

    sent_pos = np.asarray(data["sent_pos"])
    is_turn = np.asarray(data["is_turn"], dtype=bool)
    is_speaker = np.asarray(data["is_speaker"], dtype=bool)
    word_len = np.asarray(data["word_len"], dtype=float)
    log_freq = np.asarray(data["log_freq"], dtype=float)
    dist_turn = distance_to_turn(is_turn)
    surp = np.asarray(data["surp"], dtype=float)

    center_idx = center_idx[valid_target]
    target_idx = target_idx[valid_target]

    table = pd.DataFrame({
        "pid": pid,
        "region": region,
        "model": model,
        "loo_layer_neu": layer,
        "n_neural_pcs": n_pcs,
        "center_word_index": center_idx,
        "target_word_index": target_idx,
        "surprisal_lag": int(surprisal_lag),
        "surprisal": surp[target_idx],
        "c_sem": c_sem[:m][valid_target],
        "d_sem": d_sem[:m][valid_target],
        "c_neu": c_neu[:m][valid_target],
        "d_neu": d_neu[:m][valid_target],
        "sent_pos": sent_pos[target_idx].astype(float),
        "sentence_start": (sent_pos[target_idx] == 0).astype(float),
        "near_sentence_start": (sent_pos[target_idx] <= 1).astype(float),
        "turn_boundary": is_turn[target_idx].astype(float),
        "turn_boundary_center": is_turn[center_idx].astype(float),
        "is_speaker": is_speaker[target_idx].astype(float),
        "word_len": word_len[target_idx],
        "log_freq": log_freq[target_idx],
        "dist_turn": dist_turn[target_idx],
    })
    return table


def run(model: str, regions, output_dir: str, surprisal_lag: int, causal: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    all_patient_rows = []
    all_word_tables = []

    for region in regions:
        layers = load_existing_layers_podcast(model, region)
        for pid in PODCAST_PATIENTS:
            if pid not in layers:
                continue
            print(f"  {model} {region} {pid} layer={layers[pid]}")
            table = build_patient_table_podcast(pid, region, model, layers[pid], surprisal_lag, causal=causal)
            if table is None or table.empty:
                print("    skipped")
                continue
            all_word_tables.append(table)
            all_patient_rows.append(summarize_patient(table))

    patient_df = pd.DataFrame(all_patient_rows)
    word_df = pd.concat(all_word_tables, ignore_index=True) if all_word_tables else pd.DataFrame()
    summary_df = group_summary(patient_df) if not patient_df.empty else pd.DataFrame()

    stem = f"clean_fig4_podcast_pca20_{model}_lag{surprisal_lag}{'_causal' if causal else ''}"
    patient_path = os.path.join(output_dir, f"{stem}_patient_summary.csv")
    word_path = os.path.join(output_dir, f"{stem}_word_table.csv")
    group_path = os.path.join(output_dir, f"{stem}_group_summary.csv")
    patient_df.to_csv(patient_path, index=False)
    word_df.to_csv(word_path, index=False)
    summary_df.to_csv(group_path, index=False)

    print(f"\nSaved patient summary: {patient_path}")
    print(f"Saved word table:      {word_path}")
    print(f"Saved group summary:   {group_path}")
    return patient_df, summary_df


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama-3.1-8b")
    p.add_argument("--regions", nargs="+", default=["hippocampus"])
    p.add_argument("--output_dir", default=os.path.join(BASE_DIR, "results", "clean_fig4_podcast_pca20"))
    p.add_argument("--surprisal_lag", type=int, default=1)
    p.add_argument("--causal", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    patient_df, summary_df = run(args.model, args.regions, args.output_dir,
                                 args.surprisal_lag, causal=args.causal)
    if not summary_df.empty:
        print("\nGroup summary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
