"""
run_curvature_gain_direction_control_podcast.py
================================================
Podcast version of ../run_curvature_gain_direction_control.py.

Tests whether surprisal -> neural curvature (c_neu) survives controlling for
BOTH local step magnitude ("gain", d_neu) AND step direction, at the same
word, using all PODCAST_PATIENTS (target: 13, once the LOO-layer JSON covers
YEX/YFB/YFX/YFU in addition to the original 9).

Podcast has no turns/speakers (single narrator), so PODCAST_BASE_CONTROLS
drops turn_boundary / is_speaker / dist_turn relative to the convo version
(same convention as make_ablation_table_podcast.py).

Model A (baseline):        c_neu ~ surprisal + c_sem + d_sem + PODCAST_BASE_CONTROLS
Model B (+ gain):          + d_neu
Model C (+ direction):     + direction_align
Model D (+ gain + dir):    + d_neu + direction_align

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u run_curvature_gain_direction_control_podcast.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

import run_geometry_analysis_pca20 as geom
from run_clean_fig4_tests_pca20 import ols_clean, distance_to_turn
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

MODEL  = "llama-3.1-8b"
REGION = "hippocampus"

PODCAST_BASE_CONTROLS = ["sent_pos", "sentence_start", "word_len", "log_freq"]


def unit_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n < 1e-12] = np.nan
    return V / n


def load_existing_layers_podcast(model: str, region: str):
    path = os.path.join(BASE_DIR, "results", f"geometry_podcast_pca20_{model}_ctx200_{region}.json")
    with open(path) as f:
        records = json.load(f)
    return {rec["pid"]: int(rec["loo_layer_neu"]) for rec in records}


def build_table_with_direction_podcast(pid, region, model, layer, surprisal_lag=0):
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

    steps_neu = Y_neu[1:] - Y_neu[:-1]
    if len(steps_neu) < 2:
        return None
    d_sem_mag = geom.trajectory_displacement(E)
    d_sem = d_sem_mag[1:]
    d_neu_mag = np.linalg.norm(steps_neu[1:], axis=1)

    unit_steps = unit_rows(steps_neu)
    mu = np.nanmean(unit_steps, axis=0)
    mu_norm = np.linalg.norm(mu)
    if mu_norm < 1e-12:
        return None
    mu_hat = mu / mu_norm
    arriving_unit = unit_rows(steps_neu[1:])
    direction_align = arriving_unit @ mu_hat

    m = min(len(c_sem), len(d_sem), len(c_neu), len(d_neu_mag), len(direction_align))
    center_idx = np.arange(2, 2 + m)
    target_idx = center_idx + int(surprisal_lag)
    valid_target = (target_idx >= 0) & (target_idx < n)
    center_idx = center_idx[valid_target]
    target_idx = target_idx[valid_target]

    sent_pos = np.asarray(data["sent_pos"])
    word_len = np.asarray(data["word_len"], dtype=float)
    log_freq = np.asarray(data["log_freq"], dtype=float)
    surp = np.asarray(data["surp"], dtype=float)

    table = pd.DataFrame({
        "pid": pid, "region": region, "model": model,
        "surprisal": surp[target_idx],
        "c_sem": c_sem[:m][valid_target], "d_sem": d_sem[:m][valid_target],
        "c_neu": c_neu[:m][valid_target], "d_neu": d_neu_mag[:m][valid_target],
        "direction_align": direction_align[:m][valid_target],
        "sent_pos": sent_pos[target_idx].astype(float),
        "sentence_start": (sent_pos[target_idx] == 0).astype(float),
        "word_len": word_len[target_idx],
        "log_freq": log_freq[target_idx],
    })
    return table


def main():
    layers = load_existing_layers_podcast(MODEL, REGION)
    print(f"Layer JSON covers {len(layers)} patients: {sorted(layers)}")
    rows = []
    for pid in PODCAST_PATIENTS:
        if pid not in layers:
            print(f"  {pid}: no LOO layer -- skipped")
            continue
        table = build_table_with_direction_podcast(pid, REGION, MODEL, layers[pid])
        if table is None or table.empty:
            print(f"  {pid}: skipped (no data)")
            continue

        base_preds = {"surprisal": table["surprisal"], "c_sem": table["c_sem"], "d_sem": table["d_sem"],
                      **{c: table[c] for c in PODCAST_BASE_CONTROLS}}

        model_a = ols_clean(table["c_neu"], base_preds)
        model_b = ols_clean(table["c_neu"], {**base_preds, "d_neu": table["d_neu"]})
        model_c = ols_clean(table["c_neu"], {**base_preds, "direction_align": table["direction_align"]})
        model_d = ols_clean(table["c_neu"], {**base_preds, "d_neu": table["d_neu"],
                                              "direction_align": table["direction_align"]})

        def g(m, k):
            return m["betas"].get(k, np.nan), m["pvals"].get(k, np.nan)

        beta_a, p_a = g(model_a, "surprisal")
        beta_b, p_b = g(model_b, "surprisal")
        beta_c, p_c = g(model_c, "surprisal")
        beta_d, p_d = g(model_d, "surprisal")
        beta_dir, p_dir = g(model_c, "direction_align")
        beta_dir_full, p_dir_full = g(model_d, "direction_align")

        rows.append(dict(pid=pid, n=len(table),
                          beta_surp_base=beta_a, p_surp_base=p_a,
                          beta_surp_gain=beta_b, p_surp_gain=p_b,
                          beta_surp_dir=beta_c, p_surp_dir=p_c,
                          beta_surp_gain_dir=beta_d, p_surp_gain_dir=p_d,
                          beta_direction=beta_dir, p_direction=p_dir,
                          beta_direction_full=beta_dir_full, p_direction_full=p_dir_full))
        print(f"  {pid}: n={len(table)}  "
              f"beta_surp[base]={beta_a:+.4f}  [+gain]={beta_b:+.4f}  [+dir]={beta_c:+.4f}  [+gain+dir]={beta_d:+.4f}   "
              f"beta_dir(alone)={beta_dir:+.4f} p={p_dir:.3g}")

    df = pd.DataFrame(rows)
    print(f"\nn_patients = {len(df)}")
    for col in ["beta_surp_base", "beta_surp_gain", "beta_surp_dir", "beta_surp_gain_dir",
                "beta_direction", "beta_direction_full"]:
        vals = df[col].dropna().to_numpy()
        t, p = stats.ttest_1samp(vals, 0.0)
        print(f"  {col}: mean={vals.mean():+.4f}  sem={vals.std(ddof=1)/np.sqrt(len(vals)):.4f}  "
              f"t({len(vals)-1})={t:.3f}  p={p:.4g}")

    for a, b, label in [("beta_surp_base", "beta_surp_dir", "dir-alone vs base"),
                         ("beta_surp_gain", "beta_surp_gain_dir", "+dir vs +gain-only"),
                         ("beta_surp_base", "beta_surp_gain_dir", "+gain+dir vs base")]:
        diff = (df[b] - df[a]).dropna().to_numpy()
        t, p = stats.ttest_1samp(diff, 0.0)
        print(f"  paired diff surp beta [{label}]: mean={diff.mean():+.4f}  t({len(diff)-1})={t:.3f}  p={p:.4g}")

    out_dir = os.path.join(BASE_DIR, "results", "clean_fig4_podcast_pca20")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "curvature_gain_direction_control_podcast_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
