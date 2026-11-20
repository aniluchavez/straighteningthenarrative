"""
run_curvature_gain_direction_control.py
========================================
Extends run_curvature_gain_control.py: tests whether surprisal -> neural
curvature (c_neu) survives controlling for BOTH local step magnitude
("gain", d_neu) AND step direction, at the same word.

"Direction" here = how aligned the word's arriving step vector is with the
patient's own overall mean step direction (session-wide, condition-agnostic,
so it isn't circular with the hi/lo surprisal split or with curvature itself,
which is the angle between TWO adjacent steps rather than one step's
alignment to the session mean):

    steps[k]       = Y_neu[k+1] - Y_neu[k]
    mu             = mean(unit(steps[k]))  over the whole session
    direction_align[j] = cos_angle(steps[j+1], mu)   -- same step used by d_neu[j]

Model A (baseline):        c_neu ~ surprisal + c_sem + d_sem + BASE_CONTROLS
Model B (+ gain):          + d_neu
Model C (+ direction):     + direction_align
Model D (+ gain + dir):    + d_neu + direction_align
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import run_geometry_analysis as geom
from run_clean_fig4_tests import (
    load_existing_layers, ols_clean, BASE_CONTROLS,
    distance_to_turn,
)

MODEL  = "llama-3.1-8b"
REGION = "hippocampus"


def unit_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n < 1e-12] = np.nan
    return V / n


def build_table_with_direction(pid, region, model, layer, surprisal_lag=0):
    data = geom.load_patient(pid, geom.PATIENTS[pid], region, model)
    if data is None or data.get("neu_vecs") is None:
        return None
    n = data["n_words"]
    Y_neu, n_pcs = geom.fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    E = np.asarray(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm

    c_sem = geom.trajectory_curvature(E)
    c_neu = geom.trajectory_curvature(Y_neu)

    steps_neu = Y_neu[1:] - Y_neu[:-1]                        # n-1 vectors, steps[k]=Y[k+1]-Y[k]
    if len(steps_neu) < 2:
        return None
    d_sem_mag = geom.trajectory_displacement(E)
    d_sem = d_sem_mag[1:]                                     # causal, matches steps_neu[1:]
    d_neu_mag = np.linalg.norm(steps_neu[1:], axis=1)          # ||steps[j+1]||, m = n-2

    unit_steps = unit_rows(steps_neu)
    mu = np.nanmean(unit_steps, axis=0)
    mu_norm = np.linalg.norm(mu)
    if mu_norm < 1e-12:
        return None
    mu_hat = mu / mu_norm
    arriving_unit = unit_rows(steps_neu[1:])                  # unit(steps[j+1])
    direction_align = arriving_unit @ mu_hat                  # cos angle to session mean dir

    m = min(len(c_sem), len(d_sem), len(c_neu), len(d_neu_mag), len(direction_align))
    center_idx = np.arange(2, 2 + m)
    target_idx = center_idx + int(surprisal_lag)
    valid_target = (target_idx >= 0) & (target_idx < n)
    center_idx = center_idx[valid_target]
    target_idx = target_idx[valid_target]

    sent_pos = np.asarray(data["sent_pos"])
    is_turn = np.asarray(data["is_turn"], dtype=bool)
    is_speaker = np.asarray(data["is_speaker"], dtype=bool)
    word_len = np.asarray(data["word_len"], dtype=float)
    log_freq = np.asarray(data["log_freq"], dtype=float)
    dist_turn = distance_to_turn(is_turn)
    surp = np.asarray(data["surp"], dtype=float)

    table = pd.DataFrame({
        "pid": pid, "region": region, "model": model,
        "surprisal": surp[target_idx],
        "c_sem": c_sem[:m][valid_target], "d_sem": d_sem[:m][valid_target],
        "c_neu": c_neu[:m][valid_target], "d_neu": d_neu_mag[:m][valid_target],
        "direction_align": direction_align[:m][valid_target],
        "sent_pos": sent_pos[target_idx].astype(float),
        "sentence_start": (sent_pos[target_idx] == 0).astype(float),
        "turn_boundary": is_turn[target_idx].astype(float),
        "is_speaker": is_speaker[target_idx].astype(float),
        "word_len": word_len[target_idx],
        "log_freq": log_freq[target_idx],
        "dist_turn": dist_turn[target_idx],
    })
    return table


def main():
    layers = load_existing_layers(MODEL, REGION)
    no_surp_controls = [c for c in BASE_CONTROLS if c != "surprisal"]
    rows = []
    for pid in geom.PATIENTS:
        if REGION not in geom.PATIENTS[pid] or pid not in layers:
            continue
        table = build_table_with_direction(pid, REGION, MODEL, layers[pid])
        if table is None or table.empty:
            print(f"  {pid}: skipped")
            continue

        base_preds = {"surprisal": table["surprisal"], "c_sem": table["c_sem"], "d_sem": table["d_sem"],
                      **{c: table[c] for c in no_surp_controls}}

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

    out_path = "results/clean_fig4/curvature_gain_direction_control_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
