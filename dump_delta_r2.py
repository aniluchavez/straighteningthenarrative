"""Compute exact ΔR² summary stats for the curvature ablation figure table."""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import sem as scipy_sem, f as f_dist

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(BASE_DIR, "results", "clean_fig4")
WORD_TABLE = os.path.join(RESULTS, "clean_fig4_llama-3.1-8b_lag0_word_table.csv")
REGION     = "hippocampus"

CONTROL_SETS = {
    "Raw (no controls)":     [],
    "+ sentence position":   ["sent_pos", "sentence_start"],
    "+ turn & speaker":      ["sent_pos", "sentence_start", "turn_boundary", "is_speaker", "dist_turn"],
    "+ length & frequency":  ["sent_pos", "sentence_start", "turn_boundary", "is_speaker", "dist_turn", "word_len", "log_freq"],
}


def delta_r2_and_ftest(y, sub, controls):
    cols_full = ["surprisal"] + controls
    avail_full = [c for c in cols_full if c in sub.columns]
    X_full = sub[avail_full].copy()
    mask = np.isfinite(y) & X_full.notna().all(axis=1)
    y_  = np.asarray(y, dtype=float)[mask]
    X_f = X_full[mask].to_numpy(float)
    n, k = len(y_), len(avail_full)
    if n < k + 3:
        return float("nan"), float("nan")
    y_  = (y_  - y_.mean())  / (y_.std()  + 1e-12)
    X_f = (X_f - X_f.mean(0)) / (X_f.std(0) + 1e-12)
    ss_tot = np.dot(y_, y_)
    Xm_full = np.column_stack([np.ones(n), X_f])
    b_full, _, _, _ = np.linalg.lstsq(Xm_full, y_, rcond=None)
    ss_res_full = np.sum((y_ - Xm_full @ b_full) ** 2)
    r2_full = 1 - ss_res_full / (ss_tot + 1e-12)
    if controls:
        avail_ctrl = [c for c in controls if c in sub.columns]
        X_c = sub[avail_ctrl][mask].to_numpy(float)
        X_c = (X_c - X_c.mean(0)) / (X_c.std(0) + 1e-12)
        Xm_ctrl = np.column_stack([np.ones(n), X_c])
    else:
        Xm_ctrl = np.ones((n, 1))
    b_ctrl, _, _, _ = np.linalg.lstsq(Xm_ctrl, y_, rcond=None)
    ss_res_ctrl = np.sum((y_ - Xm_ctrl @ b_ctrl) ** 2)
    r2_ctrl = 1 - ss_res_ctrl / (ss_tot + 1e-12)
    delta_r2 = r2_full - r2_ctrl
    df2 = n - k - 1
    if df2 <= 0 or ss_res_full <= 0:
        return float(delta_r2), float("nan")
    F  = (delta_r2 / 1.0) / ((1 - r2_full) / df2)
    pv = f_dist.sf(F, 1, df2)
    return float(delta_r2), float(pv)


def pstar(p):
    if not np.isfinite(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


wt = pd.read_csv(WORD_TABLE)
wt = wt[wt["region"] == REGION].copy()

out = {}
for space, c_col, d_col in [("semantic", "c_sem", "d_sem"), ("neural", "c_neu", "d_neu")]:
    out[space] = {}
    for k, ctrl in CONTROL_SETS.items():
        dr2_c, dr2_d, pv_c, pv_d = [], [], [], []
        for pid, sub in wt.groupby("pid"):
            sub = sub.copy()
            if len(sub) < 50:
                continue
            dc, pc = delta_r2_and_ftest(sub[c_col].values, sub, ctrl)
            dd, pd_ = delta_r2_and_ftest(sub[d_col].values, sub, ctrl)
            if np.isfinite(dc): dr2_c.append(dc)
            if np.isfinite(pc): pv_c.append(pc)
            if np.isfinite(dd): dr2_d.append(dd)
            if np.isfinite(pd_): pv_d.append(pd_)
        out[space][k] = {
            "kappa_mean":   float(np.mean(dr2_c)),
            "kappa_sem":    float(scipy_sem(dr2_c)),
            "kappa_p_med":  float(np.median(pv_c)) if pv_c else float("nan"),
            "kappa_sig":    pstar(np.median(pv_c) if pv_c else float("nan")),
            "d_mean":       float(np.mean(dr2_d)),
            "d_sem":        float(scipy_sem(dr2_d)),
            "d_p_med":      float(np.median(pv_d)) if pv_d else float("nan"),
            "d_sig":        pstar(np.median(pv_d) if pv_d else float("nan")),
            "n_patients":   len(dr2_c),
        }

print(json.dumps(out, indent=2))