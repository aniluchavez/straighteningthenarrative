"""
make_ablation_table_podcast.py
===============================
Podcast version of make_ablation_table.py.

Podcast has no turns/speakers (single narrator, single listener), so the
control sets are just: raw -> + sentence position -> + word length & frequency.

Outputs (in tables/):
  ablation_podcast_per_patient.csv
  ablation_podcast_summary.csv
  ablation_podcast_paired.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import sem as scipy_sem, ttest_1samp, ttest_rel, t as t_dist

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WORD_TABLE = os.path.join(
    BASE_DIR, "podcast", "results", "clean_fig4_podcast_pca20",
    "clean_fig4_podcast_pca20_llama-3.1-8b_lag0_causal_word_table.csv",
)
REGION = "hippocampus"

CONTROL_SETS = {
    "Raw (no controls)":     [],
    "+ sentence position":   ["sent_pos", "sentence_start"],
    "+ length & frequency":  ["sent_pos", "sentence_start", "word_len", "log_freq"],
}


def pstar(p: float) -> str:
    if not np.isfinite(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def ols_beta(y: np.ndarray, sub: pd.DataFrame, controls: list[str]):
    cols = ["surprisal"] + controls
    available = [c for c in cols if c in sub.columns]
    X = sub[available].copy()
    mask = np.isfinite(y) & X.notna().all(axis=1)
    y  = np.asarray(y, dtype=float)[mask]
    X  = X[mask].to_numpy(float)
    n  = len(y)
    if n < len(available) + 3:
        return float("nan"), float("nan")
    y  = (y  - y.mean())  / (y.std()  + 1e-12)
    X  = (X  - X.mean(0)) / (X.std(0) + 1e-12)
    Xm = np.column_stack([np.ones(n), X])
    b, _, _, _ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ b
    s2 = np.dot(resid, resid) / max(n - Xm.shape[1], 1)
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
        tv = b / np.where(se > 0, se, np.nan)
        pv = 2 * t_dist.sf(np.abs(tv), df=max(n - Xm.shape[1], 1))
    except np.linalg.LinAlgError:
        pv = np.full(len(b), np.nan)
    return float(b[1]), float(pv[1])


def main():
    wt = pd.read_csv(WORD_TABLE)
    wt = wt[wt["region"] == REGION].copy()

    per_pat_rows = []
    for pid, sub in wt.groupby("pid"):
        sub = sub.copy()
        if len(sub) < 50:
            continue
        for ctrl_name, ctrl_vars in CONTROL_SETS.items():
            for col, space, predictor in [
                ("c_sem", "Semantic", "Curvature (κ)"),
                ("d_sem", "Semantic", "Displacement (d)"),
                ("c_neu", "Neural",   "Curvature (κ)"),
                ("d_neu", "Neural",   "Displacement (d)"),
            ]:
                beta, pv = ols_beta(sub[col].values, sub, ctrl_vars)
                per_pat_rows.append({
                    "pid": pid, "space": space, "predictor": predictor,
                    "control_set": ctrl_name, "beta": beta, "p_value": pv,
                    "sig": pstar(pv),
                })

    per_pat = pd.DataFrame(per_pat_rows)

    summary_rows = []
    for (space, predictor, ctrl_name), grp in per_pat.groupby(
            ["space", "predictor", "control_set"], sort=False):
        betas = grp["beta"].dropna().values
        if len(betas) < 2:
            continue
        mu = betas.mean()
        se = scipy_sem(betas)
        _, p = ttest_1samp(betas, 0)
        summary_rows.append({
            "space": space, "predictor": predictor, "control_set": ctrl_name,
            "n_patients": len(betas), "mean_beta": round(mu, 4),
            "sem_beta": round(se, 4), "p_vs_zero": round(p, 4), "sig": pstar(p),
        })
    summary = pd.DataFrame(summary_rows)

    paired_rows = []
    for (space, ctrl_name), grp in per_pat.groupby(["space", "control_set"], sort=False):
        curv = grp[grp["predictor"] == "Curvature (κ)"].set_index("pid")["beta"]
        disp = grp[grp["predictor"] == "Displacement (d)"].set_index("pid")["beta"]
        shared = curv.index.intersection(disp.index)
        bc = curv.loc[shared].values
        bd = disp.loc[shared].values
        valid = np.isfinite(bc) & np.isfinite(bd)
        if valid.sum() < 3:
            continue
        _, p_diff = ttest_rel(bc[valid], bd[valid])
        paired_rows.append({
            "space": space, "control_set": ctrl_name, "n_patients": valid.sum(),
            "mean_curv": round(bc[valid].mean(), 4), "mean_disp": round(bd[valid].mean(), 4),
            "p_curv_vs_disp": round(p_diff, 4), "sig": pstar(p_diff),
        })
    paired = pd.DataFrame(paired_rows)

    out_dir = os.path.join(BASE_DIR, "tables")
    os.makedirs(out_dir, exist_ok=True)
    per_pat.to_csv(os.path.join(out_dir, "ablation_podcast_per_patient.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "ablation_podcast_summary.csv"), index=False)
    paired.to_csv(os.path.join(out_dir, "ablation_podcast_paired.csv"), index=False)

    ctrl_order = list(CONTROL_SETS.keys())

    print("\n" + "="*70)
    print(f"PODCAST — Surprisal -> geometry std beta, group mean +/- SEM (n patients={wt['pid'].nunique()})")
    print("="*70)
    for space in ["Semantic", "Neural"]:
        print(f"\n  [{space}]")
        sub = summary[summary["space"] == space].copy()
        sub["ctrl_order"] = sub["control_set"].apply(lambda x: ctrl_order.index(x))
        sub = sub.sort_values(["predictor", "ctrl_order"])
        print(sub[["predictor","control_set","n_patients",
                   "mean_beta","sem_beta","p_vs_zero","sig"]].to_string(index=False))

    print("\n" + "="*70)
    print("PODCAST — Curvature vs Displacement paired t-test per control set")
    print("="*70)
    p2 = paired.copy()
    p2["ctrl_order"] = p2["control_set"].apply(lambda x: ctrl_order.index(x))
    p2 = p2.sort_values(["space", "ctrl_order"])
    print(p2[["space","control_set","n_patients",
              "mean_curv","mean_disp","p_curv_vs_disp","sig"]].to_string(index=False))

    print(f"\nCSVs written to: {out_dir}/")


if __name__ == "__main__":
    main()
