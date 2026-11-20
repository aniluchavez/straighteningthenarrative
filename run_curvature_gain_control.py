"""
run_curvature_gain_control.py
==============================
Tests whether the surprisal -> neural curvature (c_neu) relationship survives
after controlling for local step magnitude ("gain", d_neu) at the same word.

c_neu is already scale-invariant by construction (angle between two
consecutive displacement vectors, magnitude cancels in the cosine), so gain
cannot mechanically produce curvature. But with noisy high-D trajectories,
larger-magnitude steps are less perturbed (in angle) by isotropic noise than
smaller-magnitude steps, so an apparent surprisal->curvature effect could
still be partly attributable to a concurrent surprisal->magnitude effect
rather than a genuine change in movement direction. This adds d_neu (the
magnitude of the step arriving at the same word) as a covariate and checks
whether the surprisal beta on curvature survives.

Model A (no gain control): c_neu ~ surprisal + c_sem + d_sem + BASE_CONTROLS
Model B (gain-controlled):  c_neu ~ surprisal + c_sem + d_sem + d_neu + BASE_CONTROLS

Uses the same lag0-causal patient tables as run_clean_fig4_tests.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import run_geometry_analysis as geom
from run_clean_fig4_tests import (
    load_existing_layers, build_patient_table, ols_clean,
    BASE_CONTROLS,
)

MODEL  = "llama-3.1-8b"
REGION = "hippocampus"


def main():
    layers = load_existing_layers(MODEL, REGION)
    rows = []
    for pid in geom.PATIENTS:
        if REGION not in geom.PATIENTS[pid] or pid not in layers:
            continue
        table = build_patient_table(pid, REGION, MODEL, layers[pid], surprisal_lag=0, causal=True)
        if table is None or table.empty:
            print(f"  {pid}: skipped (no table)")
            continue

        model_a = ols_clean(
            table["c_neu"],
            {"surprisal": table["surprisal"], "c_sem": table["c_sem"], "d_sem": table["d_sem"],
             **{c: table[c] for c in BASE_CONTROLS if c != "surprisal"}},
        )
        model_b = ols_clean(
            table["c_neu"],
            {"surprisal": table["surprisal"], "c_sem": table["c_sem"], "d_sem": table["d_sem"],
             "d_neu": table["d_neu"],
             **{c: table[c] for c in BASE_CONTROLS if c != "surprisal"}},
        )

        beta_a, p_a = model_a["betas"].get("surprisal", np.nan), model_a["pvals"].get("surprisal", np.nan)
        beta_b, p_b = model_b["betas"].get("surprisal", np.nan), model_b["pvals"].get("surprisal", np.nan)
        beta_gain   = model_b["betas"].get("d_neu", np.nan)
        p_gain      = model_b["pvals"].get("d_neu", np.nan)

        rows.append(dict(pid=pid, n=len(table),
                          beta_surp_nogain=beta_a, p_surp_nogain=p_a,
                          beta_surp_gaincontrol=beta_b, p_surp_gaincontrol=p_b,
                          beta_gain=beta_gain, p_gain=p_gain))
        print(f"  {pid}: n={len(table)}  "
              f"beta_surp(no gain)={beta_a:+.4f} p={p_a:.4g}   "
              f"beta_surp(gain-ctrl)={beta_b:+.4f} p={p_b:.4g}   "
              f"beta_gain={beta_gain:+.4f} p={p_gain:.4g}")

    df = pd.DataFrame(rows)

    print(f"\nn_patients = {len(df)}")
    for col in ["beta_surp_nogain", "beta_surp_gaincontrol", "beta_gain"]:
        vals = df[col].dropna().to_numpy()
        t, p = stats.ttest_1samp(vals, 0.0)
        print(f"  {col}: mean={vals.mean():+.4f}  sem={vals.std(ddof=1)/np.sqrt(len(vals)):.4f}  "
              f"t({len(vals)-1})={t:.3f}  p={p:.4g}")

    diff = (df["beta_surp_gaincontrol"] - df["beta_surp_nogain"]).dropna().to_numpy()
    t, p = stats.ttest_1samp(diff, 0.0)
    print(f"\n  paired diff (gain-ctrl minus no-gain) beta_surp: mean={diff.mean():+.4f}  "
          f"t({len(diff)-1})={t:.3f}  p={p:.4g}")

    out_path = "results/clean_fig4/curvature_gain_control_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
