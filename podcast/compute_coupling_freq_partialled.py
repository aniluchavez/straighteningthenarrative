"""
compute_coupling_freq_partialled.py
===================================
Does HPC-LLM curvature coupling survive partialling word frequency out of
both curvatures?

For each patient, timescale {word (1), chunk (15)}, and surprisal subset
{all, high (top 25% within patient), low (bottom 25%)}:
  r(c_sem, c_neu) under nested covariate sets
    raw            : no covariates
    freq           : partial on freq
    freq+nuis      : partial on freq + dur + pos + isfunc
    freq+surp+nuis : partial on freq + surp + dur + pos + isfunc
Partial r = correlation of OLS residuals of both curvatures on the covariates
(within patient, within subset).

Also pooled nested unique-R2 of c_sem on c_neu beyond each covariate set
(c_neu ~ covars [+ c_sem]), patient-cluster bootstrap.

Same data/curvatures as compute_freq_surprisal_full_grid_test.py
(podcast, hippocampus, gpt2-xl L20, 13 patients).

Saves: results/coupling_freq_partialled.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp, ttest_rel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from compute_freq_surprisal_full_grid_test import (
    get_patient_base, _chunk_mean, zscore, r2_ols, PODCAST_PATIENTS, SEM_MODEL, SEM_LAYER, REGION)

CHUNKS = [1, 15]
HI_PCT, LO_PCT = 75, 25
N_BOOT = 2000
SEED = 0
COVAR_SETS = {
    "raw": [],
    "freq": ["freq"],
    "freq+nuis": ["freq", "dur", "pos", "isfunc"],
    "freq+surp+nuis": ["freq", "surp", "dur", "pos", "isfunc"],
}
COLS = ["c_sem", "c_neu", "freq", "surp", "dur", "pos", "isfunc"]

RESULTS = os.path.join(BASE_DIR, "results")
OUT_PATH = os.path.join(RESULTS, "coupling_freq_partialled.json")


def chunk_patient(d, chunk):
    """Chunk-mean every column on the c_neu chunk grid; return z-scored DataFrame."""
    if chunk == 1:
        out = {c: np.asarray(d[c], dtype=float) for c in COLS}
    else:
        _, idx = _chunk_mean(d["c_neu"], chunk)
        out = {c: np.array([np.nanmean(d[c][i:i + chunk]) for i in idx]) for c in COLS}
    df = pd.DataFrame(out)
    df = df[np.isfinite(df.values).all(axis=1)].reset_index(drop=True)
    return df


def resid(y, X):
    if X.shape[1] == 0:
        return y - y.mean()
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def partial_r(df, covars):
    X = df[covars].values if covars else np.empty((len(df), 0))
    return pearsonr(resid(df.c_sem.values, X), resid(df.c_neu.values, X))[0]


def subsets(df):
    hi, lo = np.percentile(df.surp, [HI_PCT, LO_PCT])
    return {"all": df, "high": df[df.surp >= hi], "low": df[df.surp <= lo]}


def pooled_unique_r2(pooled, covars, rng):
    def _u(sub):
        y = sub.c_neu.values
        X_a = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in covars])
        X_b = np.column_stack([X_a, sub.c_sem.values])
        return r2_ols(y, X_b) - r2_ols(y, X_a)

    obs = _u(pooled)
    groups = {p: g for p, g in pooled.groupby("patient")}
    pts = list(groups)
    boot = []
    for _ in range(N_BOOT):
        samp = rng.choice(pts, size=len(pts), replace=True)
        boot.append(_u(pd.concat([groups[p] for p in samp])))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return dict(unique_r2=float(obs), ci_lo=float(lo), ci_hi=float(hi), p=float(p))


def main():
    patient_data = {}
    for pid in PODCAST_PATIENTS:
        d = get_patient_base(pid)
        if d is None:
            print(f"  {pid}: SKIP", flush=True)
            continue
        patient_data[pid] = d
        print(f"  {pid}: loaded", flush=True)

    rng = np.random.default_rng(SEED)
    results = dict(meta=dict(region=REGION, sem_model=SEM_MODEL, sem_layer=SEM_LAYER,
                             patients=list(patient_data), hi_pct=HI_PCT, lo_pct=LO_PCT,
                             n_boot=N_BOOT, covar_sets=COVAR_SETS))
    for chunk in CHUNKS:
        per_pat = {}  # subset -> covar -> {pid: r}
        pooled_parts = {s: [] for s in ["all", "high", "low"]}
        for pid, d in patient_data.items():
            df = chunk_patient(d, chunk)
            for sname, sub in subsets(df).items():
                subz = sub.apply(zscore)
                subz["patient"] = pid
                pooled_parts[sname].append(subz)
                for cname, covars in COVAR_SETS.items():
                    per_pat.setdefault(sname, {}).setdefault(cname, {})[pid] = float(partial_r(subz, covars))

        block = {}
        print(f"\n=== chunk {chunk} ===", flush=True)
        for sname in ["all", "high", "low"]:
            pooled = pd.concat(pooled_parts[sname], ignore_index=True)
            for cname, covars in COVAR_SETS.items():
                rs = np.array(list(per_pat[sname][cname].values()))
                t, p = ttest_1samp(rs, 0)
                u = pooled_unique_r2(pooled, covars, rng)
                block.setdefault(sname, {})[cname] = dict(
                    per_patient=per_pat[sname][cname], mean_r=float(rs.mean()),
                    sem=float(rs.std(ddof=1) / np.sqrt(len(rs))), t=float(t), p=float(p),
                    n_pos=int((rs > 0).sum()), n=int(len(rs)), n_obs=int(len(pooled)), pooled=u)
                print(f"  {sname:4s} {cname:15s} mean r={rs.mean():+.4f}  t={t:+.2f} p={p:.2g}  "
                      f"{(rs > 0).sum()}/{len(rs)} pos   pooled uR2={u['unique_r2']:.5f} p={u['p']:.3g}",
                      flush=True)
        for cname in COVAR_SETS:
            hi = np.array([block["high"][cname]["per_patient"][p] for p in patient_data])
            lo = np.array([block["low"][cname]["per_patient"][p] for p in patient_data])
            t, p = ttest_rel(hi, lo)
            block.setdefault("high_minus_low", {})[cname] = dict(
                mean_diff=float((hi - lo).mean()), t=float(t), p=float(p))
            print(f"  hi-lo {cname:15s} diff={(hi - lo).mean():+.4f}  t={t:+.2f} p={p:.2g}", flush=True)
        results[f"chunk{chunk}"] = block

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
