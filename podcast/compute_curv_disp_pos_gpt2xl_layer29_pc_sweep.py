"""
compute_curv_disp_pos_gpt2xl_layer29_pc_sweep.py
==================================================
Why do curvature / displacement / position correlations change with the
PCs used? Extends compute_curv_disp_pos_gpt2xl_layer29.py (top-8 PCs both
sides, chunk=3, dur80/40, raw counts, gpt2-xl layer 29) with two PC
manipulations, all drawn from one master PCA per side per patient
(standardized, top K_POOL=50 components; neural pool capped by channel
count and n-1):

  1. Random PC sweep -- N_DRAWS draws; each draw independently picks K=8
     random PCs from the semantic pool and K from the neural pool
     (independently per patient). Position = coordinate on the FIRST drawn
     PC (random). Per-PC inclusion effect: mean group r over draws that
     include PC j minus draws that exclude it, per side and metric.

  2. PC dropout -- drop the top j PCs, use PCs j..j+K-1. Three schemes:
       "both": same offset on both sides
       "sem":  semantic offset j, neural fixed top-K
       "neu":  neural offset j, semantic fixed top-K

Reference "top8" (offset 0 on both) should reproduce
results/curv_disp_pos_gpt2xl_layer29.json.

Saves: results/curv_disp_pos_gpt2xl_layer29_pc_sweep.json

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed --no-capture-output python3 -u compute_curv_disp_pos_gpt2xl_layer29_pc_sweep.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

K = 8
K_POOL = 50
CHUNK_SIZE = 3
MODEL = "gpt2-xl"
LAYER = 29
N_DRAWS = 300
SEED = 0
METRICS = ("curv", "disp", "pos")
OUT_PATH = os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_pc_sweep.json")


def pca_pool(X, k=K_POOL):
    X = np.asarray(X, dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k)
    p = PCA(n_components=max_pc, svd_solver="randomized", random_state=0)
    return p.fit_transform(Xs), p.explained_variance_ratio_


def curvature(Y):
    # vectorized trajectory_curvature (degrees; zero-norm steps -> 0)
    adj = np.diff(Y, axis=0)
    a, b = adj[:-1], adj[1:]
    nrm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    dot = np.einsum("ij,ij->i", a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.degrees(np.arccos(np.clip(dot / nrm, -1.0, 1.0)))
    out[nrm == 0] = 0.0
    return out


def chunk_r(x, y, chunk_size=CHUNK_SIZE, min_chunks=3):
    fin = np.isfinite(x) & np.isfinite(y)
    n = len(x)
    xa, ya = [], []
    for i in range(0, n - chunk_size + 1, chunk_size):
        sl = slice(i, i + chunk_size)
        f = fin[sl]
        if f.sum() < chunk_size // 2:
            continue
        xa.append(np.mean(x[sl][f])); ya.append(np.mean(y[sl][f]))
    if len(xa) < min_chunks:
        return float("nan")
    r, _ = pearsonr(xa, ya)
    return float(r)


def features(Y, valid):
    """curv/disp/pos masked exactly as in compute_curv_disp_pos_gpt2xl_layer29.py."""
    c = curvature(Y)
    d = np.linalg.norm(np.diff(Y, axis=0), axis=1)
    p = Y[:, 0]
    out = {}
    for name, f in (("curv", c), ("disp", d), ("pos", p)):
        m = min(len(f), len(valid))
        out[name] = f[:m][valid[:m]]
    return out


def corr3(Ys, Yn, valid):
    fs, fn = features(Ys, valid), features(Yn, valid)
    res = {}
    for m in METRICS:
        n = min(len(fs[m]), len(fn[m]))
        res[m] = chunk_r(fn[m][:n], fs[m][:n])
    return res


def group_stats(rs):
    """rs: list of per-patient dicts -> per-metric group summary."""
    out = {}
    for m in METRICS:
        v = np.array([r[m] for r in rs], dtype=float)
        v = v[np.isfinite(v)]
        t, p = ttest_1samp(v, 0.0) if len(v) >= 2 else (np.nan, np.nan)
        out[m] = dict(mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                      t=float(t), p=float(p), n_pos=int((v > 0).sum()), n=int(len(v)))
    return out


def load_patients():
    pts = {}
    for pid in PODCAST_PATIENTS:
        try:
            _, onset_ms, offset_ms = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
        if data is None:
            continue
        n0 = min(data["n_words"], len(onset_ms))
        preg = _convo_regions(pid)
        neu_vecs = windowed_neu_vecs(pid, "hippocampus", preg, onset_ms[:n0], offset_ms[:n0], 80, 40, rate=False)
        if neu_vecs is None:
            continue
        n = min(data["n_words"], n0)
        valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
        if valid.sum() < 20:
            continue
        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if LAYER >= emb_mm.shape[0]:
            continue
        emb_l = np.array(emb_mm[LAYER, :n], dtype=np.float64)
        del emb_mm
        Ys, evr_s = pca_pool(emb_l)
        Yn, evr_n = pca_pool(neu_vecs[:n])
        pts[pid] = dict(Ys=Ys, Yn=Yn, valid=valid, evr_s=evr_s, evr_n=evr_n,
                        n_ch=int(neu_vecs.shape[1]))
        print(f"  loaded {pid}: n={n}  valid={int(valid.sum())}  channels={neu_vecs.shape[1]}  "
              f"sem_pool={Ys.shape[1]}  neu_pool={Yn.shape[1]}", flush=True)
    return pts


def main():
    pts = load_patients()
    pids = list(pts)
    out = dict(config=dict(K=K, K_POOL=K_POOL, CHUNK_SIZE=CHUNK_SIZE, MODEL=MODEL, LAYER=LAYER,
                           N_DRAWS=N_DRAWS, SEED=SEED),
               patients={p: dict(n_ch=pts[p]["n_ch"], sem_pool=int(pts[p]["Ys"].shape[1]),
                                 neu_pool=int(pts[p]["Yn"].shape[1]),
                                 evr_sem=pts[p]["evr_s"].tolist(), evr_neu=pts[p]["evr_n"].tolist())
                         for p in pids})

    # reference: top-K both sides
    ref = {p: corr3(pts[p]["Ys"][:, :K], pts[p]["Yn"][:, :K], pts[p]["valid"]) for p in pids}
    out["top8"] = dict(per_patient=ref, group=group_stats(list(ref.values())))
    print("\n=== reference top-8 ===")
    for p in pids:
        print(f"  {p}: " + "  ".join(f"{m}={ref[p][m]:+.4f}" for m in METRICS))
    for m, s in out["top8"]["group"].items():
        print(f"  {m}: mean={s['mean']:+.4f} t={s['t']:.2f} n_pos={s['n_pos']}/{s['n']}")

    # 1. random PC sweep
    rng = np.random.default_rng(SEED)
    draws = []
    for d in range(N_DRAWS):
        per_pt, idx = {}, {}
        for p in pids:
            Ys, Yn = pts[p]["Ys"], pts[p]["Yn"]
            si = rng.choice(Ys.shape[1], size=K, replace=False)
            ni = rng.choice(Yn.shape[1], size=min(K, Yn.shape[1]), replace=False)
            per_pt[p] = corr3(Ys[:, si], Yn[:, ni], pts[p]["valid"])
            idx[p] = dict(sem=si.tolist(), neu=ni.tolist())
        g = group_stats(list(per_pt.values()))
        draws.append(dict(per_patient=per_pt, idx=idx, group=g,
                          winner=max(METRICS, key=lambda m: g[m]["mean"])))
        if (d + 1) % 25 == 0:
            print(f"  draw {d + 1}/{N_DRAWS}", flush=True)

    # per-PC inclusion effect: patient-level r, included vs excluded
    incl = {}
    for side, pool_key in (("sem", "Ys"), ("neu", "Yn")):
        incl[side] = {}
        for m in METRICS:
            eff = []
            for j in range(K_POOL):
                w, wo = [], []
                for dr in draws:
                    for p in pids:
                        if j >= pts[p][pool_key].shape[1]:
                            continue
                        r = dr["per_patient"][p][m]
                        if not np.isfinite(r):
                            continue
                        (w if j in dr["idx"][p][side] else wo).append(r)
                eff.append(float(np.mean(w) - np.mean(wo)) if w and wo else float("nan"))
            incl[side][m] = eff

    summ = {}
    for m in METRICS:
        v = np.array([dr["group"][m]["mean"] for dr in draws])
        summ[m] = dict(mean=float(v.mean()), sd=float(v.std()), p2_5=float(np.percentile(v, 2.5)),
                       p97_5=float(np.percentile(v, 97.5)), frac_pos=float((v > 0).mean()),
                       frac_win=float(np.mean([dr["winner"] == m for dr in draws])))
    out["random"] = dict(draws=draws, summary=summ, inclusion_effect=incl)
    print(f"\n=== random {K} of {K_POOL} PCs, both sides independent, {N_DRAWS} draws ===")
    for m, s in summ.items():
        print(f"  {m}: group-mean r over draws {s['mean']:+.4f} (sd {s['sd']:.4f}, 95% [{s['p2_5']:+.4f}, "
              f"{s['p97_5']:+.4f}])  frac>0={s['frac_pos']:.2f}  wins={s['frac_win']:.2f}")
    for side in ("sem", "neu"):
        for m in METRICS:
            e = np.array(incl[side][m])
            top = np.argsort(-np.abs(np.nan_to_num(e)))[:5]
            print(f"  inclusion effect {side}/{m}: largest |effect| PCs " +
                  ", ".join(f"PC{j + 1}:{e[j]:+.4f}" for j in top))

    # 2. PC dropout
    out["dropout"] = {}
    print("\n=== PC dropout (use PCs j+1..j+K) ===")
    for scheme in ("both", "sem", "neu"):
        rows = []
        for j in range(K_POOL - K + 1):
            per_pt = {}
            for p in pids:
                Ys, Yn = pts[p]["Ys"], pts[p]["Yn"]
                js = j if scheme in ("both", "sem") else 0
                jn = j if scheme in ("both", "neu") else 0
                if js + K > Ys.shape[1] or jn + K > Yn.shape[1]:
                    continue
                per_pt[p] = corr3(Ys[:, js:js + K], Yn[:, jn:jn + K], pts[p]["valid"])
            if len(per_pt) < 3:
                break
            g = group_stats(list(per_pt.values()))
            rows.append(dict(offset=j, per_patient=per_pt, group=g))
            print(f"  [{scheme}] drop {j:2d}: n={len(per_pt):2d}  " +
                  "  ".join(f"{m}={g[m]['mean']:+.4f}(t={g[m]['t']:.1f})" for m in METRICS), flush=True)
        out["dropout"][scheme] = rows

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print(f"\n  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
