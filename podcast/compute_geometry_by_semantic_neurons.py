"""
compute_geometry_by_semantic_neurons.py
==========================================
Geometry by neuron subset: do semantic-encoding neurons carry the
semantic-neural geometric alignment at gpt2-xl layer 29?

Inputs (from compute_semantic_neuron_encoding_fixed250.py):
  results/semantic_neuron_encoding_fixed250.json   per-unit dD2, p, q
  results/semantic_neuron_pseudopop_noYFX.npz      dur80/40 rate vectors,
                                                   12 patients (no YFX)

Arg: "poisson" (default) or "ridge" (compute_semantic_neuron_encoding_ridge_fixed250.py;
splits "shift"/"shuffle": sem = p < 0.05 vs. non-sem = p >= 0.05 under that null).

Poisson splits (among units with an encoding fit):
  "fdr":  sem = q < 0.05            vs. non-sem = p >= 0.05
  "rank": sem = top quartile by dD2 vs. non-sem = bottom half by dD2
Non-sem is subsampled to the sem count N_SUB times (matched size).
"all": all fitted units, for reference.

Null (argv[2], default "shuffle"): chunk-mean order permutation for
curv/disp/pos; Mantel word-label permutation for RSA. "shift" = circular shift
(no RSA null).

Per neuron set (rate vectors z-scored per unit, same as the pseudopop
geometry analyses):
  - curv/disp/pos at top-K PCs, K in KS (K < n units), chunk=3, valid-word
    mask; circular-shift null for sem and all sets
  - RSA: Pearson r between semantic (layer 29) and neural cosine distances
    over all valid-word pairs

Saves: results/geometry_by_semantic_neurons.json

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed --no-capture-output python3 -u compute_geometry_by_semantic_neurons.py
"""
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import load_podcast_patient
from compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop import (
    REF_PID, REGION, MODEL, LAYER, CHUNK_SIZE, METRICS, pca_pool, corr3, Nuller,
)

KS = [4, 8, 16, 32]
N_SUB = 50
SEED = 0
ENC = sys.argv[1] if len(sys.argv) > 1 else "poisson"   # "poisson" or "ridge"
ENC_PATH = os.path.join(RESULTS, {"poisson": "semantic_neuron_encoding_fixed250.json",
                                  "ridge": "semantic_neuron_encoding_ridge_fixed250.json"}[ENC])
CACHE_PATH = os.path.join(RESULTS, "semantic_neuron_pseudopop_noYFX.npz")
NULL_MODE = sys.argv[2] if len(sys.argv) > 2 else "shuffle"   # "shuffle" or "shift"
N_RSA_NULL = 200
OUT_PATH = os.path.join(RESULTS, f"geometry_by_semantic_neurons_{ENC}_{NULL_MODE}null.json")


def cos_dist_upper(X, iu):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    sim = Xn @ Xn.T
    return 1.0 - sim[iu]


def rsa_null(X, d_sem, iu, rng):
    """Mantel: permute word labels of the neural distance matrix (shuffle mode)."""
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    D = 1.0 - Xn @ Xn.T
    null = np.empty(N_RSA_NULL)
    for i in range(N_RSA_NULL):
        pm = rng.permutation(len(D))
        null[i] = np.corrcoef(d_sem, D[np.ix_(pm, pm)][iu])[0, 1]
    return null


def evaluate(Z, units, valid, Ys, d_sem, iu, nuller=None, rng=None):
    Zu = Z[:, units]
    Yn, _ = pca_pool(Zu, k=max(KS))
    out = dict(n_units=len(units), byK={})
    for K in KS:
        if K >= len(units) or K > Yn.shape[1]:
            continue
        r = corr3(Ys[:, :K], Yn[:, :K], valid, nuller)
        out["byK"][str(K)] = r
    d_neu = cos_dist_upper(Zu[np.where(valid)[0]], iu)
    out["rsa_r"] = float(np.corrcoef(d_sem, d_neu)[0, 1])
    if nuller is not None and NULL_MODE == "shuffle":
        nul = rsa_null(Zu[np.where(valid)[0]], d_sem, iu, rng)
        out.update(rsa_null_lo=float(np.percentile(nul, 2.5)), rsa_null_hi=float(np.percentile(nul, 97.5)),
                   rsa_p=float((np.sum(np.abs(nul) >= abs(out["rsa_r"])) + 1) / (N_RSA_NULL + 1)))
    return out


def main():
    enc = json.load(open(ENC_PATH))
    c = np.load(CACHE_PATH, allow_pickle=True)
    rate, common = c["rate"], c["common"]
    valid = build_valid_mask(len(common), c["surp"], c["sent_pos"], c["is_turn"])
    Z = StandardScaler().fit_transform(rate)
    Z = np.nan_to_num(Z)  # silent units -> 0 after z-scoring
    print(f"words={len(common)} valid={int(valid.sum())} units={Z.shape[1]}", flush=True)

    if ENC == "ridge":
        units = enc["units"]
        idx = np.array([u["unit"] for u in units])
        p_sh = np.array([u["p_shift"] for u in units])
        p_sf = np.array([u["p_shuffle"] for u in units])
        splits = {"shift": (idx[p_sh < 0.05], idx[p_sh >= 0.05]),
                  "shuffle": (idx[p_sf < 0.05], idx[p_sf >= 0.05])}
    else:
        units = [u for u in enc["units"] if "p" in u]
        idx = np.array([u["unit"] for u in units])
        dD2 = np.array([u["dD2"] for u in units])
        p = np.array([u["p"] for u in units])
        q = np.array([u["q"] for u in units])
        order = np.argsort(-dD2)
        splits = {"fdr": (idx[q < 0.05], idx[p >= 0.05]),
                  "rank": (idx[order[: len(order) // 4]], idx[order[len(order) // 2:]])}

    data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
    emb = np.array(np.load(data_m["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
    Ys, _ = pca_pool(emb, k=max(KS))
    nv = int(valid.sum())
    iu = np.triu_indices(nv, k=1)
    d_sem = cos_dist_upper(emb[np.where(valid)[0]], iu)

    rng = np.random.default_rng(SEED)
    nuller = Nuller(nv // CHUNK_SIZE, rng, mode=NULL_MODE)

    out = dict(config=dict(KS=KS, N_SUB=N_SUB, LAYER=LAYER, n_valid=nv, n_fitted=len(units), null=NULL_MODE),
               all=evaluate(Z, idx, valid, Ys, d_sem, iu, nuller, rng), splits={})
    a = out["all"]
    print(f"[all] n={a['n_units']} rsa={a['rsa_r']:+.4f} (null 95% {a.get('rsa_null_hi', float('nan')):+.4f}) " +
          " ".join(f"K{K}:curv={v['curv']['r']:+.3f}" for K, v in a["byK"].items()), flush=True)

    for name, (sem, non) in splits.items():
        n_sem, n_non = len(sem), len(non)
        print(f"\n== split {name}: sem={n_sem}  non-sem={n_non} ==", flush=True)
        if min(n_sem, n_non) < 5:
            print("   too few units, skipped")
            out["splits"][name] = dict(n_sem=n_sem, n_non=n_non, skipped=True)
            continue
        # smaller group evaluated whole (with null); larger group subsampled to its size
        small, large = ("sem", "non") if n_sem <= n_non else ("non", "sem")
        grp = dict(sem=sem, non=non)
        s = evaluate(Z, grp[small], valid, Ys, d_sem, iu, nuller, rng)
        subs = [evaluate(Z, rng.choice(grp[large], size=len(grp[small]), replace=False), valid, Ys, d_sem, iu)
                for _ in range(N_SUB)]
        out["splits"][name] = dict(n_sem=n_sem, n_non=n_non, sem_units=sem.tolist(), whole=small,
                                   subsampled=large, **{small: s, large: subs})
        rs = np.array([x["rsa_r"] for x in subs])
        print(f"   RSA  {small} (whole, n={len(grp[small])})={s['rsa_r']:+.4f} "
              f"(null 95% {s.get('rsa_null_hi', float('nan')):+.4f})   {large} subsampled: mean={rs.mean():+.4f} "
              f"[{np.percentile(rs, 2.5):+.4f},{np.percentile(rs, 97.5):+.4f}]  "
              f"frac {large}>={small}={np.mean(rs >= s['rsa_r']):.2f}")
        for K, v in s["byK"].items():
            line = f"   K={K:>2}"
            for m in METRICS:
                nn = np.array([x["byK"][K][m]["r"] for x in subs if K in x["byK"]])
                line += (f"  {m}: {small}={v[m]['r']:+.3f} (null 95% {v[m]['null_hi']:+.3f}) "
                         f"{large}={nn.mean():+.3f}[{np.percentile(nn, 2.5):+.3f},{np.percentile(nn, 97.5):+.3f}]")
            print(line, flush=True)
        json.dump(out, open(OUT_PATH, "w"))

    json.dump(out, open(OUT_PATH, "w"))
    print(f"\n  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
