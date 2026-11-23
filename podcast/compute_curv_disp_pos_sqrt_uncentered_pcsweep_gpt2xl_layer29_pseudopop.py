"""
compute_curv_disp_pos_sqrt_uncentered_pcsweep_gpt2xl_layer29_pseudopop.py
==============================================================
UNCENTERED top-K PCA (TruncatedSVD on raw x, origin kept at 0, so |y| ~ magnitude) version of compute_curv_disp_pos_sqrt_full_gpt2xl_layer29_pseudopop.py.

Same inputs (pseudopop 2*sqrt(count), 12 patients / 436 units; GPT-2 XL L29 raw),
then centered, UNSCALED PCA per space (keeps the sqrt / raw geometry), top-K
PCs, same K in both spaces. Curvature, symmetrised displacement, |x - centroid|,
|x| computed in the K-D PC space. "full" = no PCA (reference).
Reports r per K and the mean r across the K values.

Inference: circular shift of the neural chunk-mean series, N_SHIFT shifts.

Saves: results/curv_disp_pos_sqrt_uncentered_pcsweep_gpt2xl_layer29_pseudopop.json

    conda run -n gpt2_embed --no-capture-output python3 -u compute_curv_disp_pos_sqrt_pcsweep_gpt2xl_layer29_pseudopop.py
"""
import json
import os
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)
from compute_curv_disp_pos_sqrt_full_gpt2xl_layer29 import quantities, neu_transform, sem_transform, METRICS
from compute_curv_disp_pos_sqrt_full_gpt2xl_layer29_pseudopop import load_pseudopop, chunk_means, zs

KS = [2, 4, 8, 16, 32, 64, 128, 256]
N_SHIFT, SEED = 2000, 0
RESULTS = os.path.join(BASE_DIR, "results")
OUT = os.path.join(RESULTS, "curv_disp_pos_sqrt_uncentered_pcsweep_gpt2xl_layer29_pseudopop.json")


def stats(qn, qs, valid, rng):
    res = {}
    for m in METRICS:
        xn, xs = chunk_means(qn[m][valid], qs[m][valid])
        a, b = zs(xn), zs(xs)
        r = float(np.mean(a * b))
        L = len(a)
        sh = rng.integers(int(0.1 * L), L - int(0.1 * L), size=N_SHIFT)
        null = np.array([np.mean(np.roll(a, s) * b) for s in sh])
        res[m] = dict(r=r, p=float((np.sum(np.abs(null) >= abs(r)) + 1) / (N_SHIFT + 1)))
    return res


def main():
    sem, K, common, valid, patients, _ = load_pseudopop()
    S, Nn = sem_transform(sem), neu_transform(K)
    pS = TruncatedSVD(n_components=max(KS), random_state=0).fit(S)
    pN = TruncatedSVD(n_components=max(KS), random_state=0).fit(Nn)
    YS, YN = pS.transform(S), pN.transform(Nn)
    evS = np.cumsum((YS ** 2).sum(0)) / (S ** 2).sum()   # share of sum ||x||^2 kept
    evN = np.cumsum((YN ** 2).sum(0)) / (Nn ** 2).sum()
    rng = np.random.default_rng(SEED)
    out = {"patients": patients, "n_units": int(K.shape[1]), "n_valid": int(valid.sum()), "by_k": {}}
    print(f"{'K':>5} {'varS':>5} {'varN':>5}  " + "  ".join(f"{m:>15}" for m in METRICS))
    for k in KS + ["full"]:
        if k == "full":
            qs, _ = quantities(S); qn, _ = quantities(Nn); vs = vn = 1.0
        else:
            qs, _ = quantities(YS[:, :k]); qn, _ = quantities(YN[:, :k]); vs, vn = evS[k - 1], evN[k - 1]
        res = stats(qn, qs, valid, rng)
        res["var_explained"] = {"sem": float(vs), "neu": float(vn)}
        out["by_k"][str(k)] = res
        print(f"{str(k):>5} {vs:5.2f} {vn:5.2f}  " +
              "  ".join(f"{res[m]['r']:+.4f} (p={res[m]['p']:.3f})" for m in METRICS), flush=True)
    out["mean_over_k"] = {m: float(np.mean([out["by_k"][str(k)][m]["r"] for k in KS])) for m in METRICS}
    print(f"\nmean r over K={KS}: " + "  ".join(f"{m}={v:+.4f}" for m, v in out["mean_over_k"].items()))
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
