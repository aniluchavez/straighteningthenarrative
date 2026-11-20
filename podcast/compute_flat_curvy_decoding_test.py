"""
compute_flat_curvy_decoding_test.py
====================================
Does a FLATTER (low-curvature) local trajectory in one space predict the
NEXT state in the OTHER space better than a CURVIER (high-curvature) one?

For each patient (podcast, gpt2-xl layer 20 -- the layer validated earlier
this session as best shape-matching neural, raw embedding then reduced to
a FEW PCs as requested), both directions:

  neural[t]   -> semantic[t+1]   (does flat/curvy NEURAL trajectory at t
                                   predict the next SEMANTIC state better?)
  semantic[t] -> neural[t+1]     (does flat/curvy SEMANTIC trajectory at t
                                   predict the next NEURAL state better?)

Source-space curvature at t (using points t-1,t,t+1) splits timepoints into
"flat" (below median) / "curvy" (above median). A single linear decoder
(OLS, fit on a random 80% train split, evaluated on the held-out 20%) maps
source[t] -> other[t+1]; R^2 on the held-out test set is computed
separately for flat vs curvy timepoints.

K = 3 PCs on both sides ("few pcs").

Saves: results/flat_curvy_decoding_test.json
"""
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis import trajectory_curvature, build_valid_mask
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

MODEL = "gpt2-xl"
LAYER = 20
REGION = "hippocampus"
K = 3
SEED = 0
TEST_FRAC = 0.2

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS, "flat_curvy_decoding_test.json")


def fit_pca_k(X, k=K):
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k)
    if max_pc < k:
        return None
    return PCA(n_components=k, random_state=0).fit_transform(Xs)


def get_patient(pid):
    data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
    if data is None or data["neu_vecs"] is None:
        return None
    n = data["n_words"]
    valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
    if valid.sum() < 100:
        return None

    Y_neu = fit_pca_k(data["neu_vecs"][:n])
    if Y_neu is None:
        return None

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    if LAYER >= emb_mm.shape[0]:
        del emb_mm
        return None
    emb_L = np.array(emb_mm[LAYER, :n], dtype=np.float64)
    del emb_mm
    Y_sem = fit_pca_k(emb_L)
    if Y_sem is None:
        return None

    return Y_neu, Y_sem, n


def build_decode_rows(Y_source, Y_other, n):
    curv_full = trajectory_curvature(Y_source)
    ts = np.arange(1, n - 1)
    X = Y_source[ts]
    y = Y_other[ts + 1]
    curv = curv_full[ts - 1]
    return X, y, curv


def r2_multi(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def run_direction(Y_source, Y_other, n, rng):
    X, y, curv = build_decode_rows(Y_source, Y_other, n)
    finite = np.isfinite(curv) & np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(y), axis=1)
    X, y, curv = X[finite], y[finite], curv[finite]
    m = len(X)
    if m < 50:
        return None

    idx = rng.permutation(m)
    n_test = int(m * TEST_FRAC)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    reg = LinearRegression().fit(X[train_idx], y[train_idx])
    y_pred_test = reg.predict(X[test_idx])
    curv_test = curv[test_idx]
    med = np.median(curv[train_idx])

    flat_mask = curv_test <= med
    curvy_mask = curv_test > med

    r2_flat = r2_multi(y[test_idx][flat_mask], y_pred_test[flat_mask])
    r2_curvy = r2_multi(y[test_idx][curvy_mask], y_pred_test[curvy_mask])
    r2_all = r2_multi(y[test_idx], y_pred_test)
    return dict(r2_flat=float(r2_flat), r2_curvy=float(r2_curvy), r2_all=float(r2_all),
                n_flat=int(flat_mask.sum()), n_curvy=int(curvy_mask.sum()), n_total=m)


def main():
    rng = np.random.default_rng(SEED)
    results = {"neural_to_semantic": {}, "semantic_to_neural": {}}
    n2s_flat, n2s_curvy, s2n_flat, s2n_curvy = [], [], [], []

    for pid in PODCAST_PATIENTS:
        rec = get_patient(pid)
        if rec is None:
            print(f"  {pid}: SKIP")
            continue
        Y_neu, Y_sem, n = rec

        out_n2s = run_direction(Y_neu, Y_sem, n, rng)
        out_s2n = run_direction(Y_sem, Y_neu, n, rng)
        if out_n2s is None or out_s2n is None:
            print(f"  {pid}: SKIP (insufficient rows)")
            continue

        results["neural_to_semantic"][pid] = out_n2s
        results["semantic_to_neural"][pid] = out_s2n
        n2s_flat.append(out_n2s["r2_flat"]); n2s_curvy.append(out_n2s["r2_curvy"])
        s2n_flat.append(out_s2n["r2_flat"]); s2n_curvy.append(out_s2n["r2_curvy"])
        print(f"  {pid}: neu->sem  flat={out_n2s['r2_flat']:+.4f} curvy={out_n2s['r2_curvy']:+.4f}   "
              f"sem->neu  flat={out_s2n['r2_flat']:+.4f} curvy={out_s2n['r2_curvy']:+.4f}", flush=True)

    def summarize(flat, curvy, label):
        flat, curvy = np.array(flat), np.array(curvy)
        print(f"\n{label}: mean R2_flat={flat.mean():+.4f}  mean R2_curvy={curvy.mean():+.4f}  "
              f"diff (flat-curvy)={flat.mean()-curvy.mean():+.4f}  n_patients={len(flat)}")

    summarize(n2s_flat, n2s_curvy, "neural[t] -> semantic[t+1]")
    summarize(s2n_flat, s2n_curvy, "semantic[t] -> neural[t+1]")

    json.dump(results, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
