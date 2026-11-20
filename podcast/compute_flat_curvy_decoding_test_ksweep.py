"""
compute_flat_curvy_decoding_test_ksweep.py
====================================
compute_flat_curvy_decoding_test.py, but sweeping K (PCs on both sides)
instead of fixing K=3 -- the K=3 result found ~zero decoding R^2
everywhere (flat and curvy alike), so before concluding "flatness doesn't
matter" this checks whether the cross-space next-state decoder has ANY
signal at higher K, and whether flat-vs-curvy diverges once it does.

Efficiency: one master PCA per patient per space, fit at K_MAX (capped by
available dims -- neural channel counts are small, ~10-53 per patient),
sliced to each K in KS (nested/hierarchical PCA trick).

Saves: results/flat_curvy_decoding_test_ksweep.json
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
KS = [3, 5, 10, 20, 30, 50]
K_MAX = 100
SEED = 0
TEST_FRAC = 0.2

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS, "flat_curvy_decoding_test_ksweep.json")


def fit_pca_master(X, k_max=K_MAX):
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k_max)
    if max_pc < 2:
        return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def get_patient(pid):
    data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
    if data is None or data["neu_vecs"] is None:
        return None
    n = data["n_words"]
    valid = build_valid_mask(n, data["surp"][:n], data["sent_pos"][:n], data["is_turn"][:n])
    if valid.sum() < 100:
        return None

    Y_neu_master = fit_pca_master(data["neu_vecs"][:n])
    if Y_neu_master is None:
        return None

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    if LAYER >= emb_mm.shape[0]:
        del emb_mm
        return None
    emb_L = np.array(emb_mm[LAYER, :n], dtype=np.float64)
    del emb_mm
    Y_sem_master = fit_pca_master(emb_L)
    if Y_sem_master is None:
        return None

    return Y_neu_master, Y_sem_master, n


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


def run_direction(Y_source_master, Y_other_master, n, k, rng):
    kk_source = min(k, Y_source_master.shape[1])
    kk_other = min(k, Y_other_master.shape[1])
    Y_source = Y_source_master[:, :kk_source]
    Y_other = Y_other_master[:, :kk_other]

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
    return dict(r2_flat=float(r2_flat), r2_curvy=float(r2_curvy), r2_all=float(r2_all), n_total=m)


def main():
    rng = np.random.default_rng(SEED)
    patients = {}
    for pid in PODCAST_PATIENTS:
        rec = get_patient(pid)
        if rec is None:
            print(f"  {pid}: SKIP")
            continue
        patients[pid] = rec
        print(f"  {pid}: neu_master_dims={rec[0].shape[1]}  sem_master_dims={rec[1].shape[1]}", flush=True)

    results = {"neural_to_semantic": {}, "semantic_to_neural": {}}
    for k in KS:
        n2s_flat, n2s_curvy, s2n_flat, s2n_curvy = [], [], [], []
        for pid, (Y_neu_m, Y_sem_m, n) in patients.items():
            out_n2s = run_direction(Y_neu_m, Y_sem_m, n, k, rng)
            out_s2n = run_direction(Y_sem_m, Y_neu_m, n, k, rng)
            if out_n2s is None or out_s2n is None:
                continue
            n2s_flat.append(out_n2s["r2_flat"]); n2s_curvy.append(out_n2s["r2_curvy"])
            s2n_flat.append(out_s2n["r2_flat"]); s2n_curvy.append(out_s2n["r2_curvy"])

        n2s_flat, n2s_curvy = np.array(n2s_flat), np.array(n2s_curvy)
        s2n_flat, s2n_curvy = np.array(s2n_flat), np.array(s2n_curvy)
        results["neural_to_semantic"][k] = dict(
            r2_flat=float(n2s_flat.mean()), r2_curvy=float(n2s_curvy.mean()),
            diff=float(n2s_flat.mean() - n2s_curvy.mean()), n_patients=len(n2s_flat))
        results["semantic_to_neural"][k] = dict(
            r2_flat=float(s2n_flat.mean()), r2_curvy=float(s2n_curvy.mean()),
            diff=float(s2n_flat.mean() - s2n_curvy.mean()), n_patients=len(s2n_flat))

        print(f"\nK={k}", flush=True)
        print(f"  neu->sem: R2_flat={n2s_flat.mean():+.4f}  R2_curvy={n2s_curvy.mean():+.4f}  "
              f"diff={n2s_flat.mean()-n2s_curvy.mean():+.4f}", flush=True)
        print(f"  sem->neu: R2_flat={s2n_flat.mean():+.4f}  R2_curvy={s2n_curvy.mean():+.4f}  "
              f"diff={s2n_flat.mean()-s2n_curvy.mean():+.4f}", flush=True)
        json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
