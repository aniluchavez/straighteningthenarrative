"""
compute_flat_curvy_decoding_test_pseudopop.py
====================================
compute_flat_curvy_decoding_test_ksweep.py, but the neural side is the
PSEUDOPOPULATION (all 13 patients' hippocampal channels concatenated
word-by-word via the difflib-aligned common backbone, same construction as
compute_matched_and_pseudopop_dur8040.py -- dur80/40 rate-normalized,
7317/7361 common words, 475 total channels) instead of one patient's own
~10-53 channels. This is a single continuous series (not per-patient), so
each K's R^2 is repeated over multiple random train/test splits for a CI
instead of pooling across patients.

Semantic side: reference patient (YEY)'s gpt2-xl layer 20 embedding at the
same common word backbone.

Saves: results/flat_curvy_decoding_test_pseudopop.json
"""
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis import trajectory_curvature
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)
from compute_matched_and_pseudopop_dur8040 import align_to_ref, get_neural

MODEL = "gpt2-xl"
LAYER = 20
REGION = "hippocampus"
KS = [3, 5, 10, 20, 30, 50, 100, 150]
K_MAX = 300
N_REPEATS = 20
SEED = 0
TEST_FRAC = 0.2
REF_PID = "YEY"

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS, "flat_curvy_decoding_test_pseudopop.json")


def fit_pca_master(X, k_max=K_MAX):
    Xs = StandardScaler().fit_transform(X)
    max_pc = min(Xs.shape[1], Xs.shape[0] - 1, k_max)
    if max_pc < 2:
        return None
    return PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)


def build_pseudopop():
    neural = {}
    for pid in PODCAST_PATIENTS:
        rec = get_neural(pid)
        if rec is not None:
            neural[pid] = rec
    print(f"{len(neural)} patients loaded", flush=True)

    ref_words = neural[REF_PID]["words"].tolist()
    common = set(range(len(ref_words)))
    mappings = {}
    for pid, rec in neural.items():
        m = align_to_ref(ref_words, rec["words"].tolist())
        mappings[pid] = m
        common &= set(m.keys())
    common_sorted = sorted(common)
    print(f"common words across {len(neural)} patients: {len(common_sorted)}/{len(ref_words)}", flush=True)

    pids = list(neural.keys())
    combined = []
    for pid in pids:
        m = mappings[pid]
        local_idx = [m[r] for r in common_sorted]
        combined.append(neural[pid]["neu_vecs"][local_idx])
    combined = np.concatenate(combined, axis=1)
    print(f"combined neural shape: {combined.shape}", flush=True)

    Y_neu_master = fit_pca_master(combined)
    print(f"neural master dims: {Y_neu_master.shape[1]}", flush=True)

    data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
    emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
    emb_L = np.array(emb_mm[LAYER, common_sorted], dtype=np.float64)
    del emb_mm
    Y_sem_master = fit_pca_master(emb_L)
    print(f"semantic master dims: {Y_sem_master.shape[1]}", flush=True)

    n_common = len(common_sorted)
    return Y_neu_master, Y_sem_master, n_common


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


def run_direction(Y_source_master, Y_other_master, n, k, n_repeats=N_REPEATS, seed=SEED):
    kk_source = min(k, Y_source_master.shape[1])
    kk_other = min(k, Y_other_master.shape[1])
    Y_source = Y_source_master[:, :kk_source]
    Y_other = Y_other_master[:, :kk_other]

    X, y, curv = build_decode_rows(Y_source, Y_other, n)
    finite = np.isfinite(curv) & np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(y), axis=1)
    X, y, curv = X[finite], y[finite], curv[finite]
    m = len(X)

    rng = np.random.default_rng(seed)
    r2_flats, r2_curvys = [], []
    for _ in range(n_repeats):
        idx = rng.permutation(m)
        n_test = int(m * TEST_FRAC)
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        reg = LinearRegression().fit(X[train_idx], y[train_idx])
        y_pred_test = reg.predict(X[test_idx])
        curv_test = curv[test_idx]
        med = np.median(curv[train_idx])

        flat_mask = curv_test <= med
        curvy_mask = curv_test > med
        r2_flats.append(r2_multi(y[test_idx][flat_mask], y_pred_test[flat_mask]))
        r2_curvys.append(r2_multi(y[test_idx][curvy_mask], y_pred_test[curvy_mask]))

    r2_flats, r2_curvys = np.array(r2_flats), np.array(r2_curvys)
    return dict(r2_flat_mean=float(r2_flats.mean()), r2_flat_se=float(r2_flats.std() / np.sqrt(n_repeats)),
                r2_curvy_mean=float(r2_curvys.mean()), r2_curvy_se=float(r2_curvys.std() / np.sqrt(n_repeats)),
                diff=float(r2_flats.mean() - r2_curvys.mean()), n_total=m)


def main():
    Y_neu_master, Y_sem_master, n = build_pseudopop()

    results = {"neural_to_semantic": {}, "semantic_to_neural": {}}
    for k in KS:
        out_n2s = run_direction(Y_neu_master, Y_sem_master, n, k)
        out_s2n = run_direction(Y_sem_master, Y_neu_master, n, k)
        results["neural_to_semantic"][k] = out_n2s
        results["semantic_to_neural"][k] = out_s2n
        print(f"\nK={k}", flush=True)
        print(f"  neu->sem: R2_flat={out_n2s['r2_flat_mean']:+.4f}+-{out_n2s['r2_flat_se']:.4f}  "
              f"R2_curvy={out_n2s['r2_curvy_mean']:+.4f}+-{out_n2s['r2_curvy_se']:.4f}  diff={out_n2s['diff']:+.4f}", flush=True)
        print(f"  sem->neu: R2_flat={out_s2n['r2_flat_mean']:+.4f}+-{out_s2n['r2_flat_se']:.4f}  "
              f"R2_curvy={out_s2n['r2_curvy_mean']:+.4f}+-{out_s2n['r2_curvy_se']:.4f}  diff={out_s2n['diff']:+.4f}", flush=True)
        json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
