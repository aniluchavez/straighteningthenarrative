"""
compute_samestep_decoding_ceiling_test.py
====================================
Every next-state (X[t] -> Y[t+1]) cross-space decoding test this session
maxed out around R^2 ~ 0.001-0.02 -- is that a real ceiling, or does
requiring next-STEP prediction (on top of cross-space mapping) understate
what's decodable? This compares SAME-timestep decoding (X[t] -> Y[t], the
standard "can we decode at all" test) against next-timestep (X[t] ->
Y[t+1]) side by side, same pseudopopulation neural (475 channels, 13
patients, dur80/40 rate) + YEY's gpt2-xl layer 20 semantic, across a wide
K sweep (including K up to 300 now that there's no flat/curvy split
halving the test-set size).

Saves: results/samestep_decoding_ceiling_test.json
"""
import json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient
from compute_matched_and_pseudopop_dur8040 import align_to_ref, get_neural

MODEL = "gpt2-xl"
LAYER = 20
REGION = "hippocampus"
KS = [3, 10, 20, 50, 100, 150, 200, 300]
K_MAX = 300
N_REPEATS = 30
SEED = 0
TEST_FRAC = 0.2
REF_PID = "YEY"

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS, "samestep_decoding_ceiling_test.json")


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


def r2_multi(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def run_lag(Y_source_master, Y_other_master, n, k, lag, n_repeats=N_REPEATS, seed=SEED):
    kk_source = min(k, Y_source_master.shape[1])
    kk_other = min(k, Y_other_master.shape[1])
    Y_source = Y_source_master[:, :kk_source]
    Y_other = Y_other_master[:, :kk_other]

    if lag == 0:
        X, y = Y_source, Y_other
    else:
        X = Y_source[:-lag]
        y = Y_other[lag:]
    finite = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(y), axis=1)
    X, y = X[finite], y[finite]
    m = len(X)

    rng = np.random.default_rng(seed)
    r2s = []
    for _ in range(n_repeats):
        idx = rng.permutation(m)
        n_test = int(m * TEST_FRAC)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        reg = LinearRegression().fit(X[train_idx], y[train_idx])
        r2s.append(r2_multi(y[test_idx], reg.predict(X[test_idx])))
    r2s = np.array(r2s)
    return dict(r2_mean=float(r2s.mean()), r2_se=float(r2s.std() / np.sqrt(n_repeats)), n_total=m)


def main():
    Y_neu_master, Y_sem_master, n = build_pseudopop()

    results = {"neural_to_semantic": {"lag0": {}, "lag1": {}}, "semantic_to_neural": {"lag0": {}, "lag1": {}}}
    for k in KS:
        for lag, tag in ((0, "lag0"), (1, "lag1")):
            out_n2s = run_lag(Y_neu_master, Y_sem_master, n, k, lag)
            out_s2n = run_lag(Y_sem_master, Y_neu_master, n, k, lag)
            results["neural_to_semantic"][tag][k] = out_n2s
            results["semantic_to_neural"][tag][k] = out_s2n
            print(f"K={k:4d}  {tag}  neu->sem: R2={out_n2s['r2_mean']:+.4f}+-{out_n2s['r2_se']:.4f}   "
                  f"sem->neu: R2={out_s2n['r2_mean']:+.4f}+-{out_s2n['r2_se']:.4f}", flush=True)
        json.dump(results, open(OUT_PATH, "w"), indent=2)

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
