import sys, os
sys.path.insert(0, "/scratch/aniluchavez/straighteningthenarrative")
sys.path.insert(0, "/scratch/aniluchavez/straighteningthenarrative/podcast")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from run_geometry_analysis import trajectory_curvature, build_valid_mask
from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

MODEL = "llama-3.1-8b"
K = 20
N_PERM = 50
rng = np.random.default_rng(0)

curv_means, null_means = [], []
for pid in PODCAST_PATIENTS:
    data0 = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
    if data0 is None:
        print(f"  {pid}: SKIP")
        continue
    n = data0["n_words"]
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    emb_mm = np.load(data0["emb_path"], mmap_mode="r")
    n_layers = emb_mm.shape[0]

    layer_curv, layer_null = [], []
    for L in range(n_layers):
        emb_L = np.array(emb_mm[L, :n], dtype=np.float64)
        Xs = StandardScaler().fit_transform(emb_L)
        max_pc = min(K, Xs.shape[1], Xs.shape[0]-1)
        emb_r = PCA(n_components=max_pc, svd_solver="randomized", random_state=0).fit_transform(Xs)
        c = trajectory_curvature(emb_r)[valid]
        layer_curv.append(np.mean(c))
        perm_means = []
        for _ in range(N_PERM):
            emb_shuf = emb_r[rng.permutation(n)]
            perm_means.append(np.mean(trajectory_curvature(emb_shuf)[valid]))
        layer_null.append(np.mean(perm_means))
    del emb_mm
    cm, nm = np.mean(layer_curv), np.mean(layer_null)
    curv_means.append(cm); null_means.append(nm)
    print(f"  {pid}: curv={cm:.2f}  null={nm:.2f}  diff={cm-nm:+.2f}", flush=True)

print(f"\nPodcast semantic (K=20, {MODEL}) pooled: curv={np.mean(curv_means):.2f}  null={np.mean(null_means):.2f}  diff={np.mean(curv_means)-np.mean(null_means):+.2f}")
