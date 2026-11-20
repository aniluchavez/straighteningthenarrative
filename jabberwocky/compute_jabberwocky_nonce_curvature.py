"""
compute_jabberwocky_nonce_curvature.py
====================================
Is NEURAL curvature elevated specifically at the nonsense (nonce) content
words in jabberwocky sentences, vs the real (mostly function) words?
Classifies each word as nonce (wordfreq zipf_frequency == 0, i.e. not a
real English word) vs real, using the correct duration window
(onset+80/offset+40, established today). Curvature index i is associated
with word i+2 (the endpoint of the second displacement, matching the
event-triggered convention used throughout this session).

Reports: mean neural curvature (and displacement) at nonce-word positions
vs real-word positions, patient-cluster bootstrap test of the difference.
"""
import os, sys
import numpy as np
import wordfreq
from scipy.stats import pearsonr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca
from jabberwocky_loader import JABBERWOCKY_PATIENTS, load_jabberwocky_patient
from compute_pca_sweep_jabberwocky import build_neu_vecs_dur

RESULTS_ROWS = []

for pid in JABBERWOCKY_PATIENTS:
    data0 = load_jabberwocky_patient(pid, region="hippocampus", model_tag="gpt2", context_words=200)
    if data0 is None:
        print(f"  {pid}: SKIP")
        continue
    neu_vecs, n = build_neu_vecs_dur(pid, data0)
    valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
    if valid.sum() < 30:
        print(f"  {pid}: SKIP (too few valid)")
        continue
    Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
    if Y_neu is None:
        continue
    c_neu = trajectory_curvature(Y_neu)[valid]
    d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

    words = data0["words"][:n]
    is_nonce = np.array([wordfreq.zipf_frequency(w, "en") == 0.0 and len(w) > 0 for w in words])
    # word at curvature-index i is word i+2 (0-indexed)
    cond_word_idx = np.arange(2, n)[valid]
    cond_nonce = is_nonce[cond_word_idx]

    n_nonce = cond_nonce.sum()
    n_real = (~cond_nonce).sum()
    c_nonce_mu = np.mean(c_neu[cond_nonce]) if n_nonce > 0 else np.nan
    c_real_mu = np.mean(c_neu[~cond_nonce]) if n_real > 0 else np.nan
    d_nonce_mu = np.mean(d_neu[cond_nonce]) if n_nonce > 0 else np.nan
    d_real_mu = np.mean(d_neu[~cond_nonce]) if n_real > 0 else np.nan

    print(f"  {pid}: n_nonce={n_nonce} n_real={n_real}  "
          f"curv nonce={c_nonce_mu:.2f} real={c_real_mu:.2f} diff={c_nonce_mu-c_real_mu:+.2f}   "
          f"disp nonce={d_nonce_mu:.3f} real={d_real_mu:.3f} diff={d_nonce_mu-d_real_mu:+.3f}",
          flush=True)

    for c, d, nonce in zip(c_neu, d_neu, cond_nonce):
        RESULTS_ROWS.append((pid, bool(nonce), float(c), float(d)))

import pandas as pd
df = pd.DataFrame(RESULTS_ROWS, columns=["patient", "nonce", "curv", "disp"])

def bootstrap_group_diff(df, col, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    patients = df.patient.unique()
    mu_nonce = df[df.nonce][col].mean()
    mu_real = df[~df.nonce][col].mean()
    obs_diff = mu_nonce - mu_real
    boot = []
    for _ in range(n_boot):
        samp = rng.choice(patients, size=len(patients), replace=True)
        bdf = pd.concat([df[df.patient == p] for p in samp])
        if bdf.nonce.sum() < 3 or (~bdf.nonce).sum() < 3:
            continue
        boot.append(bdf[bdf.nonce][col].mean() - bdf[~bdf.nonce][col].mean())
    boot = np.array(boot)
    p = 2 * min(np.mean(boot > 0), np.mean(boot < 0))
    ci = np.percentile(boot, [2.5, 97.5])
    return mu_nonce, mu_real, obs_diff, ci, p

for col in ["curv", "disp"]:
    mu_n, mu_r, diff, ci, p = bootstrap_group_diff(df, col)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"\n{col}: nonce={mu_n:.3f}  real={mu_r:.3f}  diff={diff:+.3f}  "
          f"CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  p={p:.4f} {stars}")

print(f"\nTotal rows: {len(df)}  (nonce={df.nonce.sum()}, real={(~df.nonce).sum()})")
