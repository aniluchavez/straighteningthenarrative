"""
compute_speak_vs_listen_baseline_curvature.py
====================================
Simple baseline-level comparison (no surprisal split): is mean curvature
systematically lower/higher during SPEAKING vs LISTENING, per patient?
Paired comparison (same 15 patients contribute both conditions),
patient-cluster / paired bootstrap for significance.

Convo dataset, raw/no-PCA embeddings, loo_layer_neu layer for semantic,
fit_neural_pca for neural -- same feature construction as the rest of this
session's event-triggered figures, but here just comparing unconditional
per-word curvature means between the two conditions (is_speaker True/False),
no lag/event structure at all.

Saves: results/speak_vs_listen_baseline_curvature.json
"""
import json, os, sys
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PARENT_DIR, "results")
CONVO_MODEL = "llama-3.1-8b"
REGION      = "hippocampus"

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import fit_neural_pca, trajectory_curvature, load_patient, PATIENTS, build_paths


def main():
    conv_path = os.path.join(RESULTS_DIR, f"geometry_{CONVO_MODEL}_{REGION}.json")
    cached = {r["pid"]: r["loo_layer_neu"] for r in json.load(open(conv_path))}

    rows = []
    for pid, preg in PATIENTS.items():
        if pid not in cached or REGION not in preg:
            continue
        layer = cached[pid]
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            continue
        n = data["n_words"]
        is_speaker = data["is_speaker"][:n].astype(bool)

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu_full = trajectory_curvature(Y_neu)

        paths = build_paths(pid, CONVO_MODEL)
        emb_mm = np.load(paths["cache"], mmap_mode="r")
        if layer >= emb_mm.shape[0]:
            del emb_mm
            continue
        emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
        del emb_mm
        c_sem_full = trajectory_curvature(emb_opt)

        # curvature array index i -> vertex at word i+1, so align is_speaker the same way
        m = len(c_sem_full)  # == n - 2
        spk = is_speaker[1:1 + m]  # vertex word for index i is i+1

        sem_speak, sem_listen = c_sem_full[spk], c_sem_full[~spk]
        neu_speak, neu_listen = c_neu_full[spk], c_neu_full[~spk]
        if len(sem_speak) < 20 or len(sem_listen) < 20:
            continue

        rows.append(dict(
            pid=pid,
            n_speak=int(spk.sum()), n_listen=int((~spk).sum()),
            sem_speak_mean=float(np.nanmean(sem_speak)), sem_listen_mean=float(np.nanmean(sem_listen)),
            neu_speak_mean=float(np.nanmean(neu_speak)), neu_listen_mean=float(np.nanmean(neu_listen)),
        ))
        print(f"  {pid}: n_speak={rows[-1]['n_speak']}  n_listen={rows[-1]['n_listen']}  "
              f"sem_speak={rows[-1]['sem_speak_mean']:.3f}  sem_listen={rows[-1]['sem_listen_mean']:.3f}  "
              f"neu_speak={rows[-1]['neu_speak_mean']:.3f}  neu_listen={rows[-1]['neu_listen_mean']:.3f}", flush=True)

    sem_diff = np.array([r["sem_speak_mean"] - r["sem_listen_mean"] for r in rows])
    neu_diff = np.array([r["neu_speak_mean"] - r["neu_listen_mean"] for r in rows])

    from scipy.stats import wilcoxon, ttest_rel
    t_sem, p_sem = ttest_rel(sem_diff, np.zeros_like(sem_diff))
    t_neu, p_neu = ttest_rel(neu_diff, np.zeros_like(neu_diff))
    w_sem = wilcoxon(sem_diff)
    w_neu = wilcoxon(neu_diff)

    print(f"\n=== Paired speaking-minus-listening curvature diff, n={len(rows)} patients ===")
    print(f"SEMANTIC: mean diff = {sem_diff.mean():+.4f} deg (sd={sem_diff.std():.4f})  "
          f"paired t={t_sem:.3f} p={p_sem:.4g}   wilcoxon p={w_sem.pvalue:.4g}")
    print(f"NEURAL:   mean diff = {neu_diff.mean():+.4f} deg (sd={neu_diff.std():.4f})  "
          f"paired t={t_neu:.3f} p={p_neu:.4g}   wilcoxon p={w_neu.pvalue:.4g}")
    n_sem_lower = int((sem_diff < 0).sum())
    n_neu_lower = int((neu_diff < 0).sum())
    print(f"patients with speaking < listening: semantic {n_sem_lower}/{len(rows)}   neural {n_neu_lower}/{len(rows)}")

    out = dict(rows=rows, sem_mean_diff=float(sem_diff.mean()), sem_p_ttest=float(p_sem), sem_p_wilcoxon=float(w_sem.pvalue),
               neu_mean_diff=float(neu_diff.mean()), neu_p_ttest=float(p_neu), neu_p_wilcoxon=float(w_neu.pvalue),
               n_patients=len(rows), n_sem_lower=n_sem_lower, n_neu_lower=n_neu_lower)
    out_path = os.path.join(RESULTS_DIR, "speak_vs_listen_baseline_curvature.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
