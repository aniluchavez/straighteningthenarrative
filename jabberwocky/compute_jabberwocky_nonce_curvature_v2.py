"""
compute_jabberwocky_nonce_curvature_v2.py
====================================
Corrected version of compute_jabberwocky_nonce_curvature.py: instead of an
unreliable dictionary/frequency heuristic (which misclassifies many nonce
strings as "real" and vice versa -- verified by inspection), uses the
actual jabberwocky<->podcast sentence alignment file to build a genuine
per-word substituted/preserved label, restricted to high-confidence
sentence pairs (similarity >= 0.6, length-matched), where a word is
"nonce" iff it differs from the corresponding word in its aligned podcast
source sentence.

This master (word, is_nonce) sequence is then aligned to each patient's
own jabberwocky transcript via text-based sequence alignment (same
technique used for the podcast ISC analysis), so only positions covered
by a high-confidence label are used.
"""
import os, sys, re, difflib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis import trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca
from jabberwocky_loader import JABBERWOCKY_PATIENTS, load_jabberwocky_patient
from compute_pca_sweep_jabberwocky import build_neu_vecs_dur

ALIGNMENT_PATH = "/scratch/aniluchavez/semantic_syntactic_emb_project_for_Chavez/Jabberwocky/jabberwocky_podcast_alignment.xlsx"
SIM_THRESHOLD = 0.6


def clean_split(text):
    words = str(text).split()
    return [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words]


def build_master_labels():
    df = pd.read_excel(ALIGNMENT_PATH)
    df = df[(df["similarity"] >= SIM_THRESHOLD) & (df["length_matched_tags_corrected"] == True)]
    df = df.sort_values("jabberwocky_sentence_id")

    master_words, master_nonce = [], []
    for _, row in df.iterrows():
        jab_w = clean_split(row["jabberwocky_text"])
        pod_w = clean_split(row["podcast_text"])
        if len(jab_w) != len(pod_w):
            continue
        for jw, pw in zip(jab_w, pod_w):
            if not jw:
                continue
            master_words.append(jw)
            master_nonce.append(jw != pw)
    return master_words, np.array(master_nonce)


def align_to_ref(ref_words, words):
    sm = difflib.SequenceMatcher(None, ref_words, words, autojunk=False)
    mapping = {}
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def main():
    master_words, master_nonce = build_master_labels()
    print(f"Master label sequence: {len(master_words)} words "
          f"({master_nonce.sum()} nonce, {(~master_nonce).sum()} preserved), "
          f"from sentences with similarity>={SIM_THRESHOLD}", flush=True)

    rows = []
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

        patient_words = data0["words"][:n]
        mapping = align_to_ref(master_words, patient_words)  # master_idx -> patient_idx
        inv_mapping = {v: k for k, v in mapping.items()}      # patient_idx -> master_idx

        cond_word_idx = np.arange(2, n)[valid]  # word i+2 for each curvature index
        labels = []
        for wi in cond_word_idx:
            m_idx = inv_mapping.get(int(wi))
            labels.append(master_nonce[m_idx] if m_idx is not None else None)

        n_labeled = sum(l is not None for l in labels)
        n_nonce = sum(l is True for l in labels)
        n_real = sum(l is False for l in labels)
        print(f"  {pid}: aligned {len(mapping)}/{len(master_words)} master words; "
              f"{n_labeled} curvature points labeled ({n_nonce} nonce, {n_real} preserved)", flush=True)

        for c, d, lab in zip(c_neu, d_neu, labels):
            if lab is None:
                continue
            rows.append((pid, bool(lab), float(c), float(d)))

    df = pd.DataFrame(rows, columns=["patient", "nonce", "curv", "disp"])
    print(f"\nTotal labeled curvature points: {len(df)}  "
          f"(nonce={df.nonce.sum()}, preserved={(~df.nonce).sum()})")

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
        print(f"\n{col}: nonce={mu_n:.3f}  preserved={mu_r:.3f}  diff={diff:+.3f}  "
              f"CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  p={p:.4f} {stars}")

    # per-patient breakdown for sanity
    print("\nPer-patient breakdown:")
    for pid, g in df.groupby("patient"):
        gn = g[g.nonce]; gr = g[~g.nonce]
        if len(gn) < 3 or len(gr) < 3:
            print(f"  {pid}: insufficient data (nonce={len(gn)}, preserved={len(gr)})")
            continue
        print(f"  {pid}: curv nonce={gn.curv.mean():.2f} (n={len(gn)})  "
              f"preserved={gr.curv.mean():.2f} (n={len(gr)})  diff={gn.curv.mean()-gr.curv.mean():+.2f}")


if __name__ == "__main__":
    main()
