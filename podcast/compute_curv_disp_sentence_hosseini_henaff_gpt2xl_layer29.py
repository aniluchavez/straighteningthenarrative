"""
compute_curv_disp_sentence_hosseini_henaff_gpt2xl_layer29.py
============================================================
Curvature and displacement computed the way Hosseini & Fedorenko (2023) and
Henaff/Bai/Goris (2021) compute them, then compared neural vs semantic.

States x_t:
  semantic : GPT-2 XL layer 29 raw activations, 1600-D (Hosseini: raw layer states)
  neural   : hippocampus 2*sqrt(spike count), dur80/40, full D (Henaff, sigma->0)
Steps v_t = x_{t+1} - x_t.
  displacement d_t = ||v_t||                   (Henaff step length; NOT symmetrised)
  curvature    c_t = arccos(v_{t-1}.v_t / (||v_{t-1}|| ||v_t||))  (both papers)
Only steps / angles whose words all lie in the same sentence are used, and
each quantity is AVERAGED PER SENTENCE (Hosseini: per sentence; Henaff: per clip).
Also per sentence: magnitude mean ||x_t||, position mean ||x_t - session centroid||.
Sentences need >= 3 words (>= 1 angle).

Alignment: Pearson r across sentences between neural and semantic sentence means.
  perpatient : r per patient, t-test across patients
  pseudopop  : 12-patient / 436-unit pseudopop (results/pseudopop_counts_dur8040_true.npz),
               circular-shift null over the sentence series (N_SHIFT)

Sentence ids: YEX, YFB, YFU have no sentence segmentation in the loader (one
"sentence" of 7346 words); YEY's sent_id is transferred to their words via the
difflib word alignment (align_to_ref). Unmatched words get no sentence.

Saves: results/curv_disp_sentence_hosseini_henaff_gpt2xl_layer29.json

    conda run -n gpt2_embed --no-capture-output python3 -u compute_curv_disp_sentence_hosseini_henaff_gpt2xl_layer29.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, ttest_1samp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)
from compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop import align_to_ref

MODEL, LAYER, REF_PID = "gpt2-xl", 29, "YEY"
METRICS = ("curv", "disp", "mag", "pos")
N_SHIFT, SEED = 5000, 0
RESULTS = os.path.join(BASE_DIR, "results")
PP_CACHE = os.path.join(RESULTS, "pseudopop_counts_dur8040_true.npz")
OUT = os.path.join(RESULTS, "curv_disp_sentence_hosseini_henaff_gpt2xl_layer29.json")


def sentence_means(X, sid):
    """X: (T, D) states, sid: (T,) sentence id (-1 = none; NaN rows allowed).
    Returns dict metric -> array over sentences, and the sentence ids used."""
    T = len(X)
    ok = np.isfinite(X).all(1) & (sid >= 0)
    V = np.diff(X, axis=0)
    L = np.linalg.norm(V, axis=1)
    step_ok = ok[:-1] & ok[1:] & (sid[:-1] == sid[1:])
    num = np.einsum("ij,ij->i", V[:-1], V[1:]); den = L[:-1] * L[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
    ang_ok = step_ok[:-1] & step_ok[1:] & (den > 0)
    cen = np.nanmean(X[ok], axis=0)
    mag = np.linalg.norm(X, axis=1); pos = np.linalg.norm(X - cen, axis=1)
    sents = np.unique(sid[sid >= 0])
    out = {m: [] for m in METRICS}; used = []
    for s in sents:
        w = ok & (sid == s)
        a = ang_ok & (sid[1:-1] == s)          # angle at word t+1
        st = step_ok & (sid[:-1] == s)
        if w.sum() < 3 or a.sum() < 1:
            continue
        out["curv"].append(c[a].mean()); out["disp"].append(L[st].mean())
        out["mag"].append(mag[w].mean()); out["pos"].append(pos[w].mean())
        used.append(int(s))
    return {m: np.array(v) for m, v in out.items()}, np.array(used)


def ref_sentences():
    ref = load_podcast_patient(REF_PID, region="hippocampus", model_tag=MODEL, context_words=200)
    return list(ref["words"]), np.asarray(ref["sent_id"]).astype(int), ref


def sent_ids_for(data, ref_words, ref_sid):
    sid = np.asarray(data["sent_id"]).astype(int)
    if len(np.unique(sid)) > 1:
        return sid, False
    m = align_to_ref(ref_words, list(data["words"]))      # ref index -> this patient's index
    out = np.full(len(sid), -1)
    for r, i in m.items():
        out[i] = ref_sid[r]
    return out, True


def paired(qn, sn, qs, ss):
    common, i_n, i_s = np.intersect1d(sn, ss, return_indices=True)
    return {m: (qn[m][i_n], qs[m][i_s]) for m in METRICS}, len(common)


def run_perpatient(ref_words, ref_sid):
    per = {}
    for pid in PODCAST_PATIENTS:
        try:
            _, on, off = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
        if data is None:
            continue
        n0 = min(data["n_words"], len(on))
        K = windowed_neu_vecs(pid, "hippocampus", _convo_regions(pid), on[:n0], off[:n0], 80, 40, rate=False)
        if K is None:
            continue
        emb = np.load(data["emb_path"], mmap_mode="r")
        n = min(n0, emb.shape[1], len(K))
        sem = np.array(emb[LAYER, :n], dtype=np.float64); del emb
        neu = 2.0 * np.sqrt(np.asarray(K[:n], dtype=np.float64))
        sid, transferred = sent_ids_for(data, ref_words, ref_sid)
        sid = sid[:n]
        sem[~np.isfinite(neu).all(1)] = np.nan               # same words in both spaces
        qn, sn = sentence_means(neu, sid); qs, ss = sentence_means(sem, sid)
        pr, ns = paired(qn, sn, qs, ss)
        r = {m: float(pearsonr(*pr[m])[0]) for m in METRICS}
        per[pid] = dict(n_neurons=int(neu.shape[1]), n_sentences=int(ns), sent_id_transferred=transferred,
                        mean_curv={"sem": float(pr["curv"][1].mean()), "neu": float(pr["curv"][0].mean())}, r=r)
        print(f"  {pid}: neurons={neu.shape[1]:>2} sentences={ns}{' (ids from YEY)' if transferred else ''}  "
              + "  ".join(f"{m}={r[m]:+.4f}" for m in METRICS), flush=True)
    summ = {}
    print(f"\n=== per patient, sentence means, n={len(per)} ===")
    for m in METRICS:
        v = np.array([per[p]["r"][m] for p in per])
        t, p = ttest_1samp(v, 0)
        summ[m] = dict(mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(len(v))), t=float(t), p=float(p),
                       n_pos=int((v > 0).sum()), n=len(v))
        print(f"  {m:<5} mean={v.mean():+.4f}  sem={summ[m]['sem']:.4f}  t={t:+.2f}  p={p:.2e}  n_pos={(v > 0).sum()}/{len(v)}")
    for s in ("sem", "neu"):
        print(f"  mean curvature {s}: {np.mean([per[p]['mean_curv'][s] for p in per]):.1f} deg")
    return dict(per_patient=per, summary=summ)


def run_pseudopop(ref, ref_sid):
    c = np.load(PP_CACHE)
    K, common = c["K"].astype(np.float64), c["common"]
    neu = 2.0 * np.sqrt(K)
    sem = np.array(np.load(ref["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
    sid = ref_sid[common].copy()
    gap = np.r_[False, np.diff(common) > 1]                    # a word after a gap starts a new segment
    sid = sid * 10 + np.cumsum(gap)                            # split sentences at gaps
    qn, sn = sentence_means(neu, sid); qs, ss = sentence_means(sem, sid)
    pr, ns = paired(qn, sn, qs, ss)
    rng = np.random.default_rng(SEED)
    res = {}
    print(f"\n=== pseudopop ({len(c['patients'])} patients, {K.shape[1]} units), sentence means, n_sentences={ns} ===")
    for m in METRICS:
        a, b = pr[m]
        az, bz = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
        r = float(np.mean(az * bz)); L = len(a)
        sh = rng.integers(int(0.1 * L), L - int(0.1 * L), size=N_SHIFT)
        null = np.array([np.mean(np.roll(az, s) * bz) for s in sh])
        res[m] = dict(r=r, p=float((np.sum(np.abs(null) >= abs(r)) + 1) / (N_SHIFT + 1)),
                      null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)))
        print(f"  {m:<5} r={r:+.4f}  p={res[m]['p']:.4f}  null 95%=[{res[m]['null_lo']:+.4f}, {res[m]['null_hi']:+.4f}]")
    mc = {"sem": float(pr["curv"][1].mean()), "neu": float(pr["curv"][0].mean())}
    print(f"  mean curvature sem={mc['sem']:.1f}  neu={mc['neu']:.1f} deg")
    return dict(patients=list(c["patients"]), n_units=int(K.shape[1]), n_sentences=int(ns), mean_curv=mc, r=res)


def main():
    ref_words, ref_sid, ref = ref_sentences()
    out = {"perpatient": run_perpatient(ref_words, ref_sid), "pseudopop": run_pseudopop(ref, ref_sid)}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
