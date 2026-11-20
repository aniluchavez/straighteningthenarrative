"""
compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.py
=============================================================
Pseudopopulation version of compute_curv_disp_pos_gpt2xl_layer29_pc_sweep.py.

Neural side = the 13-patient, 475-channel hippocampal pseudopopulation built
exactly as in make_pairwise_distance_scatter_gpt2xl_pseudopop.py (difflib
alignment to YEY, words common to all patients, dur80/40, rate=True,
per-channel z-score) -- but kept as the FULL contiguous common-word
sequence (the cached pseudopop_neural_cache.npz holds only valid words, which
would put false steps across the gaps). Trajectory features are computed on
the full sequence, then masked with build_valid_mask, as in the per-patient
pipeline. Semantic side = gpt2-xl layer 29 at the same (YEY-indexed) words.

Master PCA per side: top K_POOL=200 components. For each K in KS:
  1. Random sweep: N_DRAWS draws of K random PCs per side (independent).
     Position = coordinate on the first drawn PC.
  2. PC dropout: PCs j..j+K-1, schemes "both" / "sem" (neural fixed top-K) /
     "neu" (semantic fixed top-K).
Inference (single pseudo-subject, no across-patient t): circular-shift null
on the chunk-mean series (neural chunk means rolled by a random shift >= 10%
of length), N_NULL shifts -> 2.5/97.5 percentiles per config.

Saves: results/curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.json
       results/pseudopop_full_sequence_dur8040_rate.npz (alignment cache)

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed --no-capture-output python3 -u compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.py
"""
import difflib
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from run_geometry_analysis_pca20 import build_valid_mask
from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

REF_PID = "YEY"
REGION = "hippocampus"
MODEL = "gpt2-xl"
LAYER = 29
ONSET_PAD_MS, OFFSET_PAD_MS = 80, 40
RATE = True
K_POOL = 200
KS = [8, 16, 32, 64, 128]
CHUNK_SIZE = 3
N_DRAWS = 200
N_NULL = 500
SEED = 0
METRICS = ("curv", "disp", "pos")
CACHE_PATH = os.path.join(RESULTS, "pseudopop_full_sequence_dur8040_rate.npz")
OUT_PATH = os.path.join(RESULTS, "curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop.json")


# ---------------- pseudopopulation (full sequence) ----------------
def get_neural(pid):
    data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
    if data0 is None:
        return None
    try:
        _, onset_ms, offset_ms = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        return None
    preg = _convo_regions(pid)
    n = min(data0["n_words"], len(onset_ms))
    neu_vecs = windowed_neu_vecs(pid, REGION, preg, onset_ms[:n], offset_ms[:n],
                                 ONSET_PAD_MS, OFFSET_PAD_MS, rate=RATE)
    if neu_vecs is None:
        return None
    return dict(n=n, neu_vecs=neu_vecs, words=data0["words"][:n], surp=data0["surp"][:n],
                sent_pos=data0["sent_pos"][:n], is_turn=data0["is_turn"][:n])


def align_to_ref(ref_words, words):
    sm = difflib.SequenceMatcher(None, ref_words, words, autojunk=False)
    mapping = {}
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def build_pseudopop():
    if os.path.exists(CACHE_PATH):
        c = np.load(CACHE_PATH)
        return c["neu_z"], c["common"], c["valid"]
    neural = {}
    for pid in PODCAST_PATIENTS:
        rec = get_neural(pid)
        if rec is not None:
            neural[pid] = rec
            print(f"  {pid}: n_words={rec['n']}  channels={rec['neu_vecs'].shape[1]}", flush=True)
    ref_words = neural[REF_PID]["words"].tolist()
    common = set(range(len(ref_words)))
    maps = {}
    for pid, rec in neural.items():
        maps[pid] = align_to_ref(ref_words, rec["words"].tolist())
        common &= set(maps[pid].keys())
    common = np.array(sorted(common))
    combined = np.concatenate([neural[p]["neu_vecs"][[maps[p][r] for r in common]] for p in neural], axis=1)
    ref = neural[REF_PID]
    valid = build_valid_mask(len(common), ref["surp"][common], ref["sent_pos"][common], ref["is_turn"][common])
    neu_z = StandardScaler().fit_transform(combined)
    np.savez_compressed(CACHE_PATH, neu_z=neu_z, common=common, valid=valid,
                        patients=np.array(list(neural)))
    return neu_z, common, valid


# ---------------- geometry ----------------
def pca_pool(X, k=K_POOL):
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    p = PCA(n_components=min(k, Xs.shape[1], Xs.shape[0] - 1), svd_solver="randomized", random_state=0)
    return p.fit_transform(Xs), p.explained_variance_ratio_


def curvature(Y):
    adj = np.diff(Y, axis=0)
    a, b = adj[:-1], adj[1:]
    nrm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    dot = np.einsum("ij,ij->i", a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.degrees(np.arccos(np.clip(dot / nrm, -1.0, 1.0)))
    out[nrm == 0] = 0.0
    return out


def features(Y, valid):
    c = curvature(Y)
    d = np.linalg.norm(np.diff(Y, axis=0), axis=1)
    p = Y[:, 0]
    out = {}
    for name, f in (("curv", c), ("disp", d), ("pos", p)):
        m = min(len(f), len(valid))
        out[name] = f[:m][valid[:m]]
    return out


def chunk_means(x, cs=CHUNK_SIZE):
    n = (len(x) // cs) * cs
    return x[:n].reshape(-1, cs).mean(axis=1)


def r_vec(a, B):
    """Pearson r of vector a against each row of B."""
    a = (a - a.mean()) / a.std()
    B = (B - B.mean(axis=1, keepdims=True)) / B.std(axis=1, keepdims=True)
    return (B @ a) / len(a)


class Nuller:
    """mode="shift": circular shifts >= 10% of length; mode="shuffle": random
    permutations of the neural chunk-mean order."""
    def __init__(self, n_chunks, rng, mode="shift"):
        if mode == "shift":
            lo = int(0.1 * n_chunks)
            shifts = rng.integers(lo, n_chunks - lo, size=N_NULL)
            self.idx = (np.arange(n_chunks)[None, :] - shifts[:, None]) % n_chunks
        else:
            self.idx = np.argsort(rng.random((N_NULL, n_chunks)), axis=1)

    def run(self, xn, xs):
        return r_vec(xs, xn[self.idx])


def corr3(Ys, Yn, valid, nuller=None):
    fs, fn = features(Ys, valid), features(Yn, valid)
    res = {}
    for m in METRICS:
        n = min(len(fs[m]), len(fn[m]))
        xs, xn = chunk_means(fs[m][:n]), chunk_means(fn[m][:n])
        r = float(np.corrcoef(xs, xn)[0, 1])
        res[m] = dict(r=r)
        if nuller is not None:
            nul = nuller.run(xn, xs)
            res[m].update(null_lo=float(np.percentile(nul, 2.5)), null_hi=float(np.percentile(nul, 97.5)),
                          p=float((np.sum(np.abs(nul) >= abs(r)) + 1) / (len(nul) + 1)))
    return res


def main():
    neu_z, common, valid = build_pseudopop()
    print(f"pseudopop: n_common={len(common)}  valid={int(valid.sum())}  channels={neu_z.shape[1]}", flush=True)

    data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
    emb_mm = np.load(data_m["emb_path"], mmap_mode="r")
    emb_l = np.array(emb_mm[LAYER, common], dtype=np.float64)
    del emb_mm

    Ys, evr_s = pca_pool(emb_l)
    Yn, evr_n = pca_pool(neu_z)
    print(f"pools: sem={Ys.shape[1]}  neu={Yn.shape[1]}", flush=True)

    rng = np.random.default_rng(SEED)
    n_chunks = int(valid.sum()) // CHUNK_SIZE
    nuller = Nuller(n_chunks, rng)

    out = dict(config=dict(KS=KS, K_POOL=K_POOL, CHUNK_SIZE=CHUNK_SIZE, MODEL=MODEL, LAYER=LAYER,
                           N_DRAWS=N_DRAWS, N_NULL=N_NULL, SEED=SEED, RATE=RATE,
                           n_common=int(len(common)), n_valid=int(valid.sum()), n_channels=int(neu_z.shape[1])),
               evr_sem=evr_s.tolist(), evr_neu=evr_n.tolist(), byK={})

    for K in KS:
        print(f"\n===== K={K} =====", flush=True)
        ref = corr3(Ys[:, :K], Yn[:, :K], valid, nuller)
        print(f"  top-{K}: " + "  ".join(f"{m}={ref[m]['r']:+.4f} [null {ref[m]['null_lo']:+.3f},{ref[m]['null_hi']:+.3f}]"
                                      for m in METRICS), flush=True)

        draws = []
        for _ in range(N_DRAWS):
            si = rng.choice(Ys.shape[1], size=K, replace=False)
            ni = rng.choice(Yn.shape[1], size=K, replace=False)
            r = corr3(Ys[:, si], Yn[:, ni], valid)
            draws.append(dict(sem=si.tolist(), neu=ni.tolist(), **{m: r[m]["r"] for m in METRICS}))
        summ, incl = {}, {"sem": {}, "neu": {}}
        for m in METRICS:
            v = np.array([dr[m] for dr in draws])
            summ[m] = dict(mean=float(v.mean()), sd=float(v.std()), p2_5=float(np.percentile(v, 2.5)),
                           p97_5=float(np.percentile(v, 97.5)), frac_pos=float((v > 0).mean()),
                           frac_win=float(np.mean([max(METRICS, key=lambda q: dr[q]) == m for dr in draws])))
            for side, pool in (("sem", Ys.shape[1]), ("neu", Yn.shape[1])):
                eff = []
                for j in range(pool):
                    w = [dr[m] for dr in draws if j in dr[side]]
                    wo = [dr[m] for dr in draws if j not in dr[side]]
                    eff.append(float(np.mean(w) - np.mean(wo)) if w and wo else float("nan"))
                incl[side][m] = eff
            print(f"  random: {m} mean={summ[m]['mean']:+.4f} 95%[{summ[m]['p2_5']:+.4f},{summ[m]['p97_5']:+.4f}] "
                  f"wins={summ[m]['frac_win']:.2f}", flush=True)

        dropout = {}
        for scheme in ("both", "sem", "neu"):
            rows = []
            for j in range(K_POOL - K + 1):
                js = j if scheme in ("both", "sem") else 0
                jn = j if scheme in ("both", "neu") else 0
                r = corr3(Ys[:, js:js + K], Yn[:, jn:jn + K], valid, nuller)
                rows.append(dict(offset=j, **r))
            dropout[scheme] = rows
            for j in (0, 1, 2, 4, 8, 16, 32, 64):
                if j < len(rows):
                    print(f"  [{scheme}] drop {j:3d}: " + "  ".join(f"{m}={rows[j][m]['r']:+.4f}" for m in METRICS),
                          flush=True)

        out["byK"][str(K)] = dict(top=ref, random=dict(draws=draws, summary=summ, inclusion_effect=incl),
                                  dropout=dropout)
        json.dump(out, open(OUT_PATH, "w"))

    print(f"\n  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
