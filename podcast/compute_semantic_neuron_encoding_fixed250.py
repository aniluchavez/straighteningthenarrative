"""
compute_semantic_neuron_encoding_fixed250.py
===============================================
Per-neuron semantic encoding (Poisson ridge, nested deviance) on the podcast
pseudopopulation, to split neurons into semantic-encoding vs. not.

Neurons: hippocampal units of the 12 podcast patients EXCLUDING YFX (its
transcript is +574.9 s off the spike clock in podcast_loader). Words: those
common to all 12 (difflib alignment to YEY, as in
make_pairwise_distance_scatter_gpt2xl_pseudopop.py).

Response: spike count in a FIXED window [onset+80, onset+330] ms.
Controls C: log word duration, Zipf frequency (wordfreq).
Semantic S: top N_SEM PCs of gpt2-xl layer 29 (YEY embeddings at common words).
Per neuron, blocked 5-fold CV (contiguous folds):
  - alpha per model picked from ALPHAS by CV deviance
  - stat = dD2 = (dev_C - dev_CS) / dev_null   (held-out, summed over folds)
  - null: circularly shift S relative to (spikes, C) by >=10% of n,
          N_NULL shifts, same alpha; p = (#null >= obs + 1)/(N_NULL + 1)
  - BH-FDR across neurons.

Also caches, for the geometry-by-subset step, the dur80/40 rate vectors
(same construction as the pseudopop geometry analyses) on the same words.

Saves: results/semantic_neuron_encoding_fixed250.json
       results/semantic_neuron_pseudopop_noYFX.npz

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed --no-capture-output python3 -u compute_semantic_neuron_encoding_fixed250.py
"""
import difflib
import json
import os
import sys
import warnings

import numpy as np
import wordfreq
from joblib import Parallel, delayed
from scipy.stats import false_discovery_control
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, BASE_DIR)

from podcast_loader import (
    PODCAST_PATIENTS, load_podcast_patient, _convo_regions,
    load_words_onset_offset, windowed_neu_vecs,
)

REF_PID = "YEY"
EXCLUDE = {"YFX"}
REGION = "hippocampus"
MODEL = "gpt2-xl"
LAYER = 29
WIN_START, WIN_LEN = 80, 250
N_SEM = 50
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0]
N_FOLDS = 5
N_NULL = 200
SEED = 0
N_JOBS = 32
CACHE_PATH = os.path.join(RESULTS, "semantic_neuron_pseudopop_noYFX.npz")
OUT_PATH = os.path.join(RESULTS, "semantic_neuron_encoding_fixed250.json")


def align_to_ref(ref_words, words):
    sm = difflib.SequenceMatcher(None, ref_words, words, autojunk=False)
    mapping = {}
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def build():
    if os.path.exists(CACHE_PATH):
        return dict(np.load(CACHE_PATH, allow_pickle=True))
    recs = {}
    for pid in PODCAST_PATIENTS:
        if pid in EXCLUDE:
            continue
        data0 = load_podcast_patient(pid, region=REGION, model_tag="gpt2", context_words=200)
        if data0 is None:
            continue
        try:
            _, onset_ms, offset_ms = load_words_onset_offset(pid)
        except (AssertionError, FileNotFoundError, OSError):
            continue
        preg = _convo_regions(pid)
        n = min(data0["n_words"], len(onset_ms))
        on, off = np.asarray(onset_ms[:n], float), np.asarray(offset_ms[:n], float)
        # fixed window: t0 = onset+80, t1 = (onset+250)+80 = onset+330
        cnt = windowed_neu_vecs(pid, REGION, preg, on, on + WIN_LEN, WIN_START, WIN_START, rate=False)
        rate = windowed_neu_vecs(pid, REGION, preg, on, off, 80, 40, rate=True)
        if cnt is None or rate is None:
            continue
        recs[pid] = dict(cnt=cnt, rate=rate, words=data0["words"][:n], on=on, off=off,
                         surp=data0["surp"][:n], sent_pos=data0["sent_pos"][:n], is_turn=data0["is_turn"][:n])
        print(f"  {pid}: n_words={n}  units={cnt.shape[1]}", flush=True)

    ref_words = recs[REF_PID]["words"].tolist()
    common = set(range(len(ref_words)))
    maps = {}
    for pid, r in recs.items():
        maps[pid] = align_to_ref(ref_words, r["words"].tolist())
        common &= set(maps[pid].keys())
    common = np.array(sorted(common))
    pids = list(recs)
    cnt = np.concatenate([recs[p]["cnt"][[maps[p][i] for i in common]] for p in pids], axis=1)
    rate = np.concatenate([recs[p]["rate"][[maps[p][i] for i in common]] for p in pids], axis=1)
    unit_pid = np.concatenate([[p] * recs[p]["cnt"].shape[1] for p in pids])
    ref = recs[REF_PID]
    dur = (ref["off"] - ref["on"])[common]
    out = dict(cnt=cnt, rate=rate, unit_pid=unit_pid, common=common, words=ref["words"][common],
               dur=dur, surp=ref["surp"][common], sent_pos=ref["sent_pos"][common],
               is_turn=ref["is_turn"][common])
    np.savez_compressed(CACHE_PATH, **out)
    return out


def blocks(n, k=N_FOLDS):
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(np.r_[0:edges[i], edges[i + 1]:n], np.arange(edges[i], edges[i + 1])) for i in range(k)]


def poisson_dev(y, mu):
    mu = np.clip(mu, 1e-12, None)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2.0 * np.sum(t - (y - mu))


def cv_dev(X, y, alpha, folds):
    dev = 0.0
    for tr, te in folds:
        m = PoissonRegressor(alpha=alpha, max_iter=300).fit(X[tr], y[tr])
        dev += poisson_dev(y[te], m.predict(X[te]))
    return dev


def fit_neuron(y, C, S, folds, shifts):
    warnings.simplefilter("ignore", ConvergenceWarning)
    dev_null = sum(poisson_dev(y[te], np.full(len(te), y[tr].mean())) for tr, te in folds)
    if y.sum() < 50:
        return None
    dC = {a: cv_dev(C, y, a, folds) for a in ALPHAS}
    aC = min(dC, key=dC.get)
    CS = np.hstack([C, S])
    dCS = {a: cv_dev(CS, y, a, folds) for a in ALPHAS}
    aCS = min(dCS, key=dCS.get)
    obs = (dC[aC] - dCS[aCS]) / dev_null
    null = np.empty(len(shifts))
    for i, s in enumerate(shifts):
        null[i] = (dC[aC] - cv_dev(np.hstack([C, np.roll(S, s, axis=0)]), y, aCS, folds)) / dev_null
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    return dict(dD2=float(obs), p=float(p), alpha_C=aC, alpha_CS=aCS, null_mean=float(null.mean()),
                null_95=float(np.percentile(null, 95)), D2_C=float(1 - dC[aC] / dev_null),
                D2_CS=float(1 - dCS[aCS] / dev_null), n_spikes=int(y.sum()))


def main():
    d = build()
    cnt, common = d["cnt"], d["common"]
    n, n_units = cnt.shape
    print(f"common words={n}  units={n_units}  patients={sorted(set(d['unit_pid'].tolist()))}", flush=True)

    dur = d["dur"].astype(float)
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, np.nanmedian(dur[dur > 0]))
    zipf = np.array([wordfreq.zipf_frequency(str(w).lower().strip(".,!?;:\"'"), "en") for w in d["words"]])
    C = StandardScaler().fit_transform(np.column_stack([np.log(dur), zipf]))

    data_m = load_podcast_patient(REF_PID, region=REGION, model_tag=MODEL, context_words=200)
    emb = np.array(np.load(data_m["emb_path"], mmap_mode="r")[LAYER, common], dtype=np.float64)
    S = PCA(n_components=N_SEM, svd_solver="randomized", random_state=0).fit_transform(
        StandardScaler().fit_transform(emb))
    S = StandardScaler().fit_transform(S)

    folds = blocks(n)
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(int(0.1 * n), n - int(0.1 * n), size=N_NULL)

    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(fit_neuron)(cnt[:, u].astype(float), C, S, folds, shifts) for u in range(n_units))

    ok = [u for u in range(n_units) if res[u] is not None]
    q = np.full(n_units, np.nan)
    q[ok] = false_discovery_control([res[u]["p"] for u in ok])
    units = []
    for u in range(n_units):
        r = res[u] or {}
        units.append(dict(unit=u, pid=str(d["unit_pid"][u]), q=float(q[u]), **r))
    sig = [x for x in units if np.isfinite(x["q"]) and x["q"] < 0.05]
    p05 = [x for x in units if "p" in x and x["p"] < 0.05]
    print(f"\nfitted {len(ok)}/{n_units} units (>=50 spikes)")
    print(f"  raw p<0.05: {len(p05)}/{len(ok)} ({100 * len(p05) / len(ok):.1f}%)")
    print(f"  FDR q<0.05: {len(sig)}/{len(ok)} ({100 * len(sig) / len(ok):.1f}%)")
    for pid in sorted(set(d["unit_pid"].tolist())):
        uu = [x for x in units if x["pid"] == pid and "p" in x]
        print(f"    {pid}: fitted={len(uu):3d}  p<0.05={sum(x['p'] < 0.05 for x in uu):3d}  "
              f"q<0.05={sum(x['q'] < 0.05 for x in uu):3d}")
    json.dump(dict(config=dict(LAYER=LAYER, MODEL=MODEL, WIN=[WIN_START, WIN_START + WIN_LEN], N_SEM=N_SEM,
                               ALPHAS=ALPHAS, N_FOLDS=N_FOLDS, N_NULL=N_NULL, EXCLUDE=sorted(EXCLUDE),
                               n_words=int(n), n_units=int(n_units), controls=["log_dur", "zipf"]),
                   units=units), open(OUT_PATH, "w"), indent=1)
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
