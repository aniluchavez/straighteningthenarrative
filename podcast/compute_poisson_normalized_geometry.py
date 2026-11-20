"""
compute_poisson_normalized_geometry.py
========================================
Re-runs the podcast curvature / displacement / position analyses with a
Poisson-motivated neural normalization (hippocampus, gpt2-xl layer 29).

Model: k_it ~ Poisson(g_t * T_t * f_i(s_t)).  sqrt(k) is variance-stabilized
(var ~ 1/4); dividing the sqrt vector by its L2 norm removes any scalar that
multiplies the whole population at a time point (gain g_t, window length T_t).

Neural variants (each gives curvature, displacement, position):
  z            z-scored raw counts, Euclidean                 (previous pipeline)
  sqrt         Anscombe sqrt(k + 3/8), Euclidean
  sphere_euc   sqrt -> L2-normalized, Euclidean chords/angles
  sphere_tan   sqrt -> L2-normalized; curvature = angle between incoming and
               outgoing tangent vectors at each vertex (log-map directions, so
               the sphere's own curvature is removed); displacement = geodesic
               (arc) length
  sphere_tan_cv  displacement noise-debiased by binomial thinning:
               k -> (k1, k2) independent Poisson halves; displacement^2 is the
               cross-half dot product of steps (N_SPLIT random splits averaged),
               so independent Poisson noise does not inflate it. Curvature and
               position are identical to sphere_tan (cross-half curvature was
               undefined at most vertices in synthetic tests at these counts).
  Pseudopop only: L2 normalization is applied PER PATIENT block (separate
  recording sessions -> separate gains), blocks scaled to total unit norm;
  tangent projection is per block. sphere_tan_joint = one sphere over all 475.

Semantic: raw (Euclidean, as before) and sphere_tan (L2-normalized embedding,
tangent curvature, geodesic displacement).

argv[1] mode   : perpatient | pseudopop
argv[2] window : dur8040  = [onset+80, offset+40] ms
                 fixed250 = [onset+80, onset+330] ms (true 250 ms; the earlier
                            'fixed250' cache passed offset=onset, which hit the
                            loader's offset<=onset fallback (offset=onset+100)
                            and so was actually [onset+80, onset+430])

perpatient: per-patient chunk(3) r(neural, semantic) for curv/disp/pos, all
            variant pairs, t-test across patients.
pseudopop : (a) chunk(3) r(neural, semantic) with circular-shift p
            (b) unique R^2 of surprisal / log freq / log dur for curvature and
                displacement (circular-shift p)
            (c) lagged regression curv(j+L-1) ~ surprisal(j) + triplet controls

Saves: results/poisson_norm_geometry_{mode}_{window}.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MODE = sys.argv[1] if len(sys.argv) > 1 else "perpatient"
WINDOW = sys.argv[2] if len(sys.argv) > 2 else "dur8040"
# argv[3] = number of PCs (top-K) for every variant; "full" = no truncation.
# z / semantic raw: PCA on standardized columns (as before); sqrt / sphere
# variants: PCA on centered, unstandardized columns (keeps the sqrt scaling).
# sphere_tan: tangent vectors computed in the full space, then projected onto
# the top-K PCs of the sphere points; displacement = projected chord.
# sphere_tan_cv: both thinned halves projected onto PCs fit on the full data.
KPC = sys.argv[3] if len(sys.argv) > 3 else "full"
KPC = None if KPC == "full" else int(KPC)
# argv[4] = "four": only the 4-step ablation (raw / sqrt / L2 / sqrt+L2), see neural_variants4
VARSET = sys.argv[4] if len(sys.argv) > 4 else "all"
# argv[5] = "rate": neural = firing rate (count / window length in s) instead of counts
RATE = len(sys.argv) > 5 and sys.argv[5] == "rate"
assert MODE in ("perpatient", "pseudopop") and WINDOW in ("dur8040", "fixed250", "fixed300")
sys.argv = sys.argv[:1]

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
from make_podcast_curvature_allwords_vs_midfreq_gpt2xl_layer29 import subtlex_log_freq
from compute_curv_disp_pos_gpt2xl_layer29_pc_sweep_pseudopop import align_to_ref

MODEL, LAYER, REGION, REF_PID = "gpt2-xl", 29, "hippocampus", "YEY"
CHUNK_SIZE = 3
N_SPLIT = 10
N_SHIFT = 1000
LAGS = list(range(-3, 8))
METRICS = ("curv", "disp", "pos")
SEED = 0
# YEX: words 0-2205 held out (wrong clock), so it cannot join a full-sequence
# pseudopop -> excluded (user decision 2026-11-11); still in per-patient runs.
PSEUDOPOP_EXCLUDE = {"YEX"}


# ---------------------------------------------------------------- loading
def load_counts(pid):
    try:
        _, on, off = load_words_onset_offset(pid)
    except (AssertionError, FileNotFoundError, OSError):
        return None
    data = load_podcast_patient(pid, region=REGION, model_tag=MODEL, context_words=200)
    if data is None:
        return None
    n = min(data["n_words"], len(on))
    preg = _convo_regions(pid)
    if WINDOW == "dur8040":
        K = windowed_neu_vecs(pid, REGION, preg, on[:n], off[:n], 80, 40, rate=RATE)
    elif WINDOW == "fixed250":
        K = windowed_neu_vecs(pid, REGION, preg, on[:n], on[:n] + 1e-3, 80, 330, rate=False)
    else:  # fixed300 = [onset+80, onset+380]
        K = windowed_neu_vecs(pid, REGION, preg, on[:n], on[:n] + 1e-3, 80, 380, rate=False)
    if K is None:
        return None
    dur = (off - on)[:n]
    return dict(n=n, K=np.asarray(K, float), data=data, dur=dur)


# ---------------------------------------------------------------- geometry
def rowdot(a, b):
    return np.einsum("ij,ij->i", a, b)


def angle_deg(a, b):
    """angle between rows; NaN where either vector has ~zero length (identical
    consecutive points -> direction undefined; previously 0 deg, which made
    the value depend on float rounding, e.g. counts vs rates)."""
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.degrees(np.arccos(np.clip(rowdot(a, b) / (na * nb), -1.0, 1.0)))
    tol = 1e-9 * max(np.nanmax(na), np.nanmax(nb), 1e-300)
    out[(na <= tol) | (nb <= tol)] = np.nan
    return out


def pc1(Y, standardize):
    Y = np.asarray(Y, float)
    Y = StandardScaler().fit_transform(Y) if standardize else Y - Y.mean(axis=0)
    return PCA(n_components=1, svd_solver="randomized", random_state=0).fit_transform(Y)[:, 0]


def pc_basis(Y, standardize):
    """(center, scale, V) with V = top-KPC principal axes (rows); V=None -> full space."""
    Y = np.asarray(Y, float)
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0) if standardize else np.ones(Y.shape[1])
    sd = np.where(sd > 0, sd, 1.0)
    if KPC is None or KPC >= min(Y.shape[1], Y.shape[0] - 1):
        return mu, sd, None
    p = PCA(n_components=KPC, svd_solver="randomized", random_state=0).fit((Y - mu) / sd)
    return mu, sd, p.components_


def proj_pts(Y, basis):
    mu, sd, V = basis
    Z = (Y - mu) / sd
    return Z if V is None else Z @ V.T


def proj_vec(D, basis):
    """project difference / tangent vectors (no centering)."""
    _, sd, V = basis
    Z = D / sd
    return Z if V is None else Z @ V.T


def k_eff(basis, ncol):
    return int(ncol if basis[2] is None else basis[2].shape[0])


def euclid(Y, standardize):
    b = pc_basis(Y, standardize)
    P = proj_pts(Y, b)
    adj = np.diff(P, axis=0)
    pos = P[:, 0] if b[2] is not None else pc1(Y, standardize)
    return dict(curv=angle_deg(adj[:-1], adj[1:]), disp=np.linalg.norm(adj, axis=1), pos=pos,
                k=k_eff(b, Y.shape[1]))


def sphere(S, blocks):
    X = np.empty_like(S)
    P = len(blocks)
    for b in blocks:
        nb = np.linalg.norm(S[:, b], axis=1, keepdims=True)
        X[:, b] = S[:, b] / np.where(nb > 0, nb, 1.0) / np.sqrt(P)
    return X


def tangent_vecs(X, blocks):
    """incoming / outgoing tangent directions at each interior vertex (per block
    log-map directions) and squared geodesic step length."""
    P = len(blocks)
    n = len(X)
    vin = np.zeros((n - 2, X.shape[1]))
    vout = np.zeros_like(vin)
    geo2 = np.zeros(n - 1)
    for b in blocks:
        U = X[:, b] * np.sqrt(P)
        u = U[1:-1]
        vin[:, b] = rowdot(u, U[:-2])[:, None] * u - U[:-2]
        vout[:, b] = U[2:] - rowdot(u, U[2:])[:, None] * u
        geo2 += np.arccos(np.clip(rowdot(U[:-1], U[1:]), -1, 1)) ** 2 / P
    return vin, vout, geo2


def sphere_tan(X, blocks):
    vin, vout, geo2 = tangent_vecs(X, blocks)
    b = pc_basis(X, False)
    if b[2] is None:
        return dict(curv=angle_deg(vin, vout), disp=np.sqrt(geo2), pos=pc1(X, False), k=X.shape[1], basis=b)
    return dict(curv=angle_deg(proj_vec(vin, b), proj_vec(vout, b)),
                disp=np.linalg.norm(proj_vec(np.diff(X, axis=0), b), axis=1),
                pos=proj_pts(X, b)[:, 0], k=k_eff(b, X.shape[1]), basis=b)


def sphere_tan_cv(K, blocks, rng, pos, curv_plain, basis):
    Ki = np.rint(K).astype(np.int64)
    num = np.zeros(len(K) - 2); ii = np.zeros_like(num); oo = np.zeros_like(num)
    d2 = np.zeros(len(K) - 1)
    for _ in range(N_SPLIT):
        K1 = rng.binomial(Ki, 0.5)
        K2 = Ki - K1
        X1 = sphere(np.sqrt(K1 + 0.375), blocks)
        X2 = sphere(np.sqrt(K2 + 0.375), blocks)
        i1, o1, _ = tangent_vecs(X1, blocks)
        i2, o2, _ = tangent_vecs(X2, blocks)
        num += 0.5 * (rowdot(i1, o2) + rowdot(i2, o1))
        ii += rowdot(i1, i2)
        oo += rowdot(o1, o2)
        d2 += rowdot(proj_vec(np.diff(X1, axis=0), basis), proj_vec(np.diff(X2, axis=0), basis))
    ok = (ii > 0) & (oo > 0)
    d2 /= N_SPLIT
    # cross-half curvature is undefined at ~45-70% of vertices at these counts
    # (synthetic test) and no more accurate, so curvature is taken from
    # sphere_tan (passed in); only displacement is debiased here.
    return dict(curv=curv_plain, disp=np.sqrt(np.clip(d2, 0, None)), pos=pos, k=k_eff(basis, K.shape[1]),
                frac_curv_undefined=float(1 - ok.mean()), frac_disp_nonpos=float((d2 <= 0).mean()))


def neural_variants(K, blocks, rng):
    S = np.sqrt(K + 0.375)
    X = sphere(S, blocks)
    out = dict(z=euclid(np.nan_to_num(StandardScaler().fit_transform(K)), True),
               sqrt=euclid(S, False),
               sphere_euc=euclid(X, False),
               sphere_tan=sphere_tan(X, blocks))
    st = out["sphere_tan"]
    out["sphere_tan_cv"] = sphere_tan_cv(K, blocks, rng, st["pos"], st["curv"], st.pop("basis"))
    if len(blocks) > 1:
        allb = [np.arange(K.shape[1])]
        out["sphere_tan_joint"] = sphere_tan(sphere(S, allb), allb)
        out["sphere_tan_joint"].pop("basis")
    return out


def neural_variants4(K, blocks, rng):
    """4-step ablation, plain sqrt (no +3/8):
      raw     raw counts, Euclidean (PCA on centered counts)
      sqrt    sqrt(k), Euclidean
      l2      k / ||k||_2 (sphere), tangent curvature
      sqrt_l2 sqrt(k) / ||sqrt(k)||_2, tangent curvature
    L2 is over ALL units (one block; pseudopop per-patient blocks are often
    all-zero, e.g. YFG 36% of words). Words with zero total spikes have no
    direction: curvature / displacement touching them are NaN in ALL four
    variants so they share the same words."""
    allb = [np.arange(K.shape[1])]
    zero = K.sum(axis=1) == 0
    unif = np.full(K.shape[1], 1.0 / np.sqrt(K.shape[1]))
    S = np.sqrt(K)

    def sph(Y):
        X = sphere(Y, allb)
        X[zero] = unif                              # placeholder, masked below
        return X

    out = dict(raw=euclid(K, False), sqrt=euclid(S, False))
    for name, Y in (("l2", K), ("sqrt_l2", S)):
        st = sphere_tan(sph(Y), allb)
        st.pop("basis")
        out[name] = st
    bad_c = zero[:-2] | zero[1:-1] | zero[2:]
    bad_d = zero[:-1] | zero[1:]
    for v in out.values():
        v["curv"] = np.where(bad_c, np.nan, v["curv"])
        v["disp"] = np.where(bad_d, np.nan, v["disp"])
    out["raw"]["frac_zero_rows"] = float(zero.mean())
    return out


def semantic_variants(emb):
    nb = np.linalg.norm(emb, axis=1, keepdims=True)
    allb = [np.arange(emb.shape[1])]
    st = sphere_tan(emb / nb, allb)
    st.pop("basis")
    return dict(raw=euclid(emb, True), sphere_tan=st)


# ---------------------------------------------------------------- stats
def chunk_r(x, y, cs=CHUNK_SIZE, min_chunks=3, return_chunks=False):
    fin = np.isfinite(x) & np.isfinite(y)
    xa, ya = [], []
    for i in range(0, len(x) - cs + 1, cs):
        sl = slice(i, i + cs)
        f = fin[sl]
        if f.sum() < cs // 2 or f.sum() == 0:
            continue
        xa.append(np.mean(x[sl][f])); ya.append(np.mean(y[sl][f]))
    if len(xa) < min_chunks:
        return (float("nan"), None, None) if return_chunks else float("nan")
    r = float(pearsonr(xa, ya)[0])
    return (r, np.array(xa), np.array(ya)) if return_chunks else r


def masked(f, valid):
    m = min(len(f), len(valid))
    return f[:m][valid[:m]]


def r2(X, y):
    X1 = np.c_[np.ones(len(y)), X]
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ b
    return 1 - res @ res / ((y - y.mean()) @ (y - y.mean())), b[1:]


def r2_many(X, Y):
    X1 = np.c_[np.ones(len(X)), X]
    B = np.linalg.lstsq(X1, Y, rcond=None)[0]
    R = Y - X1 @ B
    Yc = Y - Y.mean(axis=0)
    return 1 - (R ** 2).sum(axis=0) / (Yc ** 2).sum(axis=0)


def zs(x):
    return (x - x.mean()) / x.std()


# ---------------------------------------------------------------- per patient
def run_perpatient():
    rng = np.random.default_rng(SEED)
    per = {}
    for pid in PODCAST_PATIENTS:
        rec = load_counts(pid)
        if rec is None:
            continue
        n, data = rec["n"], rec["data"]
        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        if LAYER >= emb_mm.shape[0]:
            continue
        emb = np.array(emb_mm[LAYER, :n], dtype=np.float64)
        del emb_mm
        K = rec["K"][:n]
        # words with no neural data (NaN rows: missing onset, YEX held-out stories
        # 1-2) are dropped; angles / steps that would bridge a dropped word are
        # invalidated below.
        keep = np.where(np.all(np.isfinite(K), axis=1))[0]
        K, emb = K[keep], emb[keep]
        surp_k = np.asarray(data["surp"][:n])[keep]
        valid = build_valid_mask(len(keep), surp_k, np.asarray(data["sent_pos"][:n])[keep],
                                 np.asarray(data["is_turn"][:n])[keep])
        if len(keep) < n:
            contiguous = (keep[2:] - keep[:-2]) == 2              # triplet i..i+2 has no gap
            valid = valid & contiguous[:len(valid)]
            print(f"  {pid}: dropped {n - len(keep)} words with no neural data", flush=True)
        n = len(keep)
        if valid.sum() < 20:
            continue
        nv = neural_variants4(K, [np.arange(K.shape[1])], rng) if VARSET == "four" else \
            neural_variants(K, [np.arange(K.shape[1])], rng)
        sv = semantic_variants(emb)
        if VARSET == "four":
            sv = dict(raw=sv["raw"])
        res = dict(n_units=int(K.shape[1]), n_words_used=int(n), mean_count=float(K.mean()),
                   frac_zero_rows=float((K.sum(1) == 0).mean()),
                   cv_frac_curv_undefined=nv.get("sphere_tan_cv", {}).get("frac_curv_undefined"),
                   cv_frac_disp_nonpos=nv.get("sphere_tan_cv", {}).get("frac_disp_nonpos"),
                   k_eff={**{"neural_" + a: b["k"] for a, b in nv.items()}, **{"sem_" + a: b["k"] for a, b in sv.items()}},
                   r={})
        for nk, nd in nv.items():
            for sk, sd in sv.items():
                res["r"][f"{nk}|{sk}"] = {mt: chunk_r(masked(nd[mt], valid), masked(sd[mt], valid)) for mt in METRICS}
        per[pid] = res
        line = "  ".join(f"{k.split('|')[0]}:c{v['curv']:+.3f}/d{v['disp']:+.3f}"
                         for k, v in res["r"].items() if k.endswith("|raw"))
        print(f"  {pid} units={K.shape[1]} meancount={K.mean():.2f}  {line}", flush=True)

    summary = {}
    print(f"\n=== per-patient, K={KPC or 'full'}, {WINDOW} counts, gpt2-xl L{LAYER}, chunk={CHUNK_SIZE}, n={len(per)} ===")
    pairs = list(next(iter(per.values()))["r"].keys())
    for pair in pairs:
        summary[pair] = {}
        for mt in METRICS:
            v = np.array([per[p]["r"][pair][mt] for p in per])
            v = v[np.isfinite(v)]
            t, p = ttest_1samp(v, 0.0) if len(v) >= 2 else (np.nan, np.nan)
            summary[pair][mt] = dict(mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                                     t=float(t), p=float(p), n_pos=int((v > 0).sum()), n=int(len(v)))
        s = summary[pair]
        print(f"  {pair:28s} " + "  ".join(
            f"{mt}={s[mt]['mean']:+.4f}±{s[mt]['sem']:.4f} p={s[mt]['p']:.1e} ({s[mt]['n_pos']}/{s[mt]['n']})"
            for mt in METRICS))
    return dict(per_patient=per, summary=summary)


# ---------------------------------------------------------------- pseudopop
def run_pseudopop():
    rng = np.random.default_rng(SEED)
    recs = {}
    for pid in PODCAST_PATIENTS:
        if pid in PSEUDOPOP_EXCLUDE:
            print(f"  {pid}: excluded from pseudopop", flush=True)
            continue
        r = load_counts(pid)
        if r is not None:
            recs[pid] = r
            print(f"  {pid}: n_words={r['n']} units={r['K'].shape[1]} meancount={r['K'].mean():.2f}", flush=True)
    ref = recs[REF_PID]["data"]
    ref_words = list(ref["words"][:recs[REF_PID]["n"]])
    common = set(range(len(ref_words)))
    maps = {}
    for pid, r in recs.items():
        maps[pid] = align_to_ref(ref_words, list(r["data"]["words"][:r["n"]]))
        common &= set(maps[pid].keys())
    common = np.array(sorted(common))
    blocks, cols, start = [], [], 0
    for pid, r in recs.items():
        Kp = r["K"][[maps[pid][c] for c in common]]
        cols.append(Kp)
        blocks.append(np.arange(start, start + Kp.shape[1]))
        start += Kp.shape[1]
    K = np.concatenate(cols, axis=1)
    n = len(common)
    np.savez_compressed(os.path.join(RESULTS, f"pseudopop_{'rates' if RATE else 'counts'}_{WINDOW}_true.npz"), K=K, common=common,
                        patients=np.array(list(recs)), block_sizes=np.array([len(b) for b in blocks]))
    print(f"pseudopop: {n} words x {K.shape[1]} units, {len(recs)} patients", flush=True)

    valid = build_valid_mask(n, ref["surp"][common], ref["sent_pos"][common], ref["is_turn"][common])
    surp = np.asarray(ref["surp"], float)[common]
    lf = subtlex_log_freq(np.asarray(ref["words"])[common])
    dur = recs[REF_PID]["dur"][common]
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, np.nanmedian(dur[dur > 0]))
    ld = np.log(dur)
    emb = np.array(np.load(ref["emb_path"], mmap_mode="r")[LAYER, common], float)

    nv = neural_variants4(K, blocks, rng) if VARSET == "four" else neural_variants(K, blocks, rng)
    sv = semantic_variants(emb)
    if VARSET == "four":
        sv = dict(raw=sv["raw"])
    out = dict(n_words=int(n), n_units=int(K.shape[1]), patients=list(recs), window=WINDOW,
               frac_zero_rows=float((K.sum(1) == 0).mean()),
               cv_frac_curv_undefined=nv.get("sphere_tan_cv", {}).get("frac_curv_undefined"),
               cv_frac_disp_nonpos=nv.get("sphere_tan_cv", {}).get("frac_disp_nonpos"),
               k_eff={**{"neural_" + a: b["k"] for a, b in nv.items()}, **{"sem_" + a: b["k"] for a, b in sv.items()}})
    print("effective K:", out["k_eff"])
    if VARSET != "four":
      print(f"cv: curvature undefined at {out['cv_frac_curv_undefined'] * 100:.1f}% of vertices, "
          f"d^2<=0 at {out['cv_frac_disp_nonpos'] * 100:.1f}% of steps")

    # (a) coupling
    print("\n=== (a) chunk(3) r(neural, semantic), circular-shift p ===")
    coup = {}
    for nk, nd in nv.items():
        for sk, sd in sv.items():
            key = f"{nk}|{sk}"
            coup[key] = {}
            for mt in METRICS:
                r, xa, ya = chunk_r(masked(nd[mt], valid), masked(sd[mt], valid), return_chunks=True)
                L = len(xa)
                sh = rng.integers(int(0.1 * L), L - int(0.1 * L), size=N_SHIFT)
                xz, yz = zs(xa), zs(ya)
                null = np.array([np.mean(np.roll(xz, s) * yz) for s in sh])
                coup[key][mt] = dict(r=r, p=float((np.sum(np.abs(null) >= abs(r)) + 1) / (N_SHIFT + 1)))
            print(f"  {key:28s} " + "  ".join(f"{mt}: r={coup[key][mt]['r']:+.4f} p={coup[key][mt]['p']:.3f}" for mt in METRICS))
    out["coupling"] = coup

    # (b) variance partition
    vfull = np.zeros(n, bool)
    vfull[:len(valid)] = valid[:n]
    series = {}
    for tag, dct in [("neural_" + k, v) for k, v in nv.items()] + [("sem_" + k, v) for k, v in sv.items()]:
        series[(tag, "curv")] = np.r_[np.nan, dct["curv"], np.nan][:n]   # angle at word i
        series[(tag, "disp")] = np.r_[np.nan, dct["disp"]][:n]          # step into word i
    print("\n=== (b) unique R^2 (%) of surprisal / log freq / log dur, circular-shift p ===")
    part = {}
    PRED = ["surprisal", "log_freq", "log_dur"]
    for (tag, mt), y0 in series.items():
        m = vfull & np.isfinite(surp) & np.isfinite(lf) & np.isfinite(y0)
        X = StandardScaler().fit_transform(np.c_[surp[m], lf[m], ld[m]])
        y = zs(y0[m])
        L = len(y)
        Ysh = np.stack([np.roll(y, s) for s in rng.integers(int(0.1 * L), L - int(0.1 * L), size=N_SHIFT)], axis=1)
        R2f, betas = r2(X, y)
        nullf = r2_many(X, Ysh)
        res = dict(n=int(L), R2_full=float(R2f), p_full=float((np.sum(nullf >= R2f) + 1) / (N_SHIFT + 1)), preds={})
        for j, pn in enumerate(PRED):
            keep = [k for k in range(3) if k != j]
            u = R2f - r2(X[:, keep], y)[0]
            un = nullf - r2_many(X[:, keep], Ysh)
            res["preds"][pn] = dict(beta=float(betas[j]), r=float(np.corrcoef(X[:, j], y)[0, 1]), unique_R2=float(u),
                                    p_unique=float((np.sum(un >= u) + 1) / (N_SHIFT + 1)))
        part[f"{tag}|{mt}"] = res
        print(f"  {tag:24s} {mt}  n={L}  R2={R2f * 100:.2f}%  " + "  ".join(
            f"{pn}: b={res['preds'][pn]['beta']:+.3f} uR2={res['preds'][pn]['unique_R2'] * 100:.3f}% p={res['preds'][pn]['p_unique']:.3f}"
            for pn in PRED), flush=True)
    out["partition"] = part

    # (c) lagged regression (curvature)
    print("\n=== (c) lagged: beta for surprisal(j) on curvature at vertex j+L-1, triplet controls ===")
    lag = {}
    for (tag, mt), c in series.items():
        if mt != "curv":
            continue
        rows = []
        for L_ in LAGS:
            v = np.arange(n) + L_ - 1
            ok = vfull & (v - 1 >= 0) & (v + 1 < n)
            j = np.where(ok)[0]
            vv = v[j]
            colz = [surp[j]]
            for o in (-1, 0, 1):
                t = vv + o
                inside = t == j
                for arr, nm in ((surp, "surp"), (lf, "freq"), (ld, "dur")):
                    if nm == "surp" and inside.all():
                        continue
                    colz.append(arr[t])
            X = np.column_stack(colz)
            y = c[vv]
            f = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            X = np.column_stack([zs(col) for col in X[f].T])
            y = zs(y[f])
            X1 = np.c_[np.ones(len(y)), X]
            beta = np.linalg.lstsq(X1, y, rcond=None)[0][1]
            Lh = len(y)
            null = np.empty(N_SHIFT)
            for i, s in enumerate(rng.integers(int(0.1 * Lh), Lh - int(0.1 * Lh), size=N_SHIFT)):
                Xs = X1.copy(); Xs[:, 1] = np.roll(X[:, 0], s)
                null[i] = np.linalg.lstsq(Xs, y, rcond=None)[0][1]
            p = (np.sum(np.abs(null) >= abs(beta)) + 1) / (N_SHIFT + 1)
            rows.append(dict(lag=L_, beta=float(beta), null_lo=float(np.percentile(null, 2.5)),
                             null_hi=float(np.percentile(null, 97.5)), p=float(p), n=int(Lh)))
        lag[tag] = rows
        print(f"  {tag:24s} " + " ".join(f"{r['lag']:+d}:{r['beta']:+.3f}{'*' if r['p'] < 0.05 else ' '}" for r in rows),
              flush=True)
    out["lagged"] = dict(lags=LAGS, results=lag)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = run_perpatient() if MODE == "perpatient" else run_pseudopop()
    tag = "full" if KPC is None else f"pc{KPC}"
    out["K_requested"] = tag
    vs = ("_4step" if VARSET == "four" else "") + ("_rate" if RATE else "")
    path = os.path.join(RESULTS, f"poisson_norm_geometry{vs}_{MODE}_{WINDOW}_{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
