"""
run_geometry_analysis.py
========================
Full geometric analysis for NeurIPS paper.
Runs for every combination of LLM model × brain region.

Analyses (two-pass structure per model × region):
  Pass 1 — per-layer profiles:
    A. Mean trajectory curvature per layer        (do middle layers flatten?)
    B. r(c_layer, surprisal[t+1]) per layer       (predictive alignment)
    C. r(c_layer, c_neural) per layer             (neural-semantic alignment)

  LOO layer selection:
    LOO-surp:   maximize |r(c_layer, surprisal)|  across other patients
    LOO-neural: maximize |r(c_layer, c_neural)|   across other patients

  Pass 2 — at each patient's LOO-neural layer:
    D. Regression: c_sem[t] -> surprisal[t+1] + partial-R² of neural curvature
    E. Event-triggered curvature around high/low surprisal words

Outputs: geometry_paper/results/geometry_{model}_{region}.json  per combination.

Run:
    conda run -n gpt2_embed python3 run_geometry_analysis.py
"""

import math, os, json, warnings
import h5py, scipy.sparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, sem as scipy_sem
from scipy.stats import t as t_dist
from scipy.spatial import procrustes as scipy_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT      = "/scratch/aniluchavez/ConvoDATAS"
CACHE_DIR = os.path.join(ROOT, "EmbedCache")
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS   = os.path.join(BASE_DIR, "results")
FIGURES   = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Models to compare: tag -> number of layers in cache
MODELS = {
    "gpt2":         13,
    "gpt2-medium":  25,
    "opt-350m":     24,
    "bert-base":    13,
    "roberta-base": 13,
    "llama-2-7b":   33,
    "llama-3.1-8b": 33,
    "gemma-2-9b":   43,
    "mistral-7b":   33,   # Llama-2 architecture variant (sliding window + GQA)
}

# Brain regions to compare
REGIONS = ["hippocampus", "ACC"]

ACCEPTED_QUAL      = (4, 5)
WIN_MS             = 500
SPEAK_OFFSET_MS    = 200
LISTEN_OFFSET_MS   = 0
NEURAL_VAR_THRESH  = 0.90
MIN_PAIRS          = 20
SENT_BOUNDARY_EXCL = 2
EVENT_WINDOW       = 5
EVENT_PCTILE_HI    = 75
EVENT_PCTILE_LO    = 25

PATIENTS = {
    "PTYEU_task147": {"hippocampus": [(1,16),(25,40)],  "ACC": [(17,24),(41,48)]},
    "PTYEV_task37":  {"hippocampus": [(1,16),(25,40)],  "ACC": [(17,24),(41,48)]},
    "PTYEY_task86":  {"hippocampus": [(1,16)]},
    "PTYEZ_task60":  {"hippocampus": [(1,16)],          "ACC": [(17,24)]},
    "PTYFA_task25":  {"hippocampus": [(1,16),(25,40)],  "ACC": [(17,24)]},
    "PTYFC_task28":  {"hippocampus": [(1,8),(33,48)],   "ACC": [(17,32),(49,64)]},
    "PTYFF_task17":  {"hippocampus": [(9,16),(25,40)],  "ACC": [(17,24),(41,48)]},
    "PTYFG_task18":  {"hippocampus": [(9,16)],          "ACC": [(25,56)]},
    "PTYFI_task81":  {"hippocampus": [(1,8),(25,40)],   "ACC": [(9,16)]},
    "PTYFK_task40":  {"hippocampus": [(1,16),(25,40)],  "ACC": [(49,56)],
                      "thalamus":    [(17,24)],          "OFC": [(41,48)]},
}


# ── Path helpers ───────────────────────────────────────────────────────────────
def build_paths(pid, model_tag):
    short = pid.split("_")[0].upper()[2:]
    task  = pid.split("_", 1)[1]
    cache = (f"{pid}_{model_tag}_word_emb_layers.npy" if model_tag
             else f"{pid}_word_emb_layers.npy")
    return {
        "spikes":    os.path.join(ROOT, "SpikesMAT", short,
                                  f"pt{short}_{task}_new_spikes.mat"),
        "xlsx":      os.path.join(ROOT, "BERTEmbeds",
                                  f"{pid}_words_english_only",
                                  f"{pid}_filtered_used_rows_withNP_withClusterIDNew.xlsx"),
        "surprisal": os.path.join(ROOT, "Surprisal", f"{pid}_surprisal.csv"),
        "cache":     os.path.join(CACHE_DIR, cache),
    }


# ── Geometry ───────────────────────────────────────────────────────────────────
def trajectory_curvature(X):
    X   = np.asarray(X, dtype=np.float64)
    adj = X[1:] - X[:-1]
    out = np.empty(len(adj) - 1)
    for i in range(len(out)):
        n = np.linalg.norm(adj[i]) * np.linalg.norm(adj[i+1])
        d = np.dot(adj[i], adj[i+1])
        out[i] = np.degrees(math.acos(np.clip(d/n, -1., 1.))) if n != 0 else 0.
    return out


def trajectory_displacement(X):
    X = np.asarray(X, dtype=np.float64)
    return np.linalg.norm(X[1:] - X[:-1], axis=1)


def trajectory_torsion(X):
    """
    High-D torsion proxy: angle (°) each successive displacement vector makes
    out of the osculating plane defined by the two preceding vectors.
    Vectorised — O(n × D). Returns array of length len(X) - 3.
    """
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 4:
        return np.zeros(0)
    d  = X[1:] - X[:-1]           # (n-1, D)
    d0, d1, d2 = d[:-2], d[1:-1], d[2:]  # each (n-3, D)
    # unit tangent along d0
    n0 = np.linalg.norm(d0, axis=1, keepdims=True).clip(1e-12)
    e1 = d0 / n0
    # Gram-Schmidt: component of d1 perpendicular to e1
    rej1 = d1 - (d1 * e1).sum(-1, keepdims=True) * e1
    n1   = np.linalg.norm(rej1, axis=1, keepdims=True).clip(1e-12)
    e2   = rej1 / n1
    # out-of-plane component of d2
    in_plane = ((d2 * e1).sum(-1, keepdims=True) * e1
              + (d2 * e2).sum(-1, keepdims=True) * e2)
    oop   = d2 - in_plane
    n_oop = np.linalg.norm(oop, axis=1)
    n_d2  = np.linalg.norm(d2,  axis=1).clip(1e-12)
    return np.degrees(np.arcsin((n_oop / n_d2).clip(0., 1.)))


def trajectory_straightness(X):
    """End-to-end distance / total arc length.  1 = straight, 0 = closed loop."""
    X = np.asarray(X, dtype=np.float64)
    arc = np.linalg.norm(X[1:] - X[:-1], axis=1).sum()
    return float(np.linalg.norm(X[-1] - X[0]) / arc) if arc > 1e-12 else np.nan


def geodesic_curvature(X, k_neighbors=20, tangent_dim=10):
    """
    Decompose trajectory curvature into geodesic (on-manifold) and normal
    (off-manifold) components.  Returns two arrays of length len(X)-2,
    matching trajectory_curvature().

    Geodesic curvature = component of the curvature vector *within* the local
    tangent space of the manifold.  A geodesic path has κ_geo = 0 — any
    deviation means the trajectory is being forced off its natural path on the
    manifold, i.e. a geometric analogue of prediction error.

    Normal curvature = component perpendicular to the tangent space — caused by
    the manifold itself bending in the ambient space, not by the computation.
    """
    X = np.asarray(X, dtype=np.float64)
    n, D = X.shape
    n_int = n - 2
    geo = np.full(n_int, np.nan)
    nrm = np.full(n_int, np.nan)
    if n < 4:
        return geo, nrm

    steps = X[1:] - X[:-1]
    T = steps / np.linalg.norm(steps, axis=1, keepdims=True).clip(1e-12)
    kappa = T[1:] - T[:-1]  # (n-2, D) curvature vectors

    # Batch all pairwise squared distances once
    Xsq = np.sum(X ** 2, axis=1)
    D2 = np.maximum(Xsq[:, None] + Xsq[None, :] - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(D2, np.inf)

    k  = min(k_neighbors, n - 1)
    td = min(tangent_dim, D, k - 1)

    for i in range(n_int):
        kap = kappa[i]
        kap_mag = np.linalg.norm(kap)
        if kap_mag < 1e-15:
            geo[i] = nrm[i] = 0.0
            continue
        t = i + 1
        nbrs = X[np.argsort(D2[t])[:k]] - X[t]
        if np.allclose(nbrs, 0):
            geo[i] = kap_mag
            nrm[i] = 0.0
            continue
        _, _, vh = np.linalg.svd(nbrs, full_matrices=False)
        basis = vh[:td].T  # (D, td)
        kap_t = basis @ (basis.T @ kap)
        geo[i] = float(np.linalg.norm(kap_t))
        nrm[i] = float(np.linalg.norm(kap - kap_t))

    return geo, nrm


def scale_curvature_slope(X, scales=(1, 2, 4, 8)):
    """
    Sectional curvature proxy: per-word slope of log(curvature) vs log(scale).

    At temporal scale s the curvature at word t is the turning angle between
    X[t]-X[t-s] and X[t+s]-X[t].  The slope across scales reveals the manifold's
    sectional curvature character:
        slope < 0 : positive sectional curvature (sphere-like, geodesics converge)
        slope ≈ 0 : flat manifold
        slope > 0 : negative sectional curvature (hyperbolic, geodesics diverge)

    Returns array of length len(X)-2, matching trajectory_curvature().
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    n_int = n - 2
    log_s = np.log(np.array(scales, dtype=float))
    mean_ls = log_s.mean()
    ss = float(np.sum((log_s - mean_ls) ** 2))

    curv_mat = np.full((n_int, len(scales)), np.nan)
    for si, s in enumerate(scales):
        if n < 2 * s + 1:
            continue
        t_vals = np.arange(s, n - s)
        i_vals = t_vals - 1
        d0 = X[t_vals] - X[t_vals - s]
        d1 = X[t_vals + s] - X[t_vals]
        dots  = np.einsum("ij,ij->i", d0, d1)
        denom = np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1)
        safe  = denom > 1e-12
        cos_a = np.where(safe, dots / np.where(safe, denom, 1.0), 0.0)
        curv_mat[i_vals, si] = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    slopes = np.full(n_int, np.nan)
    ok = ~np.any(np.isnan(curv_mat), axis=1) & np.all(curv_mat > 0, axis=1)
    if ok.any() and ss > 0:
        lc  = np.log(curv_mat[ok])
        num = ((log_s - mean_ls) * lc).sum(axis=1)
        slopes[ok] = num / ss
    return slopes


# ── OLS ────────────────────────────────────────────────────────────────────────
def ols_numpy(y, X_cols, col_names):
    n   = len(y)
    Xm  = np.column_stack([np.ones(n)] + X_cols)
    k   = Xm.shape[1]
    b, _, _, _ = np.linalg.lstsq(Xm, y, rcond=None)
    yhat  = Xm @ b
    resid = y - yhat
    s2    = np.dot(resid, resid) / max(n - k, 1)
    try:
        cov_b = s2 * np.linalg.inv(Xm.T @ Xm)
        se_b  = np.sqrt(np.diag(cov_b))
    except np.linalg.LinAlgError:
        se_b  = np.full(k, np.nan)
    tvals  = b / np.where(se_b > 0, se_b, np.nan)
    pvals  = 2 * t_dist.sf(np.abs(tvals), df=max(n - k, 1))
    ss_tot = np.sum((y - y.mean())**2)
    ss_res = np.dot(resid, resid)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2_adj = 1 - (1-r2)*(n-1)/max(n-k, 1) if ss_tot > 0 else np.nan
    return {
        "r2": float(r2), "r2_adj": float(r2_adj),
        "betas": dict(zip(col_names, b[1:].tolist())),
        "pvals": dict(zip(col_names, pvals[1:].tolist())),
        "n": int(n),
    }


# ── Metadata ───────────────────────────────────────────────────────────────────
def get_sent_positions(df):
    sent_pos = np.zeros(len(df), dtype=int)
    pos = 0
    for i, row in df.iterrows():
        sent_pos[i] = pos
        ends = any(str(row.get(c,"")).strip().endswith((".", "?", "!"))
                   for c in ["Speaker1","Speaker2","Speaker3"])
        pos = 0 if ends else pos + 1
    return sent_pos


def get_log_freq(words):
    counts = Counter(w.lower() for w in words)
    total  = max(sum(counts.values()), 1)
    return np.array([math.log(counts[w.lower()] / total + 1e-9) for w in words])


# ── Patient loader ─────────────────────────────────────────────────────────────
def load_patient(pid, patient_regions, region, model_tag):
    """Load word-level data for one patient/region/model combination."""
    paths   = build_paths(pid, model_tag)
    missing = [k for k in ("xlsx","surprisal","cache") if not os.path.exists(paths[k])]
    if missing:
        return None

    df = pd.read_excel(paths["xlsx"], sheet_name="Sheet1", keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).reset_index(drop=True)
    n_words = len(df)

    emb_shape = np.load(paths["cache"], mmap_mode="r").shape
    if emb_shape[1] != n_words:
        n_words = min(emb_shape[1], n_words)
        df = df.iloc[:n_words].reset_index(drop=True)

    surp = np.full(n_words, np.nan)
    for _, row in pd.read_csv(paths["surprisal"]).iterrows():
        wi = int(row["RowIndex"])
        if wi < n_words:
            surp[wi] = float(row["surprisal"])

    sent_pos  = get_sent_positions(df)[:n_words]
    log_freq  = get_log_freq(df["CleanedWord"].astype(str).values[:n_words])
    words     = df["CleanedWord"].astype(str).values[:n_words]
    word_len  = np.array([len(w) for w in words], dtype=float)
    # is_speaker: True when the patient (Speaker1 column) has non-empty text
    is_speaker = df["Speaker1"].astype(str).str.strip().str.len().gt(0).values[:n_words]
    # sent_id: incrementing sentence index (resets each time sent_pos hits 0)
    sent_id    = (sent_pos == 0).astype(int).cumsum() - 1

    def _spk(row):
        for c in ["Speaker1","Speaker2","Speaker3","Speaker4"]:
            if str(row.get(c,"")).strip():
                return c
        return None
    spk     = df.apply(_spk, axis=1)
    is_turn = (spk != spk.shift(1)).values[:n_words]

    # neural spike vectors for the requested region
    neu_vecs = None
    if region in patient_regions and os.path.exists(paths["spikes"]):
        try:
            with h5py.File(paths["spikes"], "r") as f:
                spikes = scipy.sparse.csr_matrix((
                    f["spikes"]["data"][()],
                    f["spikes"]["ir"][()],
                    f["spikes"]["jc"][()]
                )).toarray()
                qual = np.asarray(f["qual"][()]).ravel()
                chan = np.asarray(f["chan"][()]).ravel()
            T = spikes.shape[0]
            neuron_idx = [
                idx
                for lo, hi in patient_regions[region]
                for idx in np.where(
                    (chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL)
                )[0].tolist()
            ]
            if len(neuron_idx) >= 3:
                is_spk1 = df["Speaker1"].astype(str).str.strip() != ""
                reg_sp  = spikes[:, neuron_idx]
                vecs = []
                for wi in range(n_words):
                    onset_ms = int(df.iloc[wi]["onset"])
                    t0 = max(0, onset_ms - SPEAK_OFFSET_MS) if is_spk1.iloc[wi] \
                         else max(0, onset_ms + LISTEN_OFFSET_MS)
                    t1 = min(T, t0 + WIN_MS)
                    vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64)
                                if t1 > t0 else np.zeros(len(neuron_idx)))
                neu_vecs = np.array(vecs)
        except Exception as e:
            pass

    return dict(surp=surp, emb_path=paths["cache"], n_words=n_words,
                sent_pos=sent_pos, sent_id=sent_id, log_freq=log_freq,
                neu_vecs=neu_vecs, is_turn=is_turn,
                is_speaker=is_speaker, word_len=word_len, words=words)


# ── Shared helpers ─────────────────────────────────────────────────────────────
def build_valid_mask(n, surp, sp, is_turn):
    return np.array([
        not np.isnan(surp[i+2])
        and sp[i+1] >= SENT_BOUNDARY_EXCL
        and sp[i+2] >= SENT_BOUNDARY_EXCL
        and not is_turn[i+2]
        for i in range(n - 2)
    ])


def fit_neural_pca(neu_vecs, n):
    if neu_vecs is None:
        return None, None
    nv     = neu_vecs[:n]
    max_pc = min(nv.shape[1], n - 1)
    if nv.std() == 0 or max_pc < 2:
        return None, None
    nv_sc    = StandardScaler().fit_transform(nv)
    pca_full = PCA(n_components=max_pc).fit(nv_sc)
    cum      = np.cumsum(pca_full.explained_variance_ratio_)
    n_pcs    = min(int(np.searchsorted(cum, NEURAL_VAR_THRESH) + 1), max_pc)
    return pca_full.transform(nv_sc)[:, :n_pcs], n_pcs


def fit_neural_raw(neu_vecs, n):
    """Standardize neural spike vectors without PCA reduction.

    Used for geodesic curvature decomposition where a higher-dimensional ambient
    space gives the on/off-manifold split more room to be meaningful. With PCA
    the tangent space can span most of the variance, leaving almost nothing
    genuinely perpendicular.
    """
    if neu_vecs is None:
        return None
    nv = neu_vecs[:n]
    if nv.shape[1] < 2 or nv.std() == 0:
        return None
    return StandardScaler().fit_transform(nv)


def surprisal_quantile_event_masks(surp):
    """High/low event labels from word-by-word surprisal quantiles."""
    surp = np.asarray(surp, dtype=float)
    surp_valid = surp[np.isfinite(surp)]
    hi_thr = float(np.percentile(surp_valid, EVENT_PCTILE_HI))
    lo_thr = float(np.percentile(surp_valid, EVENT_PCTILE_LO))
    return surp >= hi_thr, surp <= lo_thr, hi_thr, lo_thr


# ── Pass 1: per-layer profiles ─────────────────────────────────────────────────
def compute_profiles(pid, data, n_layers):
    n       = data["n_words"]
    surp    = data["surp"]
    sp      = data["sent_pos"]
    is_turn = data["is_turn"]

    valid = build_valid_mask(n, surp, sp, is_turn)
    if valid.sum() < MIN_PAIRS:
        return None

    surp_y = surp[np.arange(1, n-1) + 1][valid]
    Y_neu, n_pcs = fit_neural_pca(data["neu_vecs"], n)
    c_neu = trajectory_curvature(Y_neu)[valid] if Y_neu is not None else None

    N_TEMP_PERM = 50
    rng_null = np.random.default_rng(42)

    emb_mm = np.load(data["emb_path"], mmap_mode="r")
    mean_curv, std_curv = [], []
    mean_disp, std_disp = [], []
    r_surp,    p_surp   = [], []
    r_neural,  p_neural = [], []
    r_sec_surp, p_sec_surp = [], []   # sectional curvature proxy vs surprisal
    temp_null_mu, temp_null_lo, temp_null_hi = [], [], []

    for li in range(n_layers):
        emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
        c_l   = trajectory_curvature(emb_l)[valid]
        d_l   = trajectory_displacement(emb_l)[:n-2][valid]

        # temporal shuffle null: permute word order, preserve embedding distribution
        perm_means = []
        for _ in range(N_TEMP_PERM):
            emb_shuf = emb_l[rng_null.permutation(n)]
            perm_means.append(float(np.mean(trajectory_curvature(emb_shuf)[valid])))
        temp_null_mu.append(float(np.mean(perm_means)))
        temp_null_lo.append(float(np.percentile(perm_means, 2.5)))
        temp_null_hi.append(float(np.percentile(perm_means, 97.5)))

        mean_curv.append(float(np.mean(c_l)));  std_curv.append(float(np.std(c_l)))
        mean_disp.append(float(np.mean(d_l)));  std_disp.append(float(np.std(d_l)))

        r_s, p_s = pearsonr(c_l, surp_y)
        r_surp.append(float(r_s));  p_surp.append(float(p_s))

        if c_neu is not None:
            r_n, p_n = pearsonr(c_l, c_neu)
        else:
            r_n, p_n = float("nan"), float("nan")
        r_neural.append(float(r_n));  p_neural.append(float(p_n))

        # Sectional curvature slope vs surprisal (cheap — no k-NN needed)
        sec_l = scale_curvature_slope(emb_l)[valid]
        finite_sec = np.isfinite(sec_l)
        if finite_sec.sum() > 10:
            r_sc, p_sc = pearsonr(sec_l[finite_sec], surp_y[finite_sec])
        else:
            r_sc, p_sc = float("nan"), float("nan")
        r_sec_surp.append(float(r_sc));  p_sec_surp.append(float(p_sc))

    del emb_mm
    return dict(pid=pid, n=int(valid.sum()), n_pcs=n_pcs,
                mean_curv=mean_curv, std_curv=std_curv,
                mean_disp=mean_disp, std_disp=std_disp,
                r_surp=r_surp, p_surp=p_surp,
                r_neural=r_neural, p_neural=p_neural,
                r_sec_surp=r_sec_surp, p_sec_surp=p_sec_surp,
                temp_null_mu=temp_null_mu,
                temp_null_lo=temp_null_lo,
                temp_null_hi=temp_null_hi)


# ── Event-triggered analysis ───────────────────────────────────────────────────
def compute_event_triggered(data, layer, Y_neu):
    n    = data["n_words"]
    surp = data["surp"]
    W    = EVENT_WINDOW

    emb_mm  = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm

    c_sem_full = trajectory_curvature(emb_opt)
    c_neu_full = trajectory_curvature(Y_neu)[:n-2] if Y_neu is not None else None

    hi_mask, lo_mask, hi_thr, lo_thr = surprisal_quantile_event_masks(surp)

    lags   = list(range(-W, W + 1))
    hi_sem = [[] for _ in lags];  lo_sem = [[] for _ in lags]
    hi_neu = [[] for _ in lags];  lo_neu = [[] for _ in lags]

    for j in range(n):
        is_hi = hi_mask[j]
        is_lo = lo_mask[j]
        if not (is_hi or is_lo):
            continue
        for ki, k in enumerate(lags):
            # trajectory_curvature()[ci] is angle between
            # x[ci+1]-x[ci] and x[ci+2]-x[ci+1].  Therefore event word j is
            # the endpoint of the second displacement at ci=j-2; no word after
            # the event enters lag-0 curvature.
            ci = (j + k) - 2
            if 0 <= ci < len(c_sem_full):
                (hi_sem if is_hi else lo_sem)[ki].append(c_sem_full[ci])
            if c_neu_full is not None and 0 <= ci < len(c_neu_full):
                (hi_neu if is_hi else lo_neu)[ki].append(c_neu_full[ci])

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    return dict(lags=lags,
                hi_sem=[_m(x) for x in hi_sem], lo_sem=[_m(x) for x in lo_sem],
                hi_neu=[_m(x) for x in hi_neu] if c_neu_full is not None else None,
                lo_neu=[_m(x) for x in lo_neu] if c_neu_full is not None else None,
                hi_thr=hi_thr, lo_thr=lo_thr,
                event_rule="quantile_word_surprisal_endpoint_curvature",
                n_hi=len(hi_sem[W]), n_lo=len(lo_sem[W]))


# ── Pass 2: full regression ────────────────────────────────────────────────────
def run_regression(data, layer, Y_neu, n_pcs):
    n        = data["n_words"]
    surp     = data["surp"]
    sp       = data["sent_pos"]
    lf       = data["log_freq"]
    is_turn  = data["is_turn"]
    is_spk   = data["is_speaker"]
    wlen     = data["word_len"]

    valid = build_valid_mask(n, surp, sp, is_turn)
    if valid.sum() < MIN_PAIRS:
        return None

    emb_mm  = np.load(data["emb_path"], mmap_mode="r")
    emb_opt = np.array(emb_mm[layer, :n], dtype=np.float64)
    del emb_mm

    interior = np.arange(1, n - 1)
    c_sem    = trajectory_curvature(emb_opt)[valid]
    d_sem    = trajectory_displacement(emb_opt)[:n-2][valid]
    sp_v     = sp[interior][valid]
    lf_v     = lf[interior][valid]
    # Curvature at vertex t is paired with surprisal at t+1: the bend into the
    # next word, not curvature after the surprising word has passed.
    surp_y   = surp[interior + 1][valid]

    # new confounds (must be before any OLS that uses them)
    is_spk_v = is_spk[interior + 1][valid].astype(float)
    wl_v     = wlen[interior + 1][valid]
    turn_locs = np.where(is_turn)[0]
    if len(turn_locs) > 0:
        idx      = np.searchsorted(turn_locs, np.arange(n))
        before   = np.abs(np.arange(n) - turn_locs[np.clip(idx - 1, 0, len(turn_locs)-1)])
        after    = np.abs(np.arange(n) - turn_locs[np.clip(idx,     0, len(turn_locs)-1)])
        dist_arr = np.minimum(before, after).astype(float)
    else:
        dist_arr = np.full(n, float(n))
    dt_v = dist_arr[interior][valid]

    # ── Geodesic curvature decomposition at the LOO layer ──────────────────────
    geo_arr, nrm_arr = geodesic_curvature(emb_opt, k_neighbors=20, tangent_dim=10)
    c_geo = geo_arr[valid]
    c_nrm = nrm_arr[valid]
    fin_geo = np.isfinite(c_geo)
    fin_nrm = np.isfinite(c_nrm)
    r_sem_geo, p_sem_geo = (pearsonr(c_geo[fin_geo], surp_y[fin_geo])
                            if fin_geo.sum() > 10 else (float("nan"), float("nan")))
    r_sem_nrm, p_sem_nrm = (pearsonr(c_nrm[fin_nrm], surp_y[fin_nrm])
                             if fin_nrm.sum() > 10 else (float("nan"), float("nan")))
    t_geo = ols_numpy(surp_y, [c_geo, sp_v, lf_v, is_spk_v, wl_v, dt_v],
                      ["c_geo","sent_pos","log_freq","is_speaker","word_len","dist_turn"])

    # ── Surprisal-binned geodesic/normal curvature (quintiles) ────────────────
    N_BINS = 5
    bin_idx = None
    fin_gn = np.isfinite(c_geo) & np.isfinite(c_nrm) & np.isfinite(surp_y)
    if fin_gn.sum() >= N_BINS * 5:
        q_edges = np.percentile(surp_y[fin_gn], np.linspace(0, 100, N_BINS + 1))
        q_edges[-1] += 1e-9
        bin_idx = np.digitize(surp_y, q_edges) - 1
        bin_idx = np.clip(bin_idx, 0, N_BINS - 1)
        geo_by_bin = [float(np.mean(c_geo[fin_gn & (bin_idx == b)]))
                      if (fin_gn & (bin_idx == b)).sum() > 0 else float("nan")
                      for b in range(N_BINS)]
        nrm_by_bin = [float(np.mean(c_nrm[fin_gn & (bin_idx == b)]))
                      if (fin_gn & (bin_idx == b)).sum() > 0 else float("nan")
                      for b in range(N_BINS)]
        surp_bin_centers = [float((q_edges[b] + q_edges[b+1]) / 2)
                            for b in range(N_BINS)]
    else:
        geo_by_bin = nrm_by_bin = surp_bin_centers = [float("nan")] * N_BINS

    # ── Sectional curvature slope at the LOO layer ─────────────────────────────
    sec_arr = scale_curvature_slope(emb_opt)[valid]
    fin_sec = np.isfinite(sec_arr)
    mean_sec_slope = float(np.nanmean(sec_arr))
    r_sem_sec, p_sem_sec = (pearsonr(sec_arr[fin_sec], surp_y[fin_sec])
                             if fin_sec.sum() > 10 else (float("nan"), float("nan")))

    r_sem, p_sem = pearsonr(c_sem, surp_y)
    # baseline model: sent_pos + log_freq
    t_base = ols_numpy(surp_y, [c_sem, sp_v, lf_v],
                       ["c_sem","sent_pos","log_freq"])
    # full model: + is_speaker + word_len + dist_to_turn
    t_full = ols_numpy(surp_y, [c_sem, sp_v, lf_v, is_spk_v, wl_v, dt_v],
                       ["c_sem","sent_pos","log_freq","is_speaker","word_len","dist_turn"])

    if Y_neu is not None:
        c_neu        = trajectory_curvature(Y_neu)[valid]
        r_neu, p_neu = pearsonr(c_neu, surp_y)
        r_sn,  p_sn  = pearsonr(c_sem, c_neu)
        same_dir     = bool(np.sign(r_neu) == np.sign(r_sem))
        t_neu = ols_numpy(surp_y, [c_sem, c_neu, sp_v, lf_v, is_spk_v, wl_v, dt_v],
                          ["c_sem","c_neu","sent_pos","log_freq",
                           "is_speaker","word_len","dist_turn"])
        delta_r2 = float(t_neu["r2"] - t_full["r2"])
        p_delta  = float(t_neu["pvals"]["c_neu"])

        # Geodesic + normal curvature of neural trajectory.
        # Use raw-standardized data (not PCA-reduced) so the ambient space has
        # enough dimensions for the perpendicular component to be meaningful.
        Y_neu_raw = fit_neural_raw(data["neu_vecs"], n)
        _neu_for_geo = Y_neu_raw if Y_neu_raw is not None else Y_neu
        geo_neu_arr, nrm_neu_arr = geodesic_curvature(
            _neu_for_geo,
            k_neighbors=min(15, len(_neu_for_geo) - 1),
            tangent_dim=min(5, _neu_for_geo.shape[1] - 1))
        c_geo_neu = geo_neu_arr[valid]
        c_nrm_neu = nrm_neu_arr[valid]
        fin_gn  = np.isfinite(c_geo_neu) & np.isfinite(c_geo)
        fin_nn  = np.isfinite(c_nrm_neu) & np.isfinite(surp_y)
        r_geo_sn, p_geo_sn = (pearsonr(c_geo[fin_gn], c_geo_neu[fin_gn])
                               if fin_gn.sum() > 10 else (float("nan"), float("nan")))
        r_neu_geo, p_neu_geo = (pearsonr(c_geo_neu[fin_gn & fin_geo], surp_y[fin_gn & fin_geo])
                                 if (fin_gn & fin_geo).sum() > 10 else (float("nan"), float("nan")))
        r_neu_nrm, p_neu_nrm = (pearsonr(c_nrm_neu[fin_nn], surp_y[fin_nn])
                                  if fin_nn.sum() > 10 else (float("nan"), float("nan")))

        # Hippocampal component activation by the same surprisal quintiles used
        # for the LLM components above.
        fin_hpc_bins = (np.isfinite(c_geo_neu) & np.isfinite(c_nrm_neu) &
                        np.isfinite(c_neu) & np.isfinite(surp_y))
        if bin_idx is not None and fin_hpc_bins.sum() >= N_BINS * 5:
            hpc_geo_by_bin = [
                float(np.mean(c_geo_neu[fin_hpc_bins & (bin_idx == b)]))
                if (fin_hpc_bins & (bin_idx == b)).sum() > 0 else float("nan")
                for b in range(N_BINS)]
            hpc_nrm_by_bin = [
                float(np.mean(c_nrm_neu[fin_hpc_bins & (bin_idx == b)]))
                if (fin_hpc_bins & (bin_idx == b)).sum() > 0 else float("nan")
                for b in range(N_BINS)]
            hpc_curv_by_bin = [
                float(np.mean(c_neu[fin_hpc_bins & (bin_idx == b)]))
                if (fin_hpc_bins & (bin_idx == b)).sum() > 0 else float("nan")
                for b in range(N_BINS)]
        else:
            hpc_geo_by_bin = hpc_nrm_by_bin = hpc_curv_by_bin = [float("nan")] * N_BINS

        # Full 2×2 LLM↔hpc component coupling matrix
        fin_nn_neu  = np.isfinite(c_nrm) & np.isfinite(c_nrm_neu)
        fin_gn_cross = np.isfinite(c_geo) & np.isfinite(c_nrm_neu)
        fin_ng_cross = np.isfinite(c_nrm) & np.isfinite(c_geo_neu)
        r_nrm_sn, p_nrm_sn = (pearsonr(c_nrm[fin_nn_neu], c_nrm_neu[fin_nn_neu])
                               if fin_nn_neu.sum() > 10 else (float("nan"), float("nan")))
        r_geo_nrm_sn, p_geo_nrm_sn = (pearsonr(c_geo[fin_gn_cross], c_nrm_neu[fin_gn_cross])
                                       if fin_gn_cross.sum() > 10 else (float("nan"), float("nan")))
        r_nrm_geo_sn, p_nrm_geo_sn = (pearsonr(c_nrm[fin_ng_cross], c_geo_neu[fin_ng_cross])
                                       if fin_ng_cross.sum() > 10 else (float("nan"), float("nan")))

        # LLM-curvature-binned neural curvature (quintiles): how does hippocampal
        # curvature track LLM curvature across the curvature range?
        fin_sn = np.isfinite(c_sem) & np.isfinite(c_neu)
        if fin_sn.sum() >= N_BINS * 5:
            qe = np.percentile(c_sem[fin_sn], np.linspace(0, 100, N_BINS + 1))
            qe[-1] += 1e-9
            bidx = np.clip(np.digitize(c_sem, qe) - 1, 0, N_BINS - 1)
            neu_by_sem_bin = [
                float(np.mean(c_neu[fin_sn & (bidx == b)]))
                if (fin_sn & (bidx == b)).sum() > 0 else float("nan")
                for b in range(N_BINS)]
            sem_curv_bin_centers = [
                float((qe[b] + qe[b + 1]) / 2) for b in range(N_BINS)]
        else:
            neu_by_sem_bin = sem_curv_bin_centers = [float("nan")] * N_BINS
    else:
        r_neu = p_neu = r_sn = p_sn = delta_r2 = p_delta = float("nan")
        r_geo_sn = p_geo_sn = r_neu_geo = p_neu_geo = float("nan")
        r_neu_nrm = p_neu_nrm = float("nan")
        r_nrm_sn = p_nrm_sn = r_geo_nrm_sn = p_geo_nrm_sn = float("nan")
        r_nrm_geo_sn = p_nrm_geo_sn = float("nan")
        same_dir = None
        hpc_geo_by_bin = hpc_nrm_by_bin = hpc_curv_by_bin = [float("nan")] * N_BINS
        neu_by_sem_bin = sem_curv_bin_centers = [float("nan")] * N_BINS

    # curvature vs displacement comparison (full confound set)
    t4_curv = ols_numpy(surp_y, [c_sem, sp_v, lf_v, is_spk_v, wl_v, dt_v],
                        ["c_sem","sent_pos","log_freq","is_speaker","word_len","dist_turn"])
    t4_disp = ols_numpy(surp_y, [d_sem, sp_v, lf_v, is_spk_v, wl_v, dt_v],
                        ["disp","sent_pos","log_freq","is_speaker","word_len","dist_turn"])
    t4_both = ols_numpy(surp_y, [c_sem, d_sem, sp_v, lf_v, is_spk_v, wl_v, dt_v],
                        ["c_sem","disp","sent_pos","log_freq","is_speaker","word_len","dist_turn"])

    # extra geometry at the LOO layer
    t_arr   = trajectory_torsion(emb_opt)
    t_valid = t_arr[valid[:-1]] if len(t_arr) >= valid[:-1].sum() else t_arr
    mean_torsion   = float(np.mean(t_valid)) if len(t_valid) > 0 else float("nan")
    straightness   = trajectory_straightness(emb_opt[valid.nonzero()[0] + 1])
    mean_curv_loo  = float(np.mean(c_sem))
    mean_disp_loo  = float(np.mean(d_sem))

    return dict(layer=int(layer), n=int(valid.sum()), n_pcs=n_pcs,
                r_sem=float(r_sem), p_sem=float(p_sem),
                beta_c_sem=float(t_base["betas"]["c_sem"]),
                p_beta_sem=float(t_base["pvals"]["c_sem"]),
                r2=float(t_base["r2"]),
                beta_c_sem_full=float(t_full["betas"]["c_sem"]),
                p_beta_sem_full=float(t_full["pvals"]["c_sem"]),
                r2_full=float(t_full["r2"]),
                r_neu=float(r_neu), p_neu=float(p_neu),
                r_sem_neu=float(r_sn), p_sem_neu=float(p_sn),
                same_dir=same_dir, delta_r2=delta_r2, p_delta=p_delta,
                r2_curv=float(t4_curv["r2"]),
                r2_disp=float(t4_disp["r2"]),
                r2_both=float(t4_both["r2"]),
                mean_curv=mean_curv_loo, mean_disp=mean_disp_loo,
                mean_torsion=mean_torsion, straightness=straightness,
                # geodesic curvature decomposition
                r_sem_geo=float(r_sem_geo), p_sem_geo=float(p_sem_geo),
                r_sem_nrm=float(r_sem_nrm), p_sem_nrm=float(p_sem_nrm),
                beta_c_geo=float(t_geo["betas"]["c_geo"]),
                p_beta_geo=float(t_geo["pvals"]["c_geo"]),
                r2_geo=float(t_geo["r2"]),
                # sectional curvature proxy
                mean_sec_slope=float(mean_sec_slope),
                r_sem_sec=float(r_sem_sec), p_sem_sec=float(p_sem_sec),
                # neural geodesic/normal alignment
                r_geo_sem_neu=float(r_geo_sn), p_geo_sem_neu=float(p_geo_sn),
                r_neu_geo=float(r_neu_geo), p_neu_geo=float(p_neu_geo),
                r_neu_nrm=float(r_neu_nrm), p_neu_nrm=float(p_neu_nrm),
                # full 2×2 component coupling matrix (LLM component × hpc component)
                r_nrm_sem_neu=float(r_nrm_sn), p_nrm_sem_neu=float(p_nrm_sn),
                r_geo_sem_nrm_neu=float(r_geo_nrm_sn), p_geo_sem_nrm_neu=float(p_geo_nrm_sn),
                r_nrm_sem_geo_neu=float(r_nrm_geo_sn), p_nrm_sem_geo_neu=float(p_nrm_geo_sn),
                # surprisal-binned geodesic/normal curvature (quintiles 1=low→5=high)
                geo_by_surp_bin=geo_by_bin,
                nrm_by_surp_bin=nrm_by_bin,
                surp_bin_centers=surp_bin_centers,
                hpc_geo_by_surp_bin=hpc_geo_by_bin,
                hpc_nrm_by_surp_bin=hpc_nrm_by_bin,
                hpc_curv_by_surp_bin=hpc_curv_by_bin,
                # LLM-curvature-binned neural curvature (quintiles 1=low→5=high)
                neu_by_sem_curv_bin=neu_by_sem_bin,
                sem_curv_bin_centers=sem_curv_bin_centers)


# ── Trajectory visualisation helper ───────────────────────────────────────────
def find_best_sentence_traj(data, layer):
    """
    Find the sentence with the widest surprisal range (≥8 words with surprisal).
    Returns 2D PCA trajectories for LLM and neural space, Procrustes-aligned.
    """
    if data["neu_vecs"] is None:
        return None

    n        = data["n_words"]
    surp     = data["surp"]
    sent_id  = data["sent_id"]
    words    = data["words"]

    best_range, best_wis = 0., None
    for sid in np.unique(sent_id):
        wis = np.where(sent_id == sid)[0]
        if len(wis) < 8:
            continue
        sv = surp[wis]
        if np.isnan(sv).sum() > len(wis) // 2:
            continue
        sr = float(np.nanmax(sv) - np.nanmin(sv))
        if sr > best_range:
            best_range, best_wis = sr, wis

    if best_wis is None:
        return None

    emb_mm   = np.load(data["emb_path"], mmap_mode="r")
    emb_sent = np.array(emb_mm[layer, best_wis], dtype=np.float64)
    del emb_mm
    neu_sent = data["neu_vecs"][best_wis].astype(np.float64)

    def _pca2d(X):
        k = min(2, X.shape[1], X.shape[0] - 1)
        if k < 2:
            return None
        Xc = X - X.mean(0)
        return PCA(n_components=k).fit_transform(Xc)

    llm_2d = _pca2d(emb_sent)
    neu_2d = _pca2d(neu_sent)
    if llm_2d is None or neu_2d is None:
        return None

    try:
        _, neu_aligned, disparity = scipy_procrustes(llm_2d, neu_2d)
    except Exception:
        neu_aligned, disparity = neu_2d, float("nan")

    return dict(
        words=words[best_wis].tolist(),
        surprisal=surp[best_wis].tolist(),
        llm_pca2d=llm_2d.tolist(),
        neu_pca2d=neu_aligned.tolist(),
        procrustes_disparity=float(disparity),
        surp_range=float(best_range),
        layer=int(layer),
    )


# ── One model × region run ─────────────────────────────────────────────────────
def run_one(model_tag, n_layers, region):
    print(f"\n{'='*65}")
    print(f"  MODEL: {model_tag}   REGION: {region}")
    print(f"{'='*65}")

    # Pass 1
    all_data, all_profs = {}, {}
    for pid, preg in PATIENTS.items():
        if region not in preg:
            continue
        data = load_patient(pid, preg, region, model_tag)
        if data is None:
            continue
        all_data[pid] = data
        prof = compute_profiles(pid, data, n_layers)
        if prof is not None:
            all_profs[pid] = prof
            print(f"  {pid}  n={prof['n']}  n_pcs={prof['n_pcs']}")

    pids = list(all_profs.keys())
    if len(pids) < 2:
        print(f"  Too few patients ({len(pids)}) — skip.")
        return None

    # LOO layer selection
    loo_surp, loo_neural = {}, {}
    for pid in pids:
        others = [p for p in pids if p != pid]
        rs_mat = np.array([all_profs[p]["r_surp"]   for p in others])
        rn_mat = np.array([all_profs[p]["r_neural"]  for p in others], dtype=float)
        loo_surp[pid]   = int(np.argmax(np.abs(rs_mat.mean(axis=0))))
        rn_mean = np.nanmean(rn_mat, axis=0)
        loo_neural[pid] = (int(np.nanargmax(np.abs(rn_mean)))
                           if not np.all(np.isnan(rn_mean))
                           else loo_surp[pid])

    print(f"\n  LOO-neural layers: { {p: loo_neural[p] for p in pids} }")

    # Pass 2
    results = []
    for pid in pids:
        data    = all_data[pid]
        layer_n = loo_neural[pid]
        Y_neu, n_pcs = fit_neural_pca(data["neu_vecs"], data["n_words"])
        reg  = run_regression(data, layer_n, Y_neu, n_pcs)
        evt  = compute_event_triggered(data, layer_n, Y_neu)
        traj = find_best_sentence_traj(data, layer_n)
        results.append(dict(pid=pid,
                            loo_layer_surp=loo_surp[pid],
                            loo_layer_neu=loo_neural[pid],
                            profile=all_profs[pid],
                            regression=reg,
                            event_triggered=evt,
                            traj_viz=traj))
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",  nargs="+", default=list(MODELS.keys()),
                        help="Which models to run (default: all)")
    parser.add_argument("--regions", nargs="+", default=REGIONS,
                        help="Which regions to run (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if output JSON already exists")
    args = parser.parse_args()

    for model_tag in args.models:
        if model_tag not in MODELS:
            print(f"  Unknown model: {model_tag}  (choices: {list(MODELS)})")
            continue
        n_layers = MODELS[model_tag]
        for region in args.regions:
            out_path = os.path.join(RESULTS, f"geometry_{model_tag}_{region}.json")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"  [skip] {model_tag} / {region} — already exists")
                continue
            results = run_one(model_tag, n_layers, region)
            if results is None:
                continue
            with open(out_path, "w") as fh:
                json.dump(results, fh, indent=2)
            print(f"  Saved → {out_path}")
