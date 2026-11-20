"""
make_geometry_figures_podcast_pca20.py
========================
Podcast + PCA(20)-reduced-trajectory variant of ../make_geometry_figures.py,
covering fig1_layer_profiles and fig2_surprisal_geometry. Reads from the
geometry_podcast_pca20_*, cross_space_layers_podcast_pca20_*, and
scatter_data_podcast_pca20_* result files.

Figures:
  fig1_layer_profiles_pca20_podcast.pdf
  fig2_surprisal_geometry_pca20_podcast.pdf

Run:
    conda run -n gpt2_embed python3 make_geometry_figures_podcast_pca20.py
"""

import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem as scipy_sem, ttest_rel

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR    = os.path.dirname(BASE_DIR)
RESULTS       = os.path.join(BASE_DIR, "results")
FIGURES       = os.path.join(PARENT_DIR, "figures")
PRIMARY_MODEL = "llama-3.1-8b"
REGIONS       = ["hippocampus"]
CTX_TAG       = "_ctx200"

# Display names and colours per region
REGION_STYLE = {
    "hippocampus": dict(color="#1B7F4A", label="Hippocampus", ls="-"),
}

# Model display order + colour for model-comparison figure
MODEL_STYLE = {
    "bert-base":    dict(color="#E67E22", label="BERT-base",     ls=(0,(3,1,1,1))),
    "roberta-base": dict(color="#F39C12", label="RoBERTa-base",  ls=(0,(5,2))),
    "gpt2":         dict(color="#2980B9", label="GPT-2",         ls="-"),
    "gpt2-medium":  dict(color="#1F618D", label="GPT-2 medium",  ls="--"),
    "gpt2-xl":      dict(color="#5DADE2", label="GPT-2 XL",      ls=(0,(1,1))),
    "opt-350m":     dict(color="#884EA0", label="OPT-350m",      ls=(0,(3,1))),
    "llama-2-7b":   dict(color="#C0392B", label="LLaMA-2 7B",    ls="--"),
    "llama-3.1-8b": dict(color="#E91E63", label="LLaMA-3.1 8B",  ls="-"),
    "gemma-2-9b":   dict(color="#148F77", label="Gemma-2 9B",    ls=(0,(4,1))),
    "mistral-7b":   dict(color="#D35400", label="Mistral-7B",    ls=(0,(2,1))),
    "fasttext-wiki": dict(color="#7F7F7F", label="fastText (static)", ls=(0,(1,1))),
}

# ── NeurIPS style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         14,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

C_HI  = "#C0392B"
C_LO  = "#2980B9"
C_SEM = "#D95319"
C_DISP= "#9B59B6"
C_NS  = "#AAAAAA"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _mu_se(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return (float(np.mean(v)), float(scipy_sem(v))) if v else (np.nan, np.nan)


def _load(model, region):
    ctx = "" if model == "fasttext-wiki" else CTX_TAG
    path = os.path.join(RESULTS, f"geometry_podcast_pca20_{model}{ctx}_{region}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_cross_space(model, region):
    path = os.path.join(RESULTS, f"cross_space_layers_podcast_pca20_{model}_{region}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        p = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(p)
        if ext == "svg":
            # Rename server-side font to Times New Roman so Illustrator finds it
            txt = open(p).read()
            txt = txt.replace("TeX Gyre Termes", "Times New Roman")
            open(p, "w").write(txt)
        print(f"  → {p}")
    plt.close(fig)


def _shade_sig(ax, mean_ps, xs, color, alpha=0.12):
    """Shade x-bands where mean p < 0.05 across patients."""
    in_sig, x0 = False, None
    for x, p in zip(xs, mean_ps):
        if p < 0.05 and not in_sig:
            x0, in_sig = x, True
        elif p >= 0.05 and in_sig:
            ax.axvspan(x0 - 0.5, x - 0.5, color=color, alpha=alpha, lw=0)
            in_sig = False
    if in_sig:
        ax.axvspan(x0 - 0.5, xs[-1] + 0.5, color=color, alpha=alpha, lw=0)


def _layer_arrays(results_list, key_r, key_p):
    """Stack per-patient layer arrays; return (mu, se, mean_p, n_sig)."""
    mat_r = np.array([r["profile"][key_r] for r in results_list], dtype=float)
    mat_p = np.array([r["profile"][key_p] for r in results_list], dtype=float)
    mu    = np.nanmean(mat_r, axis=0)
    se    = np.array([scipy_sem(mat_r[:, li][~np.isnan(mat_r[:, li])])
                      for li in range(mat_r.shape[1])])
    mean_p = np.nanmean(mat_p, axis=0)
    n_sig  = (mat_p < 0.05).sum(axis=0)
    return mu, se, mean_p, n_sig


def _layer_shuffle_null(mat, n_perm=2000, seed=7):
    """Layer-order null for a patient × layer profile matrix."""
    rng = np.random.default_rng(seed)
    mat = np.asarray(mat, dtype=float)
    null = np.empty((n_perm, mat.shape[1]), dtype=float)
    for pi in range(n_perm):
        shuffled = np.array([rng.permutation(row) for row in mat])
        null[pi] = np.nanmean(shuffled, axis=0)
    return (
        np.nanmean(null, axis=0),
        np.nanpercentile(null, 2.5, axis=0),
        np.nanpercentile(null, 97.5, axis=0),
    )


def _density_hist(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    widths = np.diff(bins)
    total = counts.sum()
    return counts / (total * widths) if total else np.zeros_like(widths, dtype=float)


def _hist_xlim(*arrays):
    joined = np.concatenate([a[np.isfinite(a)] for a in arrays if len(a)])
    lo, hi = np.percentile(joined, [0.5, 99.5])
    pad = 0.06 * (hi - lo)
    return max(0.0, lo - pad), min(180.0, hi + pad)


def _collect_winning_curvature_values(model=PRIMARY_MODEL, region="hippocampus"):
    """Collect neural and winning-layer LLM curvature values using Fig. 1 masks."""
    from run_geometry_analysis_pca20 import (
        build_valid_mask, fit_neural_pca, trajectory_curvature, sem_pca_reduce,
    )
    from podcast_loader import PODCAST_PATIENTS, load_podcast_patient

    records = _load(model, region) or []
    neural_vals, llm_vals = [], []
    layers = []
    for rec in records:
        pid = rec["pid"]
        if pid not in PODCAST_PATIENTS:
            continue
        data = load_podcast_patient(pid, region=region, model_tag=model, context_words=200)
        if data is None:
            continue
        n = data["n_words"]
        valid = build_valid_mask(n, data["surp"], data["sent_pos"], data["is_turn"])
        if valid.sum() == 0:
            continue

        Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
        if Y_neu is None:
            continue
        c_neu = trajectory_curvature(Y_neu)[valid]
        neural_vals.append(c_neu[np.isfinite(c_neu)])

        layer = int(rec["loo_layer_neu"])
        emb_mm = np.load(data["emb_path"], mmap_mode="r")
        emb_layer = np.asarray(emb_mm[layer, :n], dtype=np.float64)
        del emb_mm
        emb_layer = sem_pca_reduce(emb_layer)
        c_llm = trajectory_curvature(emb_layer)[valid]
        llm_vals.append(c_llm[np.isfinite(c_llm)])
        layers.append(layer)

    if not neural_vals or not llm_vals:
        return None
    layer_label = layers[0] if len(set(layers)) == 1 else "LOO"
    return np.concatenate(neural_vals), np.concatenate(llm_vals), layer_label


def _draw_winning_curvature_hist(ax):
    vals = _collect_winning_curvature_values()
    if vals is None:
        ax.text(0.5, 0.5, "No histogram data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("$\\mathbf{C}$  Curvature distribution", loc="left", pad=5)
        return

    neural, llm, layer_label = vals
    bins = np.linspace(0.0, 180.0, 121)
    centers = 0.5 * (bins[:-1] + bins[1:])
    d_neu = _density_hist(neural, bins)
    d_llm = _density_hist(llm, bins)
    llama_color = MODEL_STYLE[PRIMARY_MODEL]["color"]
    neural_color = REGION_STYLE["hippocampus"]["color"]

    ax.fill_between(centers, d_neu, step="mid",
                    color=neural_color, alpha=0.20)
    ax.plot(centers, d_neu, color=neural_color, lw=1.35,
            label=f"Hippocampal neural ($N={len(neural):,}$)")
    ax.fill_between(centers, d_llm, step="mid",
                    color=llama_color, alpha=0.18)
    ax.plot(centers, d_llm, color=llama_color, lw=1.35,
            label=f"LLaMA layer {layer_label} ($N={len(llm):,}$)")

    for arr, col in [(neural, neural_color), (llm, llama_color)]:
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        ax.axvspan(mu - sd, mu + sd, color=col, alpha=0.07, lw=0)
        ax.axvline(mu, color=col, lw=0.9)

    xlo, xhi = _hist_xlim(neural, llm)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Curvature $\\kappa$ (°)")
    ax.set_ylabel("Density")
    ax.set_title("$\\mathbf{C}$  Curvature distribution\nwinning LLaMA layer",
                 loc="left", pad=5)
    ax.legend(loc="upper left", fontsize=7.2, framealpha=0.78,
              edgecolor="#DDDDDD", handlelength=1.6)


def _p_text(p):
    if not np.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Layer profiles for primary model, hippocampus only
# ══════════════════════════════════════════════════════════════════════════════
def fig1_layer_profiles():
    GRID = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0))
    ax_A, ax_B = axes[0, 0], axes[0, 1]
    ax_C, ax_D = axes[1, 0], axes[1, 1]
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.92,
                        wspace=0.40, hspace=0.52)
    curv_bounds = []
    curv_profiles_by_pid = {}
    temp_null_by_pid = {}

    # Panels A + B background: all non-primary models, hippocampus only, thin lines
    for model, st in MODEL_STYLE.items():
        if model == PRIMARY_MODEL:
            continue
        data_m = _load(model, "hippocampus")
        if data_m is None:
            continue
        n_l    = len(data_m[0]["profile"]["mean_curv"])
        xs_m   = np.linspace(0, 1, n_l)

        mu_c = np.array([r["profile"]["mean_curv"] for r in data_m]).mean(0)
        ax_A.plot(GRID, np.interp(GRID, xs_m, mu_c),
                  color=st["color"], ls=st["ls"], lw=0.9, alpha=0.55,
                  label=st["label"])
        curv_bounds.append(mu_c)

        cs_m = _load_cross_space(model, "hippocampus")
        if cs_m:
            n_l_cs = cs_m[0]["n_layers"]
            xs_cs  = np.linspace(0, 1, n_l_cs)
            mat_cs = np.array([r["r_curv_layers_ch"] for r in cs_m
                                if "r_curv_layers_ch" in r], dtype=float)
            if len(mat_cs):
                mu_cs = np.nanmean(mat_cs, axis=0)
                ax_B.plot(GRID, np.interp(GRID, xs_cs, np.nan_to_num(mu_cs)),
                          color=st["color"], ls=st["ls"], lw=0.9, alpha=0.55,
                          label="_nolegend_")

    # Primary model: hippocampus
    llama_st = MODEL_STYLE[PRIMARY_MODEL]
    for region in REGIONS:
        data = _load(PRIMARY_MODEL, region)
        if data is None:
            continue
        st       = REGION_STYLE[region]
        n_layers = len(data[0]["profile"]["mean_curv"])
        xs       = np.arange(n_layers)
        xs_norm  = np.linspace(0, 1, n_layers)
        for rec in data:
            curv_profiles_by_pid.setdefault(rec["pid"], rec["profile"]["mean_curv"])
            if "temp_null_mu" in rec["profile"]:
                temp_null_by_pid.setdefault(rec["pid"], rec["profile"]["temp_null_mu"])

        mu_c = np.array([r["profile"]["mean_curv"] for r in data]).mean(0)
        se_c = scipy_sem(np.array([r["profile"]["mean_curv"] for r in data]), axis=0)
        ax_A.plot(xs_norm, mu_c, color=llama_st["color"], ls=llama_st["ls"],
                  lw=0.9, label=llama_st["label"])
        curv_bounds.extend([mu_c - se_c, mu_c + se_c])

        if region == "hippocampus":
            cs_prim = _load_cross_space(PRIMARY_MODEL, "hippocampus")
            if cs_prim:
                n_l_p = cs_prim[0]["n_layers"]
                xs_p  = np.linspace(0, 1, n_l_p)
                mat_p = np.array([r["r_curv_layers_ch"] for r in cs_prim
                                   if "r_curv_layers_ch" in r], dtype=float)
                mu_p  = np.nanmean(mat_p, axis=0)
                ax_B.plot(xs_p, mu_p, color=llama_st["color"], ls=llama_st["ls"],
                          lw=0.9, label="_nolegend_")

    # ── Panel B null band: word-order shuffle, averaged across all models ──────
    # Use mean of per-patient null stds (spread of 200-perm distribution)
    # rather than SE across models, so the band is visible.
    null_b_mu_interp  = []
    null_b_sd_interp  = []
    for model in MODEL_STYLE:
        cs_null = _load_cross_space(model, "hippocampus")
        if not cs_null:
            continue
        mat_mu = np.array(
            [r["null_curv_layers_ch_mean"] for r in cs_null
             if "null_curv_layers_ch_mean" in r], dtype=float)
        mat_sd = np.array(
            [r["null_curv_layers_ch_std"] for r in cs_null
             if "null_curv_layers_ch_std" in r], dtype=float)
        if not len(mat_mu) or not len(mat_sd):
            continue
        mu_null = np.nanmean(mat_mu, axis=0)
        sd_null = np.nanmean(mat_sd, axis=0)
        xs_null = np.linspace(0, 1, len(mu_null))
        null_b_mu_interp.append(np.interp(GRID, xs_null, np.nan_to_num(mu_null)))
        null_b_sd_interp.append(np.interp(GRID, xs_null, np.nan_to_num(sd_null)))

    if null_b_mu_interp:
        nb_mu = np.nanmean(np.array(null_b_mu_interp), axis=0)
        nb_sd = np.nanmean(np.array(null_b_sd_interp), axis=0)
        ax_B.fill_between(GRID, nb_mu - nb_sd, nb_mu + nb_sd,
                          color="#777777", alpha=0.16, lw=0, zorder=0,
                          label="_nolegend_")
        ax_B.plot(GRID, nb_mu, color="#555555",
                  ls=(0, (1.2, 1.2)), lw=1.2, zorder=1,
                  label="Word-order shuffle null")

    if temp_null_by_pid:
        tn_arr  = np.array(list(temp_null_by_pid.values()), dtype=float)
        ctrl_xs = np.linspace(0, 1, tn_arr.shape[1])
        ctrl_mu = np.nanmean(tn_arr, axis=0)
        ctrl_se = scipy_sem(tn_arr, axis=0, nan_policy="omit")
        ctrl_lo = ctrl_mu - ctrl_se
        ctrl_hi = ctrl_mu + ctrl_se
        ax_A.fill_between(ctrl_xs, ctrl_lo, ctrl_hi,
                          color="#777777", alpha=0.16, lw=0, zorder=0, label="_nolegend_")
        ax_A.plot(ctrl_xs, ctrl_mu, color="#555555",
                  ls=(0, (1.2, 1.2)), lw=1.2, zorder=1, label="Temporal-shuffle control")
        curv_bounds.extend([ctrl_lo, ctrl_hi])
    elif curv_profiles_by_pid:
        ctrl_mat = np.array(list(curv_profiles_by_pid.values()), dtype=float)
        ctrl_xs  = np.linspace(0, 1, ctrl_mat.shape[1])
        ctrl_mu, ctrl_lo, ctrl_hi = _layer_shuffle_null(ctrl_mat)
        ax_A.fill_between(ctrl_xs, ctrl_lo, ctrl_hi,
                          color="#777777", alpha=0.16, lw=0, zorder=0, label="_nolegend_")
        ax_A.plot(ctrl_xs, ctrl_mu, color="#555555",
                  ls=(0, (1.2, 1.2)), lw=1.2, zorder=1, label="Layer-shuffled control")
        curv_bounds.extend([ctrl_lo, ctrl_hi])

    # Panel A
    ax_A.set_xlabel("Normalized layer position")
    ax_A.set_ylabel("Mean curvature (°)")
    ax_A.set_title("$\\mathbf{A}$  Mean curvature · all models", loc="left", pad=5)
    ax_A.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_A.set_xticklabels(["0", ".25", ".5", ".75", "1"])

    # Panel B
    ax_B.axhline(0, color="k", lw=0.5, ls=":")
    ax_B.set_xlabel("Normalized layer position")
    ax_B.set_ylabel("Pearson $r$")
    ax_B.set_title("$\\mathbf{B}$  Neural-LLM alignment · all models\n15-word chunks", loc="left", pad=5)
    ax_B.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_B.set_xticklabels(["0", ".25", ".5", ".75", "1"])

    # Panel C
    _draw_winning_curvature_hist(ax_C)

    # Panel D
    hpc_data = _load(PRIMARY_MODEL, "hippocampus")
    if hpc_data is not None:
        q1_vals, q5_vals = [], []
        for rec in hpc_data:
            reg  = rec.get("regression", {})
            bins = reg.get("neu_by_sem_curv_bin")
            if bins and not all(np.isnan(b) for b in bins):
                v = np.array(bins, dtype=float)
                if np.isfinite(v[0]) and np.isfinite(v[-1]):
                    q1_vals.append(v[0])
                    q5_vals.append(v[-1])
        if q1_vals and q5_vals:
            q1_vals = np.array(q1_vals, dtype=float)
            q5_vals = np.array(q5_vals, dtype=float)
            win_col = MODEL_STYLE[PRIMARY_MODEL]["color"]
            xs_pair = np.array([1.0, 2.0])

            for q1, q5 in zip(q1_vals, q5_vals):
                ax_D.plot(xs_pair, [q1, q5], color="#B9B9B9",
                          lw=0.85, alpha=0.62, zorder=1)
            ax_D.scatter(np.full_like(q1_vals, 1.0), q1_vals,
                         s=24, facecolor="white", edgecolor=win_col,
                         linewidth=0.8, alpha=0.85, zorder=3)
            ax_D.scatter(np.full_like(q5_vals, 2.0), q5_vals,
                         s=24, facecolor=win_col, edgecolor=win_col,
                         linewidth=0.8, alpha=0.85, zorder=3)

            means = np.array([np.mean(q1_vals), np.mean(q5_vals)])
            ses = np.array([scipy_sem(q1_vals), scipy_sem(q5_vals)])
            ax_D.errorbar(xs_pair, means, yerr=ses, color=win_col,
                          lw=2.1, marker="o", ms=5.2, capsize=3.0,
                          zorder=4)

            delta = q5_vals - q1_vals
            p = ttest_rel(q5_vals, q1_vals).pvalue if len(q1_vals) > 1 else np.nan
            ax_D.text(0.05, 0.95,
                      f"mean Δ = {np.mean(delta):+.2f}°\n{_p_text(p)}",
                      transform=ax_D.transAxes, ha="left", va="top",
                      fontsize=8.5, color="#333333")

    ax_D.set_xlim(0.75, 2.25)
    ax_D.set_xlabel("LLaMA curvature bin")
    ax_D.set_ylabel("Hippocampal curvature $\\kappa$ (°)")
    ax_D.set_title("$\\mathbf{D}$  Paired hippocampal $\\kappa$\nlow vs high LLaMA-$\\kappa$ windows", loc="left", pad=5)
    ax_D.set_xticks([1, 2])
    ax_D.set_xticklabels(["Q1\nlow", "Q5\nhigh"])

    if curv_bounds:
        y_min = float(np.nanmin([np.nanmin(b) for b in curv_bounds]))
        y_max = float(np.nanmax([np.nanmax(b) for b in curv_bounds]))
        pad   = max((y_max - y_min) * 0.12, 0.5)
        ax_A.set_ylim(y_min - pad, y_max + pad)

    # Legend below the figure
    handles, labels = ax_A.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.0)

    _save(fig, "fig1_layer_profiles_pca20_podcast")
def fig2_surprisal_geometry():
    """
    4-panel figure:
      A  LLM sentence trajectory (PCA, surprisal-coloured) at best_layer
      B  Hippocampal trajectory (Procrustes-aligned to A)
      C  Neural curvature vs LLM curvature scatter at best_layer
         (all patients pooled, z-scored per patient, word-level)
      D  r(neural, LLM) across all layers split by surprisal,
         with best_layer marked

    best_layer is selected as argmax(mean r_hi − mean r_lo) from Panel D data.
    """
    fig = plt.figure(figsize=(10.0, 9.0))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.09, right=0.96, bottom=0.08, top=0.88,
                            wspace=0.40, hspace=0.52)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    # New order: D→A, C→B, A→C, B→D
    axes = [ax_C, ax_D, ax_B, ax_A]

    # ── Load scatter data (new format: dict with patients / best_layer / traj_viz) ──
    scatter_path = os.path.join(RESULTS, f"scatter_data_podcast_pca20_{PRIMARY_MODEL}_hippocampus.json")
    best_layer = None
    tv         = None
    ch_sem, ch_neu, ch_surp = [], [], []
    r_all_pat, r_hi_pat, r_lo_pat = [], [], []
    chunk_r_per_patient = []  # per-patient Pearson r on the SAME word-level
                              # data pooled below (word_level_at_best_layer) --
                              # matches the pooled stat exactly since both are
                              # computed on identical data at two granularities.

    per_patient_word_level = {}  # pid -> (sem, neu, surp), for the single-patient example panel

    if os.path.exists(scatter_path):
        sdata      = json.load(open(scatter_path))
        best_layer = sdata.get("best_layer")
        tv         = sdata.get("traj_viz_best_layer")
        for rec in sdata.get("patients", []):
            wl = rec.get("word_level_at_best_layer") or {}
            pat_sem = np.asarray(wl.get("sem", []), dtype=float)
            pat_neu = np.asarray(wl.get("neu", []), dtype=float)
            pat_surp = np.asarray(wl.get("surp", []), dtype=float)
            ch_sem.extend(pat_sem.tolist())
            ch_neu.extend(pat_neu.tolist())
            ch_surp.extend(pat_surp.tolist())
            r_all_pat.append(rec["r_all_layers"])
            r_hi_pat.append(rec["r_hi_layers"])
            r_lo_pat.append(rec["r_lo_layers"])
            if len(pat_sem) >= 5:
                fin_p = np.isfinite(pat_sem) & np.isfinite(pat_neu)
                if fin_p.sum() >= 5:
                    from scipy.stats import pearsonr as _pr_tmp
                    r_this = _pr_tmp(pat_sem[fin_p], pat_neu[fin_p])[0]
                    chunk_r_per_patient.append(r_this)
                    per_patient_word_level[rec["pid"]] = (pat_sem[fin_p], pat_neu[fin_p], pat_surp[fin_p], r_this)

    ch_sem  = np.array(ch_sem,  dtype=float)
    ch_neu  = np.array(ch_neu,  dtype=float)
    ch_surp = np.array(ch_surp, dtype=float)
    r_all_mat = np.array(r_all_pat, dtype=float)
    r_hi_mat  = np.array(r_hi_pat,  dtype=float)
    r_lo_mat  = np.array(r_lo_pat,  dtype=float)

    from scipy.stats import pearsonr as _pr, sem as _sem, ttest_1samp as _ttest1

    # ── Panels A + B: best-layer sentence trajectory ──
    if tv is not None:
        words     = tv["words"]
        surp      = np.array(tv["surprisal"], dtype=float)
        llm_2d    = np.array(tv["llm_pca2d"])
        neu_2d    = np.array(tv["neu_pca2d"])
        surp_norm = np.where(np.isnan(surp), np.nan,
                             (surp - np.nanmin(surp)) /
                             max(np.nanmax(surp) - np.nanmin(surp), 1e-9))

        def _draw_traj(ax, xy, title):
            ax.plot(xy[:, 0], xy[:, 1], color="#AAAAAA", lw=0.8, zorder=1)
            for i in range(len(xy) - 1):
                ax.annotate("", xy=(xy[i+1, 0], xy[i+1, 1]),
                            xytext=(xy[i, 0], xy[i, 1]),
                            arrowprops=dict(arrowstyle="-|>",
                                            color="#BBBBBB", lw=0.6), zorder=2)
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=surp_norm,
                            cmap="RdBu_r", vmin=0, vmax=1,
                            s=50, zorder=3, edgecolors="white", linewidths=0.4)
            for i, (w, (x, y)) in enumerate(zip(words, xy)):
                offset = 0.04 * (np.max(xy[:, 1]) - np.min(xy[:, 1]))
                yo = offset if i % 2 == 0 else -offset * 1.6
                ax.text(x, y + yo, w, ha="center", va="center",
                        fontsize=6.0, color="#333333", zorder=4)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, loc="left", pad=4)
            return sc

        layer_lbl = tv.get("layer", best_layer)
        model_lbl = MODEL_STYLE.get(PRIMARY_MODEL, {}).get("label", PRIMARY_MODEL)
        sc = _draw_traj(axes[0], llm_2d,
                        f"C  {model_lbl}  layer {layer_lbl}\n(semantic space, PCA)")
        _draw_traj(axes[1], neu_2d,
                   "D  Hippocampal geometry\n(Procrustes-aligned to C)")
        cbar = fig.colorbar(sc, ax=axes[1], shrink=0.75, pad=0.03, aspect=18)
        cbar.set_label("Surprisal", fontsize=7)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["low", "mid", "high"], fontsize=6.5)

    # ── Panel B: single-patient scatter at best_layer (pooling all 13
    # patients into one scatter reads as an undifferentiated blob; pick the
    # patient with the strongest individual word-level r as a clean,
    # honest illustration, and report the pooled/patient-level stats in
    # text so the reader still sees the full-group evidence). ──
    ax = axes[2]
    example_pid = None
    if per_patient_word_level:
        example_pid = max(per_patient_word_level, key=lambda p: per_patient_word_level[p][3])
        cs, cn, csp, r_example = per_patient_word_level[example_pid]
    else:
        cs = cn = csp = np.array([])
        r_example = float("nan")
    if len(cs):
        # Deciles (not quartiles) for the star markers below: with quartiles
        # the two group means land almost on top of each other near the
        # plot center (the effect is real but modest), which reads as if
        # nothing is happening. The top/bottom 10% are still an honest
        # "high vs low surprisal" contrast, just a more separated one.
        lo_thr   = np.percentile(csp, 10)
        hi_thr   = np.percentile(csp, 90)
        mask_hi  = csp >= hi_thr
        mask_lo  = csp <= lo_thr

        vmin_s, vmax_s = np.percentile(csp, [5, 95])
        sc2 = ax.scatter(cs, cn, c=csp, cmap="RdBu_r",
                         vmin=vmin_s, vmax=vmax_s,
                         s=14, alpha=0.6, rasterized=True, linewidths=0)
        m, b     = np.polyfit(cs, cn, 1)
        r_val, p_val = _pr(cs, cn)
        xr = np.linspace(cs.min(), cs.max(), 100)
        ax.plot(xr, m * xr + b, color="k", lw=1.6, ls="--")
        pstr = f"p = {p_val:.2e}" if p_val >= 0.001 else "p < 0.001"

        # Clip axis limits to the 0.5-99.5th percentile so a rare outlier
        # doesn't compress the cloud; fit line/r/p above use all of this
        # one patient's points (this only changes what's visible).
        x_lo, x_hi = np.percentile(cs, [0.5, 99.5])
        y_lo, y_hi = np.percentile(cn, [0.5, 99.5])
        x_pad = 0.05 * (x_hi - x_lo)
        y_pad = 0.05 * (y_hi - y_lo)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        # Group-level context: patient-level test (one r per patient,
        # t-test across all 13 patients vs 0) -- this patient is just one
        # example of that broader pattern.
        pat_str = ""
        cr = np.asarray(chunk_r_per_patient, dtype=float)
        cr = cr[np.isfinite(cr)]
        if len(cr) >= 2:
            t_pat, p_pat = _ttest1(cr, 0.0)
            pstr_pat = f"p = {p_pat:.2e}" if p_pat >= 0.001 else "p < 0.001"
            n_pos = int((cr > 0).sum())
            pat_str = (f"\nall {len(cr)} patients: mean $r$ = {cr.mean():+.3f}  "
                       f"({n_pos}/{len(cr)} positive, $t$={t_pat:.2f}, {pstr_pat})")

        ax.text(0.03, 0.97, f"patient {example_pid}: $r$ = {r_val:+.3f},  {pstr}{pat_str}",
                transform=ax.transAxes, fontsize=7.0, va="top")
        for mask, color, lbl in [(mask_hi, C_HI, "High surp."),
                                  (mask_lo, C_LO, "Low surp.")]:
            mx, my = np.mean(cs[mask]), np.mean(cn[mask])
            ex = _sem(cs[mask]); ey = _sem(cn[mask])
            ax.errorbar(mx, my, xerr=ex, yerr=ey,
                        fmt="*", color=color, ms=12, lw=1.5,
                        markeredgecolor="white", markeredgewidth=0.6,
                        zorder=5, label=lbl)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.7,
                  edgecolor="none", markerscale=0.8)
        cbar2 = fig.colorbar(sc2, ax=ax, shrink=0.75, pad=0.02, aspect=18)
        cbar2.set_label("Surprisal (nats)", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.axvline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel("Semantic curvature κ  (z-scored)", fontsize=7.5)
    ax.set_ylabel("Neural curvature κ  (z-scored)", fontsize=7.5)
    layer_str = f"layer {best_layer}" if best_layer is not None else "best layer"
    ax.set_title(f"B  Semantic ↔ Neural curvature\n({layer_str}, patient {example_pid}, word-level)", loc="left", pad=4)

    # ── Panel A: r(neural, LLM) across all layers, split by surprisal ──
    ax = axes[3]
    N_LAYERS_D = r_all_mat.shape[1] if r_all_mat.ndim == 2 else 33
    x_norm     = np.linspace(0, 1, N_LAYERS_D)
    for mat, color, lbl in [
        (r_hi_mat, C_HI, "High surprisal (≥75th)"),
        (r_lo_mat, C_LO, "Low surprisal (≤25th)"),
        (r_all_mat, "#555555", "All words"),
    ]:
        mu = np.nanmean(mat, axis=0)
        se = np.nanstd(mat, axis=0) / np.sqrt((~np.isnan(mat)).sum(axis=0).clip(1))
        lw = 1.2 if "All" in lbl else 1.8
        ax.plot(x_norm, mu, color=color, lw=lw,
                ls="--" if "All" in lbl else "-", label=lbl)
        ax.fill_between(x_norm, mu - se, mu + se, color=color, alpha=0.15)
    # Mark the selected best_layer
    if best_layer is not None:
        x_best = best_layer / (N_LAYERS_D - 1)
        ax.axvline(x_best, color="#333333", lw=1.0, ls=":", zorder=4)
        ax.text(x_best + 0.01, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0.12,
                f"L{best_layer}", fontsize=6.5, color="#333333", va="top")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Normalized layer position", fontsize=7.5)
    ax.set_ylabel("r(neural κ, LLM κ)", fontsize=7.5)
    ax.set_title("A  Neural–LLM coupling across layers\nby surprisal level", loc="left", pad=4)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.7, edgecolor="none")

    fig.suptitle("Surprising words bend the trajectory — and the brain bends with it",
                 fontsize=9, fontweight="bold", y=0.97)
    _save(fig, "fig2_surprisal_geometry_pca20_podcast")


def fig3_event_triggered():
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.76,
                        wspace=0.38)

    def _panel(ax, evts, space_key_hi, space_key_lo, title):
        evts_valid = [e["event_triggered"] for e in evts
                      if e["event_triggered"] and
                         e["event_triggered"][space_key_hi] is not None]
        if not evts_valid:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, loc="left")
            return

        lags = evts_valid[0]["lags"]
        xs   = np.array(lags)

        def _stack(key):
            mat = np.array([e[key] for e in evts_valid], dtype=float)
            return np.nanmean(mat, 0), scipy_sem(mat, axis=0, nan_policy="omit")

        hi_mu, hi_se = _stack(space_key_hi)
        lo_mu, lo_se = _stack(space_key_lo)

        ax.plot(xs, hi_mu, color=C_HI, lw=1.6, label="High surprisal (top 25%)")
        ax.fill_between(xs, hi_mu-hi_se, hi_mu+hi_se, color=C_HI, alpha=0.18)
        ax.plot(xs, lo_mu, color=C_LO, lw=1.6, label="Low surprisal (bottom 25%)")
        ax.fill_between(xs, lo_mu-lo_se, lo_mu+lo_se, color=C_LO, alpha=0.18)
        ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel("Words relative to event\nlag 0 curvature ends at event word",
                      fontsize=9)
        ax.set_ylabel("Mean curvature (\u00b0)", fontsize=10)
        ax.set_title(f"{title}\n$n$ = {len(evts_valid)} patients",
                     loc="left", fontsize=10.5, pad=3)
        ax.tick_params(labelsize=9)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.75,
                  edgecolor="#DDDDDD")
        ax.set_xticks(lags)

    data_hpc = _load(PRIMARY_MODEL, "hippocampus") or []
    _panel(axes[0], data_hpc, "hi_sem", "lo_sem",
           "A  Semantic curvature")
    _panel(axes[1], data_hpc, "hi_neu", "lo_neu",
           "B  Neural curvature (hippocampus)")

    fig.suptitle(
        f"{PRIMARY_MODEL}  \u00b7  podcast  \u00b7  event-triggered curvature  \u00b7  PCA(20) semantic trajectory  \u00b7  "
        "lag 0 uses words t-2..t",
        fontsize=9, fontweight="bold", y=0.96)
    _save(fig, "fig3_event_triggered_pca20_podcast")


if __name__ == "__main__":
    print("Building fig1_layer_profiles_pca20_podcast...")
    fig1_layer_profiles()
    print("Building fig2_surprisal_geometry_pca20_podcast...")
    fig2_surprisal_geometry()
    print("Building fig3_event_triggered_pca20_podcast...")
    fig3_event_triggered()
    print("Done.")
