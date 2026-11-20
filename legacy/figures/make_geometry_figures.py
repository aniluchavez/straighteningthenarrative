"""
make_geometry_figures.py
========================
NeurIPS-quality figures from geometry_{model}_{region}.json files.

Figures:
  fig1_layer_profiles.pdf     — mean curvature / surprisal alignment / hippocampal
                                 alignment by layer for primary model
  fig2_model_comparison.pdf   — r(c_layer, surprisal) by *normalised* layer
                                 position across all models
  fig3_event_triggered.pdf    — hippocampal curvature dynamics around surprising words
  fig4_regression.pdf         — regression summary: per-patient bars +
                                 direction agreement + R² comparison
  fig5_trajectory_viz.pdf     — highest-range sentence trajectory example
  fig5_trajectory_viz_example2.pdf
                               — second highest-range sentence trajectory example
  fig5_trajectory_viz_example3.pdf
                               — third highest-range sentence trajectory example
  fig5_trajectory_viz_short_sentence.pdf
                               — alternate short sentence trajectory example

Run:
    conda run -n gpt2_embed python3 make_geometry_figures.py
"""

import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem as scipy_sem, ttest_rel

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RESULTS       = os.path.join(BASE_DIR, "results")
FIGURES       = os.path.join(BASE_DIR, "figures")
PRIMARY_MODEL = "llama-3.1-8b"
REGIONS       = ["hippocampus"]

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
    "opt-350m":     dict(color="#884EA0", label="OPT-350m",      ls=(0,(3,1))),
    "llama-2-7b":   dict(color="#C0392B", label="LLaMA-2 7B",    ls="--"),
    "llama-3.1-8b": dict(color="#E91E63", label="LLaMA-3.1 8B",  ls="-"),
    "gemma-2-9b":   dict(color="#148F77", label="Gemma-2 9B",    ls=(0,(4,1))),
    "mistral-7b":   dict(color="#D35400", label="Mistral-7B",    ls=(0,(2,1))),
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
    path = os.path.join(RESULTS, f"geometry_{model}_{region}.json")
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
    from run_geometry_analysis import (
        PATIENTS, build_valid_mask, fit_neural_pca, load_patient,
        trajectory_curvature,
    )

    records = _load(model, region) or []
    neural_vals, llm_vals = [], []
    layers = []
    for rec in records:
        pid = rec["pid"]
        preg = PATIENTS.get(pid, {})
        if region not in preg:
            continue
        data = load_patient(pid, preg, region, model)
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

        mu_n = np.nanmean(np.array([r["profile"]["r_neural"] for r in data_m], dtype=float), axis=0)
        ax_B.plot(GRID, np.interp(GRID, xs_m, np.nan_to_num(mu_n)),
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

        mu_n, se_n, _, _ = _layer_arrays(data, "r_neural", "p_neural")
        if region == "hippocampus":
            ax_B.plot(xs_norm, mu_n, color=llama_st["color"], ls=llama_st["ls"],
                      lw=0.9, label="_nolegend_")

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
    ax_A.set_xlabel("Normalised layer position")
    ax_A.set_ylabel("Mean curvature (°)")
    ax_A.set_title("$\\mathbf{A}$  Mean curvature · all models", loc="left", pad=5)
    ax_A.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_A.set_xticklabels(["0", ".25", ".5", ".75", "1"])

    # Panel B
    ax_B.axhline(0, color="k", lw=0.5, ls=":")
    ax_B.set_xlabel("Normalised layer position")
    ax_B.set_ylabel("Pearson $r$")
    ax_B.set_title("$\\mathbf{B}$  Neural-LLM alignment · all models", loc="left", pad=5)
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

    _save(fig, "fig1_layer_profiles")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Surprisal geometry: trajectory viz + surprisal alignment + event
# ══════════════════════════════════════════════════════════════════════════════
def fig2_surprisal_geometry():
    """
    4-panel figure:
      A  LLM sentence trajectory (PCA, surprisal-coloured) at best_layer
      B  Hippocampal trajectory (Procrustes-aligned to A)
      C  Neural curvature vs LLM curvature scatter at best_layer
         (all patients pooled, z-scored, 10-word chunks)
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
    scatter_path = os.path.join(RESULTS, f"scatter_data_{PRIMARY_MODEL}_hippocampus.json")
    best_layer = None
    tv         = None
    ch_sem, ch_neu, ch_surp = [], [], []
    r_all_pat, r_hi_pat, r_lo_pat = [], [], []

    if os.path.exists(scatter_path):
        sdata      = json.load(open(scatter_path))
        best_layer = sdata.get("best_layer")
        tv         = sdata.get("traj_viz_best_layer")
        lkey       = str(best_layer) if best_layer is not None else None
        for rec in sdata.get("patients", []):
            chunks = rec.get("all_layer_chunks", {})
            lchunk = chunks.get(lkey, {}) if lkey else {}
            ch_sem.extend(lchunk.get("sem", []))
            ch_neu.extend(lchunk.get("neu", []))
            ch_surp.extend(lchunk.get("surp", []))
            r_all_pat.append(rec["r_all_layers"])
            r_hi_pat.append(rec["r_hi_layers"])
            r_lo_pat.append(rec["r_lo_layers"])

    ch_sem  = np.array(ch_sem,  dtype=float)
    ch_neu  = np.array(ch_neu,  dtype=float)
    ch_surp = np.array(ch_surp, dtype=float)
    r_all_mat = np.array(r_all_pat, dtype=float)
    r_hi_mat  = np.array(r_hi_pat,  dtype=float)
    r_lo_mat  = np.array(r_lo_pat,  dtype=float)

    from scipy.stats import pearsonr as _pr, sem as _sem

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
        sc = _draw_traj(axes[0], llm_2d,
                        f"C  LLaMA-3.1-8B  layer {layer_lbl}\n(semantic space, PCA)")
        _draw_traj(axes[1], neu_2d,
                   "D  Hippocampal geometry\n(Procrustes-aligned to C)")
        cbar = fig.colorbar(sc, ax=axes[1], shrink=0.75, pad=0.03, aspect=18)
        cbar.set_label("Surprisal", fontsize=7)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["low", "mid", "high"], fontsize=6.5)

    # ── Panel B: scatter at best_layer coloured by surprisal + high/low centroids ──
    ax = axes[2]
    fin_c = np.isfinite(ch_sem) & np.isfinite(ch_neu) & np.isfinite(ch_surp)
    cs, cn, csp = ch_sem[fin_c], ch_neu[fin_c], ch_surp[fin_c]
    if len(cs):
        lo_thr   = np.percentile(csp, 25)
        hi_thr   = np.percentile(csp, 75)
        mask_hi  = csp >= hi_thr
        mask_lo  = csp <= lo_thr

        vmin_s, vmax_s = np.percentile(csp, [5, 95])
        sc2 = ax.scatter(cs, cn, c=csp, cmap="RdBu_r",
                         vmin=vmin_s, vmax=vmax_s,
                         s=10, alpha=0.4, rasterized=True, linewidths=0)
        m, b     = np.polyfit(cs, cn, 1)
        r_val, p_val = _pr(cs, cn)
        xr = np.linspace(cs.min(), cs.max(), 100)
        ax.plot(xr, m * xr + b, color="k", lw=1.6, ls="--")
        pstr = f"p = {p_val:.2e}" if p_val >= 0.001 else "p < 0.001"
        ax.text(0.03, 0.97, f"r = {r_val:+.3f},  {pstr}",
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
        cbar2.set_label("Mean chunk surprisal (nats)", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.axvline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel("Semantic curvature κ  (z-scored)", fontsize=7.5)
    ax.set_ylabel("Neural curvature κ  (z-scored)", fontsize=7.5)
    layer_str = f"layer {best_layer}" if best_layer is not None else "best layer"
    ax.set_title(f"B  Semantic ↔ Neural curvature\n({layer_str}, 10-word chunks)", loc="left", pad=4)

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
    ax.set_xlabel("Normalised layer position", fontsize=7.5)
    ax.set_ylabel("r(neural κ, LLM κ)", fontsize=7.5)
    ax.set_title("A  Neural–LLM coupling across layers\nby surprisal level", loc="left", pad=4)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.7, edgecolor="none")

    fig.suptitle("Surprising words bend the trajectory — and the brain bends with it",
                 fontsize=9, fontweight="bold", y=0.97)
    _save(fig, "fig2_surprisal_geometry")


def fig2_surprisal_displacement():
    """
    Companion to Fig. 2 using displacement instead of curvature.

    Displacement is aligned to surprisal at word t as the step arriving at t:
    ||x_t - x_{t-1}||. This mirrors the no-future indexing used for curvature.
    """
    fig = plt.figure(figsize=(10.0, 9.0))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.09, right=0.96, bottom=0.08, top=0.88,
                            wspace=0.40, hspace=0.52)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    axes = [ax_A, ax_B, ax_C, ax_D]

    scatter_path = os.path.join(RESULTS, f"scatter_data_{PRIMARY_MODEL}_hippocampus.json")
    best_layer = None
    tv         = None
    ch_sem, ch_neu, ch_surp = [], [], []
    r_all_pat, r_hi_pat, r_lo_pat = [], [], []

    if os.path.exists(scatter_path):
        sdata      = json.load(open(scatter_path))
        best_layer = sdata.get("best_disp_layer", sdata.get("best_layer"))
        tv         = sdata.get("traj_viz_best_disp_layer") or sdata.get("traj_viz_best_layer")
        lkey       = str(best_layer) if best_layer is not None else None
        for rec in sdata.get("patients", []):
            chunks = rec.get("all_layer_disp_chunks", {})
            lchunk = chunks.get(lkey, {}) if lkey else {}
            ch_sem.extend(lchunk.get("sem", []))
            ch_neu.extend(lchunk.get("neu", []))
            ch_surp.extend(lchunk.get("surp", []))
            r_all_pat.append(rec.get("r_disp_all_layers", []))
            r_hi_pat.append(rec.get("r_disp_hi_layers", []))
            r_lo_pat.append(rec.get("r_disp_lo_layers", []))

    ch_sem  = np.array(ch_sem,  dtype=float)
    ch_neu  = np.array(ch_neu,  dtype=float)
    ch_surp = np.array(ch_surp, dtype=float)
    r_all_mat = np.array(r_all_pat, dtype=float)
    r_hi_mat  = np.array(r_hi_pat,  dtype=float)
    r_lo_mat  = np.array(r_lo_pat,  dtype=float)

    from scipy.stats import pearsonr as _pr, sem as _sem

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
        sc = _draw_traj(axes[0], llm_2d,
                        f"A  LLaMA-3.1-8B  layer {layer_lbl}\n(semantic space, PCA)")
        _draw_traj(axes[1], neu_2d,
                   "B  Hippocampal geometry\n(Procrustes-aligned to A)")
        cbar = fig.colorbar(sc, ax=axes[1], shrink=0.75, pad=0.03, aspect=18)
        cbar.set_label("Surprisal", fontsize=7)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["low", "mid", "high"], fontsize=6.5)

    ax = axes[2]
    fin_c = np.isfinite(ch_sem) & np.isfinite(ch_neu) & np.isfinite(ch_surp)
    cs, cn, csp = ch_sem[fin_c], ch_neu[fin_c], ch_surp[fin_c]
    if len(cs):
        lo_thr   = np.percentile(csp, 25)
        hi_thr   = np.percentile(csp, 75)
        mask_hi  = csp >= hi_thr
        mask_lo  = csp <= lo_thr

        vmin_s, vmax_s = np.percentile(csp, [5, 95])
        sc2 = ax.scatter(cs, cn, c=csp, cmap="RdBu_r",
                         vmin=vmin_s, vmax=vmax_s,
                         s=10, alpha=0.4, rasterized=True, linewidths=0)
        m, b     = np.polyfit(cs, cn, 1)
        r_val, p_val = _pr(cs, cn)
        xr = np.linspace(cs.min(), cs.max(), 100)
        ax.plot(xr, m * xr + b, color="k", lw=1.6, ls="--")
        pstr = f"p = {p_val:.2e}" if p_val >= 0.001 else "p < 0.001"
        ax.text(0.03, 0.97, f"r = {r_val:+.3f},  {pstr}",
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
        cbar2.set_label("Mean chunk surprisal (nats)", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.axvline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel("Semantic displacement $d$  (z-scored)", fontsize=7.5)
    ax.set_ylabel("Neural displacement $d$  (z-scored)", fontsize=7.5)
    layer_str = f"layer {best_layer}" if best_layer is not None else "best layer"
    ax.set_title(f"C  Semantic ↔ Neural displacement\n({layer_str}, 10-word chunks)",
                 loc="left", pad=4)

    ax = axes[3]
    N_LAYERS_D = r_all_mat.shape[1] if r_all_mat.ndim == 2 else 33
    x_norm     = np.linspace(0, 1, N_LAYERS_D)
    for mat, color, lbl in [
        (r_hi_mat, C_HI, "High surprisal (≥75th)"),
        (r_lo_mat, C_LO, "Low surprisal (≤25th)"),
        (r_all_mat, "#555555", "All words"),
    ]:
        if mat.ndim != 2 or mat.shape[1] == 0:
            continue
        mu = np.nanmean(mat, axis=0)
        se = np.nanstd(mat, axis=0) / np.sqrt((~np.isnan(mat)).sum(axis=0).clip(1))
        lw = 1.2 if "All" in lbl else 1.8
        ax.plot(x_norm, mu, color=color, lw=lw,
                ls="--" if "All" in lbl else "-", label=lbl)
        ax.fill_between(x_norm, mu - se, mu + se, color=color, alpha=0.15)
    if best_layer is not None:
        x_best = best_layer / (N_LAYERS_D - 1)
        ax.axvline(x_best, color="#333333", lw=1.0, ls=":", zorder=4)
        ax.text(x_best + 0.01, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0.12,
                f"L{best_layer}", fontsize=6.5, color="#333333", va="top")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Normalised layer position", fontsize=7.5)
    ax.set_ylabel("r(neural $d$, LLM $d$)", fontsize=7.5)
    ax.set_title("D  Neural–LLM displacement coupling\nby surprisal level",
                 loc="left", pad=4)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.7, edgecolor="none")

    fig.suptitle("Surprising words move the trajectory — and the brain moves with it",
                 fontsize=9, fontweight="bold", y=0.97)
    _save(fig, "fig2_surprisal_displacement")


def fig2_model_comparison():
    """
    r(c_layer, surprisal[t+1]) vs normalised layer position [0,1],
    for each model. Hippocampus only (primary region for surprisal prediction).
    """
    GRID  = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.95))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.76,
                        wspace=0.42)

    # Panel A: r(c_layer, surprisal) — all models
    ax = axes[0]
    for model, st in MODEL_STYLE.items():
        data = _load(model, "hippocampus")
        if data is None:
            continue
        n_layers = len(data[0]["profile"]["r_surp"])
        xs_norm  = np.linspace(0, 1, n_layers)
        mat = np.array([r["profile"]["r_surp"] for r in data])
        mu  = mat.mean(0)
        # interpolate to common grid
        mu_interp = np.interp(GRID, xs_norm, mu)
        ax.plot(GRID, mu_interp, color=st["color"], ls=st["ls"],
                lw=1.4, label=st["label"])

    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Normalised layer position")
    ax.set_ylabel("Pearson $r$")
    ax.set_title("A  $r$(curvature, surprisal$_{t+1}$)\nall models, hippocampus",
                 loc="left")
    ax.legend(fontsize=6.5, ncol=1, loc="lower right",
              framealpha=0.6, edgecolor="none")

    # Panel B: r(c_layer, c_neural) — all models, hippocampus
    ax = axes[1]
    for model, st in MODEL_STYLE.items():
        data = _load(model, "hippocampus")
        if data is None:
            continue
        n_layers = len(data[0]["profile"]["r_neural"])
        xs_norm  = np.linspace(0, 1, n_layers)
        mat = np.array([r["profile"]["r_neural"] for r in data], dtype=float)
        mu  = np.nanmean(mat, axis=0)
        if np.all(np.isnan(mu)):
            continue
        mu_interp = np.interp(GRID, xs_norm, np.nan_to_num(mu))
        ax.plot(GRID, mu_interp, color=st["color"], ls=st["ls"],
                lw=1.4, label=st["label"])

    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Normalised layer position")
    ax.set_ylabel("Pearson $r$")
    ax.set_title("B  $r$($c_{sem}$, $c_{neural}$)\nall models — neural-semantic alignment",
                 loc="left")
    ax.legend(fontsize=6.5, ncol=1, loc="best",
              framealpha=0.6, edgecolor="none")

    fig.suptitle("Model comparison  ·  hippocampus",
                 fontsize=9, fontweight="bold", y=0.96)
    _save(fig, "fig2_model_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Event-triggered curvature (hippocampus only)
# ══════════════════════════════════════════════════════════════════════════════
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
        ax.set_ylabel("Mean curvature (°)", fontsize=10)
        ax.set_title(f"{title}\n$n$ = {len(evts_valid)} patients",
                     loc="left", fontsize=10.5, pad=3)
        ax.tick_params(labelsize=9)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.75,
                  edgecolor="#DDDDDD")
        ax.set_xticks(lags)

    # Semantic: use hippocampus data (has largest n)
    data_hpc = _load(PRIMARY_MODEL, "hippocampus") or []
    _panel(axes[0], data_hpc, "hi_sem", "lo_sem",
           "A  Semantic curvature")
    _panel(axes[1], data_hpc, "hi_neu", "lo_neu",
           "B  Neural curvature (hippocampus)")

    fig.suptitle(
        f"{PRIMARY_MODEL}  ·  event-triggered curvature  ·  "
        "lag 0 uses words t-2..t",
        fontsize=9, fontweight="bold", y=0.96)
    _save(fig, "fig3_event_triggered")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Regression summary (hippocampus only)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_regression():
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 3.1))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.76,
                        wspace=0.50)

    # ── Panel A: per-patient r_sem, hippocampus ────────────────────────────
    ax = axes[0]
    y_offset = 0
    yticks, ylabels = [], []

    for region in REGIONS:
        data = _load(PRIMARY_MODEL, region)
        if data is None:
            continue
        st   = REGION_STYLE[region]
        regs = [r for r in data if r["regression"]]
        for ri, r in enumerate(regs):
            rv = r["regression"]["r_sem"]
            pv = r["regression"]["p_sem"]
            c  = st["color"] if pv < 0.05 else C_NS
            ax.barh(y_offset, rv, color=c, height=0.6, alpha=0.85)
            lyr  = r["loo_layer_neu"]
            yticks.append(y_offset)
            ylabels.append(f"{r['pid'].replace('PT','').replace('_',' ')} [L{lyr}]")
            y_offset += 1

    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_xlabel("Pearson $r$")
    ax.set_title("A  $r$($c_{sem}$, surprisal$_{t+1}$)\nper patient · LOO-neural layer",
                 loc="left")

    # ── Panel B: direction agreement ───────────────────────────────────────
    ax = axes[1]
    x_pos = np.arange(len(REGIONS))
    for xi, region in enumerate(REGIONS):
        data = _load(PRIMARY_MODEL, region)
        if data is None:
            continue
        regs = [r["regression"] for r in data if r["regression"]]
        n_same = sum(1 for r in regs if r["same_dir"])
        n_opp  = sum(1 for r in regs if r["same_dir"] is False)
        st = REGION_STYLE[region]
        ax.bar(xi - 0.18, n_same, 0.32, color=st["color"],  alpha=0.85,
               label="Same direction")
        ax.bar(xi + 0.18, n_opp,  0.32, color=st["color"],  alpha=0.35,
               hatch="///", label="Opposite")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([REGION_STYLE[r]["label"] for r in REGIONS])
    ax.set_ylabel("# patients")
    ax.set_title("B  $c_{sem}$ vs $c_{neural}$\ndirectional agreement",
                 loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="gray", alpha=0.85, label="Same dir."),
                       Patch(color="gray", alpha=0.35, hatch="///",
                             label="Opp. dir.")],
              fontsize=7, loc="upper right")

    # ── Panel C: R² curvature vs displacement ─────────────────────────────
    ax   = axes[2]
    bw   = 0.25
    keys = ["r2_curv", "r2_disp"]
    clrs = [C_SEM, C_DISP]
    labs = ["Curvature", "Displacement"]

    for xi, region in enumerate(REGIONS):
        data = _load(PRIMARY_MODEL, region)
        if data is None:
            continue
        regs = [r["regression"] for r in data if r["regression"]]
        st   = REGION_STYLE[region]
        for ki, (key, clr, lab) in enumerate(zip(keys, clrs, labs)):
            mu, se = _mu_se([r[key] for r in regs])
            xpos = xi + (ki - 0.5) * bw
            bar  = ax.bar(xpos, mu, bw, yerr=se, color=clr, alpha=0.85,
                          capsize=3,
                          label=f"{lab}" if xi == 0 else "_nolegend_")
            if not np.isnan(mu):
                ax.text(xpos, mu + se + 0.0003, f"{mu:.3f}",
                        ha="center", va="bottom", fontsize=6)

    ax.set_xticks(np.arange(len(REGIONS)))
    ax.set_xticklabels([REGION_STYLE[r]["label"] for r in REGIONS])
    ax.set_ylabel("$R^2$ (mean ± SEM)")
    ax.set_ylim(bottom=0)
    ax.set_title("C  Curvature vs displacement\n$R^2$ predicting surprisal$_{t+1}$",
                 loc="left")
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"{PRIMARY_MODEL}  ·  regression summary",
                 fontsize=9, fontweight="bold", y=0.96)
    _save(fig, "fig4_regression")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Sentence trajectory visualisation: LLM vs neural space
# ══════════════════════════════════════════════════════════════════════════════
def fig5_trajectory_viz(rank=1, out_name="fig5_trajectory_viz",
                        title_prefix="Sentence trajectory",
                        selection="surprisal_range"):
    """
    Select a patient × sentence trajectory from the primary model / hippocampus
    results, then plot the Procrustes-aligned 2-D PCA trajectories side-by-side
    with words coloured by surprisal.
    """
    data = _load(PRIMARY_MODEL, "hippocampus")
    if data is None:
        return

    if selection == "surprisal_range":
        candidates = sorted(
            (r for r in data if r.get("traj_viz")),
            key=lambda r: r["traj_viz"]["surp_range"],
            reverse=True,
        )
    elif selection == "word_count":
        candidates = sorted(
            (r for r in data if r.get("traj_viz")),
            key=lambda r: (len(r["traj_viz"]["words"]),
                           -r["traj_viz"]["surp_range"]),
        )
    else:
        raise ValueError(f"Unknown fig5 selection: {selection}")
    rank = int(rank)
    if rank < 1:
        raise ValueError("rank must be 1-indexed")
    if len(candidates) < rank:
        print(f"  Only {len(candidates)} traj_viz records found — skipping {out_name}.")
        return
    best_rec = candidates[rank - 1]

    tv      = best_rec["traj_viz"]
    words   = tv["words"]
    surp    = np.array(tv["surprisal"], dtype=float)
    llm_2d  = np.array(tv["llm_pca2d"])
    neu_2d  = np.array(tv["neu_pca2d"])
    layer   = tv["layer"]
    pid     = best_rec["pid"]

    # colour map: surprisal → blue (low) to red (high), grey for NaN
    surp_norm = np.where(np.isnan(surp), np.nan,
                         (surp - np.nanmin(surp)) /
                         max(np.nanmax(surp) - np.nanmin(surp), 1e-9))
    cmap   = plt.cm.RdBu_r
    colors = [cmap(v) if not np.isnan(v) else (0.6, 0.6, 0.6, 1.) for v in surp_norm]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.35))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.78,
                        wspace=0.42)

    def _draw_traj(ax, xy, title):
        # trajectory line
        ax.plot(xy[:, 0], xy[:, 1], color="#AAAAAA", lw=0.8, zorder=1)
        # arrows between consecutive points
        for i in range(len(xy) - 1):
            dx, dy = xy[i+1, 0] - xy[i, 0], xy[i+1, 1] - xy[i, 1]
            ax.annotate("", xy=(xy[i+1, 0], xy[i+1, 1]),
                        xytext=(xy[i, 0], xy[i, 1]),
                        arrowprops=dict(arrowstyle="-|>",
                                        color="#BBBBBB", lw=0.6),
                        zorder=2)
        # scatter points coloured by surprisal
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=surp_norm,
                        cmap="RdBu_r", vmin=0, vmax=1,
                        s=55, zorder=3, edgecolors="white", linewidths=0.4)
        # word labels — stagger above/below to reduce overlap
        for i, (w, (x, y)) in enumerate(zip(words, xy)):
            offset = 0.04 * (np.max(xy[:, 1]) - np.min(xy[:, 1]))
            yo = offset if i % 2 == 0 else -offset * 1.6
            ax.text(x, y + yo, w, ha="center", va="center",
                    fontsize=6.5, color="#333333", zorder=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, loc="left")
        return sc

    sc = _draw_traj(axes[0], llm_2d,
                    f"A  LLaMA-3.1-8B  layer {layer}\n(semantic embedding space, PCA)")
    _draw_traj(axes[1], neu_2d,
               "B  Hippocampal population geometry\n(Procrustes-aligned to A)")

    # shared colour bar
    cbar = fig.colorbar(sc, ax=axes, shrink=0.72, pad=0.02, aspect=20)
    cbar.set_label("Surprisal (normalised)", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["low", "mid", "high"])

    fig.suptitle(f"{title_prefix} · {pid}  (surp range = {tv['surp_range']:.1f} bits)",
                 fontsize=9, fontweight="bold", y=0.97)
    _save(fig, out_name)


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX summary table — all models, hippocampus
# ══════════════════════════════════════════════════════════════════════════════
def fig6_geodesic_decomposition():
    """
    4-panel figure telling the off-manifold excursion story (LLaMA-3.1-8B focus).

    Panel A — Schematic.
    Panel B — On/off component correlations with next-word surprisal.
    Panel C — LLM component activation approaching surprisal.
    Panel D — Hpc component activation approaching surprisal, zoomed.
    """
    from scipy.stats import sem as scipy_sem, ttest_1samp

    FOCAL_MODEL = "llama-3.1-8b"

    def _patient_vals(d, key):
        return [r["regression"][key] for r in d
                if r.get("regression") and r["regression"].get(key) is not None
                and not np.isnan(r["regression"][key])]

    # ── load LLaMA patient records ─────────────────────────────────────────────
    d_focal = _load(FOCAL_MODEL, "hippocampus")
    if d_focal is None:
        d_focal = []

    fig = plt.figure(figsize=(10.0, 9.0))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.52)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ── Panel A: schematic ─────────────────────────────────────────────────────
    ax = axA
    ax.set_xlim(-0.2, 3.2); ax.set_ylim(-0.3, 2.3)
    ax.set_aspect("equal"); ax.axis("off")

    # draw manifold as a shaded band
    t = np.linspace(0, np.pi, 200)
    mx = t / np.pi * 3
    my = 0.4 * np.sin(t)
    band = 0.22
    ax.fill_between(mx, my - band, my + band,
                    color="#F5EAD4", alpha=0.55, zorder=1)
    ax.plot(mx, my, color="#8A6010", lw=1.2, alpha=0.6, zorder=2)
    ax.text(1.5, my[100] - band - 0.14, "Semantic manifold",
            ha="center", fontsize=7, color="#8A6010", style="italic")

    # trajectory with two segments: expected → surprising
    # expected segment: stays on manifold
    t0 = np.linspace(0.3, 1.1, 30)
    tx0 = t0 / np.pi * 3; ty0 = 0.4 * np.sin(t0) + 0.08
    # surprising word: kicks off manifold
    t1 = np.linspace(1.1, 1.7, 20)
    tx1 = t1 / np.pi * 3
    ty1_man = 0.4 * np.sin(t1) + 0.08
    kick = np.linspace(0, 0.55, 20)
    ty1 = ty1_man + kick
    # back segment
    t2 = np.linspace(1.7, 2.6, 30)
    tx2 = t2 / np.pi * 3; ty2 = 0.4 * np.sin(t2) + 0.08

    ax.plot(tx0, ty0, color="#555555", lw=1.8, zorder=3)
    ax.plot(tx1, ty1, color="#B5446E", lw=1.8, zorder=3, ls="--")
    ax.plot(tx2, ty2, color="#555555", lw=1.8, zorder=3)

    # dots at word positions
    for xi, yi, label, col in [
        (tx0[0],  ty0[0],  "word\n(expected)", "#C9922A"),
        (tx1[-1], ty1[-1], "word\n(surprising!)", "#B5446E"),
    ]:
        ax.scatter([xi], [yi], s=38, color=col, zorder=5)
        ax.text(xi, yi + 0.16, label, ha="center", fontsize=6.5,
                color=col, fontweight="bold")

    # κ_nrm arrow at surprising word
    xi_s = tx1[10]; yi_s = ty1[10]
    yi_man = ty1_man[10]
    ax.annotate("", xy=(xi_s, yi_s + 0.04), xytext=(xi_s, yi_man),
                arrowprops=dict(arrowstyle="-|>", color="#B5446E",
                                lw=1.3, mutation_scale=8))
    ax.text(xi_s + 0.12, (yi_s + yi_man) / 2,
            r"$\kappa_\mathrm{nrm}\uparrow$",
            fontsize=7, color="#B5446E", va="center")

    # κ_geo label on expected segment
    xi_e = tx0[15]; yi_e = ty0[15]
    ax.annotate("", xy=(tx0[22], ty0[22]),
                xytext=(xi_e, yi_e),
                arrowprops=dict(arrowstyle="-|>", color="#C9922A",
                                lw=1.3, mutation_scale=8,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(xi_e - 0.25, yi_e - 0.26,
            r"$\kappa_\mathrm{geo}$" "\n(on-manifold)",
            fontsize=6.5, color="#C9922A", ha="center")

    ax.set_title("A  Off-manifold bend\ninto surprising words",
                 loc="left", fontsize=8, fontweight="bold")

    # ── Panel B: on/off component correlations vs surprisal ───────────────────
    ax = axB
    llm_geo_arr = np.array(_patient_vals(d_focal, "r_sem_geo"))
    llm_nrm_arr = np.array(_patient_vals(d_focal, "r_sem_nrm"))
    hpc_geo_arr = np.array(_patient_vals(d_focal, "r_neu_geo"))
    hpc_nrm_arr = np.array(_patient_vals(d_focal, "r_neu_nrm"))

    bar_data  = [llm_geo_arr, hpc_geo_arr, llm_nrm_arr, hpc_nrm_arr]
    bar_means = [v.mean() for v in bar_data]
    bar_sems  = [scipy_sem(v) for v in bar_data]
    colors_b  = ["#C9922A", "#E8C07A", "#B5446E", "#D98AAD"]
    xlabs_b   = [
        "LLM\n" r"$\kappa_\mathrm{geo}$",
        "Hpc\n" r"$\kappa_\mathrm{geo}$",
        "LLM\n" r"$\kappa_\mathrm{nrm}$",
        "Hpc\n" r"$\kappa_\mathrm{nrm}$",
    ]

    xs_b = np.array([0, 0.62, 1.65, 2.27])
    rng = np.random.default_rng(42)
    for xi, mu, se, col, dots in zip(xs_b, bar_means, bar_sems, colors_b, bar_data):
        ax.bar(xi, mu, 0.46, color=col, alpha=0.85, zorder=3)
        ax.errorbar(xi, mu, yerr=se, fmt="none", color="k",
                    capsize=4, lw=1.3, zorder=4)
        jitter = rng.uniform(-0.11, 0.11, size=len(dots))
        ax.scatter(xi + jitter, dots, s=18, color=col,
                   edgecolors="white", lw=0.4, zorder=5, alpha=0.85)

    ax.axhline(0, color="k", lw=0.9, zorder=2)
    ax.axvline(1.13, color="gray", lw=0.7, ls=":", alpha=0.5, zorder=1)
    ax.text(0.31, 0.252, "On", ha="center", va="bottom",
            fontsize=7, color="#C9922A", fontweight="bold")
    ax.text(1.96, 0.252, "Off", ha="center", va="bottom",
            fontsize=7, color="#B5446E", fontweight="bold")

    ax.set_xticks(xs_b)
    ax.set_xticklabels(xlabs_b, fontsize=7.5)
    ax.set_ylabel(r"$r$ with surprisal")
    ax.set_ylim(-0.23, 0.28)
    ax.set_title(f"B  On/off components vs next-word surprisal\n({FOCAL_MODEL})",
                 loc="left", fontsize=8, fontweight="bold")

    N_BINS = 5

    def _component_bin_mats(records, geo_key, nrm_key):
        geo_mat = []
        nrm_mat = []
        for rec in records:
            reg = rec.get("regression", {})
            gb = reg.get(geo_key)
            nb = reg.get(nrm_key)
            if gb and nb and not any(np.isnan(gb)) and not any(np.isnan(nb)):
                geo_mat.append(gb)
                nrm_mat.append(nb)
        if not geo_mat:
            return None, None
        return np.array(geo_mat, dtype=float), np.array(nrm_mat, dtype=float)

    def _component_bin_summary(records, geo_key, nrm_key):
        geo_mat, nrm_mat = _component_bin_mats(records, geo_key, nrm_key)
        if geo_mat is None:
            return None

        valid_q1 = (np.abs(geo_mat[:, 0]) > 1e-12) & (np.abs(nrm_mat[:, 0]) > 1e-12)
        if valid_q1.sum() == 0:
            return None
        geo_norm = geo_mat[valid_q1] / geo_mat[valid_q1, 0:1]
        nrm_norm = nrm_mat[valid_q1] / nrm_mat[valid_q1, 0:1]

        return dict(
            geo_mu=geo_norm.mean(0),
            geo_se=scipy_sem(geo_norm, axis=0),
            nrm_mu=nrm_norm.mean(0),
            nrm_se=scipy_sem(nrm_norm, axis=0),
        )

    def _plot_series(ax, xs_q, mu, se, color, ls, marker, label,
                     lw=2.0, alpha=0.13):
        ax.plot(xs_q, mu, color=color, lw=lw, ls=ls, marker=marker, ms=5,
                label=label, zorder=3)
        ax.fill_between(xs_q, mu - se, mu + se, color=color, alpha=alpha,
                        zorder=2)

    # ── Panel C: LLM components by surprisal quintile ─────────────────────────
    ax = axC
    xs_q = np.arange(1, N_BINS + 1)
    llm_bins = _component_bin_summary(d_focal, "geo_by_surp_bin", "nrm_by_surp_bin")

    if llm_bins is None:
        ax.text(0.5, 0.5, "Rerun needed\n(component bins missing)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
                color="gray")
    else:
        _plot_series(ax, xs_q, llm_bins["geo_mu"], llm_bins["geo_se"],
                     "#C9922A", "-", "o", r"LLM $\kappa_\mathrm{geo}$",
                     lw=2.3, alpha=0.16)
        _plot_series(ax, xs_q, llm_bins["nrm_mu"], llm_bins["nrm_se"],
                     "#B5446E", "-", "o", r"LLM $\kappa_\mathrm{nrm}$",
                     lw=2.3, alpha=0.16)

        x_lab = 5.10
        ax.text(x_lab, llm_bins["nrm_mu"][-1],
                f"off {llm_bins['nrm_mu'][-1]:.2f}x",
                color="#B5446E", fontsize=6.7, va="center", ha="left")
        ax.text(x_lab, llm_bins["geo_mu"][-1],
                f"on {llm_bins['geo_mu'][-1]:.2f}x",
                color="#C9922A", fontsize=6.7, va="center", ha="left")
        ax.legend(fontsize=7, framealpha=0.65, loc="upper left",
                  ncol=1, columnspacing=1.0, handlelength=2.2)

    ax.set_xlim(0.8, 5.85)
    ax.set_ylim(0.84, 1.38)
    ax.set_xticks(xs_q)
    ax.set_xticklabels(["Low\n(Q1)", "Q2", "Q3", "Q4", "High\n(Q5)"],
                       fontsize=7)
    ax.set_xlabel("Surprisal quintile")
    ax.set_ylabel("Curvature (normalised to Q1)")
    ax.set_title("C  LLM component activation\napproaching high-surprisal words",
                 loc="left", fontsize=8, fontweight="bold")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.4)

    # ── Panel D: hpc components by surprisal quintile, zoomed scale ───────────
    ax = axD
    hpc_bins = _component_bin_summary(d_focal, "hpc_geo_by_surp_bin", "hpc_nrm_by_surp_bin")
    if hpc_bins is None:
        ax.text(0.5, 0.5, "Rerun needed\n(hpc bins missing)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
                color="gray")
    else:
        _plot_series(ax, xs_q, hpc_bins["geo_mu"], hpc_bins["geo_se"],
                     "#C9922A", "-", "s", r"Hpc $\kappa_\mathrm{geo}$",
                     lw=2.0, alpha=0.14)
        _plot_series(ax, xs_q, hpc_bins["nrm_mu"], hpc_bins["nrm_se"],
                     "#B5446E", "-", "s", r"Hpc $\kappa_\mathrm{nrm}$",
                     lw=2.0, alpha=0.14)
        x_lab = 5.08
        ax.text(x_lab, hpc_bins["nrm_mu"][-1],
                f"off {hpc_bins['nrm_mu'][-1]:.2f}x",
                color="#B5446E", fontsize=6.7, va="center", ha="left")
        ax.text(x_lab, hpc_bins["geo_mu"][-1],
                f"on {hpc_bins['geo_mu'][-1]:.2f}x",
                color="#C9922A", fontsize=6.7, va="center", ha="left")
        ax.legend(fontsize=7, framealpha=0.65, loc="upper left",
                  ncol=1, columnspacing=1.0, handlelength=2.2)

    ax.set_xlim(0.8, 5.72)
    if hpc_bins is None:
        ax.set_ylim(0.975, 1.08)
    else:
        hpc_bounds = np.concatenate([
            hpc_bins["geo_mu"] - hpc_bins["geo_se"],
            hpc_bins["geo_mu"] + hpc_bins["geo_se"],
            hpc_bins["nrm_mu"] - hpc_bins["nrm_se"],
            hpc_bins["nrm_mu"] + hpc_bins["nrm_se"],
        ])
        hpc_bounds = hpc_bounds[np.isfinite(hpc_bounds)]
        if hpc_bounds.size:
            pad = max(0.01, 0.12 * float(hpc_bounds.max() - hpc_bounds.min()))
            ax.set_ylim(max(0.94, float(hpc_bounds.min()) - pad),
                        min(1.18, float(hpc_bounds.max()) + pad))
        else:
            ax.set_ylim(0.975, 1.08)
    ax.set_xticks(xs_q)
    ax.set_xticklabels(["Low\n(Q1)", "Q2", "Q3", "Q4", "High\n(Q5)"],
                       fontsize=7)
    ax.set_xlabel("Surprisal quintile")
    ax.set_ylabel("Curvature (normalised to Q1)")
    ax.set_title("D  Hippocampal components\nraw neural space, zoomed scale",
                 loc="left", fontsize=8, fontweight="bold")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.4)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.08, top=0.93)
    _save(fig, "fig6_geodesic_decomposition")


def fig7_component_coupling_heatmap():
    """
    Heatmap of LLM component x hippocampal component coupling across models.

    Values are per-model means over patients of within-patient Pearson
    correlations between component time series at the LOO-neural layer.
    """
    from scipy.stats import ttest_1samp

    model_order = [
        "gpt2", "gpt2-medium", "bert-base", "roberta-base",
        "opt-350m", "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
    ]
    cols = [
        ("r_geo_sem_neu", r"Model $\kappa_\mathrm{geo}$" + "\n" +
         r"Hpc $\kappa_\mathrm{geo}$"),
        ("r_nrm_sem_neu", r"Model $\kappa_\mathrm{nrm}$" + "\n" +
         r"Hpc $\kappa_\mathrm{nrm}$"),
        ("r_geo_sem_nrm_neu", r"Model $\kappa_\mathrm{geo}$" + "\n" +
         r"Hpc $\kappa_\mathrm{nrm}$"),
        ("r_nrm_sem_geo_neu", r"Model $\kappa_\mathrm{nrm}$" + "\n" +
         r"Hpc $\kappa_\mathrm{geo}$"),
    ]

    labels = []
    means = []
    ses = []
    ps = []
    ns = []
    for model in model_order:
        data = _load(model, "hippocampus")
        if data is None:
            continue
        regs = [r["regression"] for r in data if r.get("regression")]
        if not regs:
            continue

        row_mu, row_se, row_p, row_n = [], [], [], []
        for key, _ in cols:
            vals = np.array([
                reg.get(key, np.nan) for reg in regs
                if reg.get(key) is not None and np.isfinite(reg.get(key, np.nan))
            ], dtype=float)
            row_n.append(len(vals))
            if len(vals) == 0:
                row_mu.append(np.nan)
                row_se.append(np.nan)
                row_p.append(np.nan)
            else:
                row_mu.append(float(np.mean(vals)))
                row_se.append(float(scipy_sem(vals)) if len(vals) > 1 else np.nan)
                row_p.append(float(ttest_1samp(vals, 0.0).pvalue)
                             if len(vals) > 1 else np.nan)

        labels.append(MODEL_STYLE.get(model, {}).get("label", model))
        means.append(row_mu)
        ses.append(row_se)
        ps.append(row_p)
        ns.append(min(row_n) if row_n else 0)

    mat = np.array(means, dtype=float)
    se_mat = np.array(ses, dtype=float)
    p_mat = np.array(ps, dtype=float)
    if mat.size == 0:
        raise RuntimeError("No component coupling values found in hippocampus JSONs.")

    finite = mat[np.isfinite(mat)]
    vmax = max(0.03, float(np.nanmax(np.abs(finite)))) if finite.size else 0.03
    vmax = min(0.12, np.ceil(vmax * 100) / 100)

    fig, ax = plt.subplots(figsize=(6.9, 4.25))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([label for _, label in cols], fontsize=7)
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False,
                   labeltop=True, pad=4)
    ax.tick_params(axis="y", length=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                txt = "--"
                color = "#333333"
            else:
                star = "*" if np.isfinite(p_mat[i, j]) and p_mat[i, j] < 0.05 else ""
                txt = f"{val:+.3f}{star}"
                color = "white" if abs(val) > 0.55 * vmax else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color=color)

    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Mean within-patient Pearson r")

    ax.set_title("Model-Hpc on/off curvature coupling across models",
                 loc="left", fontsize=10, fontweight="bold", pad=18)
    ax.text(0.0, -0.12,
            "Entries are mean r across patients at the LOO-neural layer; "
            "* p < 0.05 vs 0 across patients.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            color="#555555")
    fig.subplots_adjust(left=0.20, right=0.93, bottom=0.13, top=0.78)
    _save(fig, "fig7_component_coupling_heatmap")


def fig8_surprisal_component_trajectory(force=False):
    """
    Event-aligned trajectory through geodesic/normal curvature space.

    This is not a literal 2-D projection of the full manifold. Each point is the
    high-surprisal Q5/Q1 response of the geodesic and normal curvature
    components, and arrows show how that component profile evolves from the bend
    into a surprising word, to the bend centered on it, to the bend after it.
    """
    from scipy.stats import pearsonr
    from run_geometry_analysis import (
        PATIENTS, SENT_BOUNDARY_EXCL, fit_neural_pca, fit_neural_raw,
        geodesic_curvature, load_patient,
    )

    model = PRIMARY_MODEL
    cache_path = os.path.join(
        RESULTS, f"surprisal_component_trajectory_{model}.json")
    alignments = [
        ("into", -1, "into"),
        ("at", 0, "at"),
        ("after", 1, "after"),
    ]
    n_bins = 5

    def _valid_for_alignment(n, surp, sp, is_turn, offset):
        c_idx, s_idx = [], []
        for s in range(n):
            vertex = s + offset
            curv_i = vertex - 1
            if curv_i < 0 or curv_i >= n - 2:
                continue
            if not np.isfinite(surp[s]):
                continue
            if sp[s] < SENT_BOUNDARY_EXCL or sp[vertex] < SENT_BOUNDARY_EXCL:
                continue
            if is_turn[s]:
                continue
            c_idx.append(curv_i)
            s_idx.append(s)
        return np.array(c_idx, dtype=int), np.array(s_idx, dtype=int)

    def _q5_q1(component, surp_y):
        finite = np.isfinite(component) & np.isfinite(surp_y)
        if finite.sum() < n_bins * 5:
            return np.nan
        edges = np.percentile(surp_y[finite], np.linspace(0, 100, n_bins + 1))
        edges[-1] += 1e-9
        bidx = np.clip(np.digitize(surp_y, edges) - 1, 0, n_bins - 1)
        vals = []
        for b in range(n_bins):
            m = finite & (bidx == b)
            vals.append(np.nan if m.sum() == 0 else float(np.mean(component[m])))
        vals = np.array(vals, dtype=float)
        if not np.all(np.isfinite(vals)) or abs(vals[0]) <= 1e-12:
            return np.nan
        return float(vals[-1] / vals[0])

    plot_regions = ("hippocampus",)

    def _empty_store():
        return {
            "llm_hpc_layer": {name: {"geo": [], "nrm": [], "r_geo": [], "r_nrm": []}
                              for name, _, _ in alignments},
            "hippocampus": {name: {"geo": [], "nrm": [], "r_geo": [], "r_nrm": []}
                            for name, _, _ in alignments},
        }

    def _summarise(store):
        out = {}
        for space, by_align in store.items():
            out[space] = {}
            for name, vals in by_align.items():
                out[space][name] = {}
                for comp in ("geo", "nrm", "r_geo", "r_nrm"):
                    arr = np.array(vals[comp], dtype=float)
                    arr = arr[np.isfinite(arr)]
                    out[space][name][comp] = {
                        "values": arr.tolist(),
                        "mean": float(np.mean(arr)) if arr.size else float("nan"),
                        "sem": float(scipy_sem(arr)) if arr.size > 1 else float("nan"),
                        "n": int(arr.size),
                    }
        return out

    if os.path.exists(cache_path) and not force:
        with open(cache_path) as f:
            summary = json.load(f)
    else:
        store = _empty_store()

        for region in plot_regions:
            records = _load(model, region) or []
            layer_by_pid = {
                r["pid"]: r["regression"]["layer"]
                for r in records if r.get("regression")
            }

            for pid, preg in PATIENTS.items():
                if pid not in layer_by_pid:
                    continue
                data = load_patient(pid, preg, region, model)
                if data is None:
                    continue

                n = data["n_words"]
                layer = layer_by_pid[pid]
                emb_mm = np.load(data["emb_path"], mmap_mode="r")
                emb = np.array(emb_mm[layer, :n], dtype=np.float64)
                del emb_mm
                llm_geo, llm_nrm = geodesic_curvature(
                    emb, k_neighbors=20, tangent_dim=10)

                Y_neu, _ = fit_neural_pca(data["neu_vecs"], n)
                Y_raw = fit_neural_raw(data["neu_vecs"], n)
                neu = Y_raw if Y_raw is not None else Y_neu
                if neu is None:
                    continue
                neu_geo, neu_nrm = geodesic_curvature(
                    neu,
                    k_neighbors=min(15, len(neu) - 1),
                    tangent_dim=min(5, neu.shape[1] - 1),
                )

                for name, offset, _ in alignments:
                    cidx, sidx = _valid_for_alignment(
                        n, data["surp"], data["sent_pos"], data["is_turn"], offset)
                    if len(cidx) < n_bins * 5:
                        continue
                    surp_y = data["surp"][sidx]

                    for key, geo_arr, nrm_arr in [
                        ("llm_hpc_layer", llm_geo, llm_nrm),
                        (region, neu_geo, neu_nrm),
                    ]:
                        if key == "llm_hpc_layer" and region != "hippocampus":
                            continue
                        comp_geo = geo_arr[cidx]
                        comp_nrm = nrm_arr[cidx]
                        g_ratio = _q5_q1(comp_geo, surp_y)
                        n_ratio = _q5_q1(comp_nrm, surp_y)
                        if np.isfinite(g_ratio):
                            store[key][name]["geo"].append(g_ratio)
                        if np.isfinite(n_ratio):
                            store[key][name]["nrm"].append(n_ratio)
                        for comp_name, comp in [("r_geo", comp_geo),
                                                ("r_nrm", comp_nrm)]:
                            finite = np.isfinite(comp) & np.isfinite(surp_y)
                            if finite.sum() > 10:
                                store[key][name][comp_name].append(
                                    float(pearsonr(comp[finite], surp_y[finite]).statistic))

        summary = _summarise(store)
        with open(cache_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> {cache_path}")

    panel_specs = [
        ("llm_hpc_layer", "A  LLaMA-3.1-8B\nsemantic trajectory",
         "#922B21", False),
        ("hippocampus", "B  Hippocampus\nneural trajectory (rotated view)",
         "#1B7F4A", True),
    ]
    point_style = {
        "into": dict(color="#C0392B", marker="o", label="into word"),
        "at": dict(color="#D35400", marker="s", label="at word"),
        "after": dict(color="#2980B9", marker="^", label="after word"),
    }
    label_offsets = {
        "llm_hpc_layer": {"into": (8, 0), "at": (8, 0), "after": (8, 0)},
        "hippocampus": {"into": (-28, -2), "at": (8, 0), "after": (8, -7)},
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.35, 3.25))
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.76,
                        wspace=0.42)

    for ax, (space, title, line_color, swap_axes) in zip(axes, panel_specs):
        pts, xerr, yerr = [], [], []
        for name, _, _ in alignments:
            rec = summary[space][name]
            pts.append([rec["geo"]["mean"], rec["nrm"]["mean"]])
            xerr.append(rec["geo"]["sem"])
            yerr.append(rec["nrm"]["sem"])
        pts = np.array(pts, dtype=float)
        xerr = np.array(xerr, dtype=float)
        yerr = np.array(yerr, dtype=float)
        plot_pts = pts[:, [1, 0]] if swap_axes else pts
        plot_xerr = yerr if swap_axes else xerr
        plot_yerr = xerr if swap_axes else yerr

        ax.axvline(1.0, color="k", lw=0.7, ls="--", alpha=0.35)
        ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.35)
        ax.plot(plot_pts[:, 0], plot_pts[:, 1], color=line_color, lw=1.8,
                alpha=0.85, zorder=2)
        for i in range(len(pts) - 1):
            ax.annotate("", xy=plot_pts[i + 1], xytext=plot_pts[i],
                        arrowprops=dict(arrowstyle="-|>", color=line_color,
                                        lw=1.4, shrinkA=4, shrinkB=4),
                        zorder=3)

        for i, (name, _, label) in enumerate(alignments):
            st = point_style[name]
            ax.errorbar(plot_pts[i, 0], plot_pts[i, 1],
                        xerr=None if not np.isfinite(plot_xerr[i]) else plot_xerr[i],
                        yerr=None if not np.isfinite(plot_yerr[i]) else plot_yerr[i],
                        fmt=st["marker"], color=st["color"], ms=6.5,
                        mec="white", mew=0.6, ecolor=st["color"],
                        elinewidth=1.0, capsize=2, zorder=4,
                        label=st["label"] if space == "llm_hpc_layer" else None)
            dx, dy = label_offsets.get(space, {}).get(name, (8, 0))
            ax.annotate(label, xy=(plot_pts[i, 0], plot_pts[i, 1]),
                        xytext=(dx, dy),
                        textcoords="offset points", fontsize=7,
                        ha="left" if dx >= 0 else "right", va="center",
                        color=st["color"])

        finite_pts = plot_pts[np.isfinite(plot_pts).all(axis=1)]
        if finite_pts.size:
            xmin, xmax = finite_pts[:, 0].min(), finite_pts[:, 0].max()
            ymin, ymax = finite_pts[:, 1].min(), finite_pts[:, 1].max()
            xpad = max(0.015, 0.22 * (xmax - xmin))
            ypad = max(0.015, 0.22 * (ymax - ymin))
            ax.set_xlim(min(0.98, xmin - xpad), max(1.02, xmax + xpad))
            ax.set_ylim(min(0.98, ymin - ypad), max(1.02, ymax + ypad))

        if swap_axes:
            ax.set_xlabel(r"Normal $\kappa$  (Q5/Q1)")
            ax.set_ylabel(r"Geodesic / tangent $\kappa$  (Q5/Q1)")
            ax.text(0.02, 0.88, "axes swapped\nfor readability",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.5, color="#777777")
        else:
            ax.set_xlabel(r"Geodesic / tangent $\kappa$  (Q5/Q1)")
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
        ax.text(0.02, 0.98, "high surprisal\nvs low surprisal",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, color="#555555")
    axes[0].set_ylabel(r"Normal $\kappa$  (Q5/Q1)")
    axes[0].legend(loc="lower left", fontsize=6.5, framealpha=0.75,
                   edgecolor="none")
    fig.suptitle("Curvature component trajectory around surprising words",
                 fontsize=10, fontweight="bold", y=0.96)
    _save(fig, "fig8_surprisal_component_trajectory")


def fig9_surprisal_component_trajectory_3d():
    """
    3-D view of the event-aligned component trajectory.

    Axes are geodesic/tangent curvature, normal curvature, and event phase
    relative to the surprising word. This keeps the third axis interpretable:
    it is time/phase, not an inferred hidden neural dimension.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    model = PRIMARY_MODEL
    cache_path = os.path.join(
        RESULTS, f"surprisal_component_trajectory_{model}.json")
    if not os.path.exists(cache_path):
        fig8_surprisal_component_trajectory(force=True)
    with open(cache_path) as f:
        summary = json.load(f)

    alignments = [
        ("into", -1.0, "into", "#C0392B", "o"),
        ("at", 0.0, "at", "#D35400", "s"),
        ("after", 1.0, "after", "#2980B9", "^"),
    ]
    panel_specs = [
        ("llm_hpc_layer", "A  LLaMA-3.1-8B\nsemantic trajectory", "#922B21"),
        ("hippocampus", "B  Hippocampus\nneural trajectory", "#1B7F4A"),
    ]

    fig = plt.figure(figsize=(7.2, 3.4))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.82,
                        wspace=0.04)

    for ax, (space, title, line_color) in zip(axes, panel_specs):
        xs, ys, zs = [], [], []
        for name, phase, _, _, _ in alignments:
            rec = summary[space][name]
            xs.append(rec["geo"]["mean"])
            ys.append(rec["nrm"]["mean"])
            zs.append(phase)
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        zs = np.array(zs, dtype=float)

        ax.plot(xs, ys, zs, color=line_color, lw=2.0, alpha=0.9)

        for name, phase, label, color, marker in alignments:
            rec = summary[space][name]
            x = rec["geo"]["mean"]
            y = rec["nrm"]["mean"]
            z = phase
            ax.scatter([x], [y], [z], color=color, marker=marker, s=58,
                       edgecolor="white", linewidth=0.7, depthshade=False)
            ax.text(x, y, z, f" {label}", color=color, fontsize=7)

        finite = np.isfinite(xs) & np.isfinite(ys)
        if finite.any():
            xpad = max(0.015, 0.24 * float(xs[finite].max() - xs[finite].min()))
            ypad = max(0.015, 0.24 * float(ys[finite].max() - ys[finite].min()))
            ax.set_xlim(min(0.98, float(xs[finite].min()) - xpad),
                        max(1.02, float(xs[finite].max()) + xpad))
            ax.set_ylim(min(0.98, float(ys[finite].min()) - ypad),
                        max(1.02, float(ys[finite].max()) + ypad))
        ax.set_zlim(-1.15, 1.15)
        ax.set_zticks([-1, 0, 1])
        ax.set_zticklabels(["into", "at", "after"], fontsize=7)
        ax.set_xlabel(r"Geo/tangent $\kappa$", labelpad=5)
        ax.set_ylabel(r"Normal $\kappa$", labelpad=6)
        ax.set_zlabel("Event phase", labelpad=5)
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=4)
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1.2, 1.0, 0.85))
        ax.view_init(elev=22, azim=-54)
        ax.grid(True, alpha=0.25)

    fig.suptitle("3-D component trajectory around surprising words",
                 fontsize=10, fontweight="bold", y=0.98)
    _save(fig, "fig9_surprisal_component_trajectory_3d")


def make_latex_table():
    """
    Booktabs LaTeX table: one row per model, columns = geometry metrics at the
    LOO-neural layer averaged across patients (hippocampus).
    """
    from scipy.stats import ttest_1samp

    MODEL_ORDER = [
        "gpt2", "gpt2-medium", "bert-base", "roberta-base",
        "opt-350m", "llama-2-7b", "llama-3.1-8b", "gemma-2-9b", "mistral-7b",
    ]
    MODEL_LABEL = {
        "gpt2":         "GPT-2",
        "gpt2-medium":  "GPT-2 medium",
        "bert-base":    "BERT-base",
        "roberta-base": "RoBERTa-base",
        "opt-350m":     "OPT-350M",
        "llama-2-7b":   "LLaMA-2 7B",
        "llama-3.1-8b": "LLaMA-3.1 8B",
        "gemma-2-9b":   "Gemma-2 9B",
        "mistral-7b":   "Mistral-7B",
    }

    def _fmt(mu, se, p, bold=False):
        """Format mean ± SE with significance star."""
        star = "*" if p < 0.05 else ""
        s    = f"{mu:+.3f}\\tiny{{±{se:.3f}{star}}}"
        return f"\\textbf{{{s}}}" if bold else s

    def _fmtx(mu, se):
        return f"{mu:.3f}\\tiny{{±{se:.3f}}}"

    rows = []
    for model in MODEL_ORDER:
        d = _load(model, "hippocampus")
        if d is None:
            continue
        regs = [r["regression"] for r in d if r.get("regression")]
        if not regs:
            continue

        def _col(key):
            vals = [r[key] for r in regs if r.get(key) is not None
                    and not np.isnan(r[key])]
            if not vals:
                return np.nan, np.nan, np.nan
            mu  = float(np.mean(vals))
            se  = float(scipy_sem(vals))
            _, p = ttest_1samp(vals, 0.)
            return mu, se, p

        r_sem_mu,  r_sem_se,  r_sem_p   = _col("r_sem")
        r_neu_mu,  r_neu_se,  r_neu_p   = _col("r_neu")
        r_geo_mu,  r_geo_se,  r_geo_p   = _col("r_sem_geo")
        r_nrm_mu,  r_nrm_se,  r_nrm_p   = _col("r_sem_nrm")
        sec_mu,    sec_se,    _          = _col("mean_sec_slope")
        curv_mu,   curv_se,   _          = _col("mean_curv")
        disp_mu,   disp_se,   _          = _col("mean_disp")
        tors_mu,   tors_se,   _          = _col("mean_torsion")
        str_mu,    str_se,    _          = _col("straightness")
        r2b_mu,    r2b_se,    _          = _col("r2")
        r2f_mu,    r2f_se,    _          = _col("r2_full")

        n_layers  = len(d[0]["profile"]["r_surp"])
        loo_mean  = float(np.mean([r["loo_layer_neu"] for r in d]))
        loo_norm  = loo_mean / (n_layers - 1)
        positive  = r_sem_mu > 0

        rows.append(dict(
            label=MODEL_LABEL[model],
            r_sem_mu=r_sem_mu, r_sem_se=r_sem_se, r_sem_p=r_sem_p,
            r_neu_mu=r_neu_mu, r_neu_se=r_neu_se, r_neu_p=r_neu_p,
            r_geo_mu=r_geo_mu, r_geo_se=r_geo_se, r_geo_p=r_geo_p,
            r_nrm_mu=r_nrm_mu, r_nrm_se=r_nrm_se, r_nrm_p=r_nrm_p,
            sec_mu=sec_mu,     sec_se=sec_se,
            curv_mu=curv_mu,   curv_se=curv_se,
            disp_mu=disp_mu,   disp_se=disp_se,
            tors_mu=tors_mu,   tors_se=tors_se,
            str_mu=str_mu,     str_se=str_se,
            r2b_mu=r2b_mu,     r2b_se=r2b_se,
            r2f_mu=r2f_mu,     r2f_se=r2f_se,
            loo_norm=loo_norm,  positive=positive,
        ))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        r"Model & $r_\mathrm{sem}$ & $r_\mathrm{geo}$ & $r_\mathrm{nrm}$ & "
        r"$\bar\kappa_\mathrm{sec}$ & $r_\mathrm{neu}$ & "
        r"Curv (°) & Disp & Torsion (°) & Straight & "
        r"$R^2_\mathrm{base}$ & $R^2_\mathrm{full}$ \\",
        r"\midrule",
    ]

    for r in rows:
        bold_sem = r["r_sem_p"] < 0.05
        bold_geo = r.get("r_geo_p", 1.0) < 0.05
        sec_str  = (f"{r['sec_mu']:+.3f}\\tiny{{±{r['sec_se']:.3f}}}"
                    if r.get("sec_mu") is not None and not np.isnan(r.get("sec_mu", float("nan")))
                    else "---")
        row = (
            f"{r['label']} & "
            f"{_fmt(r['r_sem_mu'], r['r_sem_se'], r['r_sem_p'], bold=bold_sem)} & "
            f"{_fmt(r['r_geo_mu'], r['r_geo_se'], r['r_geo_p'], bold=bold_geo)} & "
            f"{_fmt(r['r_nrm_mu'], r['r_nrm_se'], r['r_nrm_p'])} & "
            f"{sec_str} & "
            f"{_fmt(r['r_neu_mu'], r['r_neu_se'], r['r_neu_p'])} & "
            f"{_fmtx(r['curv_mu'], r['curv_se'])} & "
            f"{_fmtx(r['disp_mu'], r['disp_se'])} & "
            f"{_fmtx(r['tors_mu'], r['tors_se'])} & "
            f"{_fmtx(r['str_mu'], r['str_se'])} & "
            f"{r['r2b_mu']:.4f} & "
            f"{r['r2f_mu']:.4f} \\\\"
        )
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Geometry metrics at the LOO-neural layer (hippocampus, mean $\pm$ SEM "
        r"across patients). "
        r"$r_\mathrm{sem}$: total curvature--surprisal correlation. "
        r"$r_\mathrm{geo}$: geodesic (on-manifold) curvature--surprisal. "
        r"$r_\mathrm{nrm}$: normal (off-manifold) curvature--surprisal. "
        r"$\bar\kappa_\mathrm{sec}$: mean sectional curvature slope (negative = sphere-like). "
        r"$r_\mathrm{neu}$: neural curvature--surprisal. "
        r"Straight: end-to-end / arc length. "
        r"$R^2_\mathrm{base}$: model with sent\_pos + log\_freq. "
        r"$R^2_\mathrm{full}$: + is\_speaker + word\_length + dist\_to\_turn. "
        r"* $p < 0.05$ (one-sample $t$-test vs 0).}",
        r"\label{tab:geometry}",
        r"\end{table}",
    ]

    out = "\n".join(lines)
    path = os.path.join(RESULTS, "geometry_table.tex")
    with open(path, "w") as fh:
        fh.write(out)
    print(f"  → {path}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Primary model: {PRIMARY_MODEL}")
    print(f"Results dir:   {RESULTS}\n")

    print("Figure 1: layer profiles …")
    fig1_layer_profiles()

    print("Figure 2: model comparison …")
    fig2_surprisal_geometry()
    fig2_surprisal_displacement()
    fig2_model_comparison()

    print("Figure 3: event-triggered …")
    fig3_event_triggered()

    print("Figure 4: regression summary …")
    fig4_regression()

    print("Figure 5: trajectory visualisation …")
    fig5_trajectory_viz()

    print("Figure 5 example 2: trajectory visualisation …")
    fig5_trajectory_viz(rank=2, out_name="fig5_trajectory_viz_example2",
                        title_prefix="Sentence trajectory example 2")

    print("Figure 5 example 3: trajectory visualisation …")
    fig5_trajectory_viz(rank=3, out_name="fig5_trajectory_viz_example3",
                        title_prefix="Sentence trajectory example 3")

    print("Figure 5 short sentence: trajectory visualisation …")
    fig5_trajectory_viz(rank=6,
                        out_name="fig5_trajectory_viz_short_sentence",
                        title_prefix="Short sentence trajectory",
                        selection="word_count")

    print("Figure 6: geodesic curvature decomposition …")
    fig6_geodesic_decomposition()

    print("Figure 7: component coupling heatmap …")
    fig7_component_coupling_heatmap()

    print("Figure 8: surprisal component trajectory …")
    fig8_surprisal_component_trajectory()

    print("Figure 9: 3-D surprisal component trajectory …")
    fig9_surprisal_component_trajectory_3d()

    print("LaTeX table …")
    make_latex_table()

    print(f"\nDone. Figures → {FIGURES}")
