"""
make_fig1_AB_podcast_pca20_dur8040.py
========================
Standalone remake of Panels A + B from fig1_layer_profiles_pca20_podcast.pdf
(make_geometry_figures_podcast_pca20.py), with Panel B recomputed using the
duration-locked, rate-normalized neural window (onset+80ms/offset+40ms,
spike rate) validated in compute_pca_sweep_podcast_dur80_40_rate.py, instead
of the pipeline-default fixed 500ms/raw-count window.

Panel A (mean semantic curvature per layer) is UNCHANGED -- it only depends
on the LLM embedding, not the neural window, so it's reused as-is from the
original geometry_podcast_pca20_* cache.

Panel B (neural-LLM curvature alignment per layer) is rebuilt from
cross_space_layers_podcast_pca20_dur8040_{model}_hippocampus.json
(compute_cross_space_layers_podcast_pca20_dur8040.py's output).

Saves: figures/fig1_AB_podcast_pca20_dur8040.{pdf,png,svg}
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
RESULTS    = os.path.join(BASE_DIR, "results")
FIGURES    = os.path.join(PARENT_DIR, "figures")
PRIMARY_MODEL = "llama-3.1-8b"
CTX_TAG    = "_ctx200"

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
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         14,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})


def _load_profile(model):
    path = os.path.join(RESULTS, f"geometry_podcast_pca20_{model}{CTX_TAG}_hippocampus.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def _load_cross_space_dur8040(model):
    path = os.path.join(RESULTS, f"cross_space_layers_podcast_pca20_dur8040_{model}_hippocampus.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def main():
    GRID = np.linspace(0, 1, 100)
    fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.82, wspace=0.32)

    # Background: all non-primary models
    for model, st in MODEL_STYLE.items():
        if model == PRIMARY_MODEL:
            continue
        data_m = _load_profile(model)
        if data_m is not None:
            n_l  = len(data_m[0]["profile"]["mean_curv"])
            xs_m = np.linspace(0, 1, n_l)
            mu_c = np.array([r["profile"]["mean_curv"] for r in data_m]).mean(0)
            ax_A.plot(GRID, np.interp(GRID, xs_m, mu_c),
                      color=st["color"], ls=st["ls"], lw=0.9, alpha=0.55, label=st["label"])

        cs_m = _load_cross_space_dur8040(model)
        if cs_m:
            n_l_cs = cs_m[0]["n_layers"]
            xs_cs  = np.linspace(0, 1, n_l_cs)
            mat_cs = np.array([r["r_curv_layers_ch"] for r in cs_m if "r_curv_layers_ch" in r], dtype=float)
            if len(mat_cs):
                mu_cs = np.nanmean(mat_cs, axis=0)
                ax_B.plot(GRID, np.interp(GRID, xs_cs, np.nan_to_num(mu_cs)),
                          color=st["color"], ls=st["ls"], lw=0.9, alpha=0.55, label="_nolegend_")

    # Primary model: LLaMA-3.1-8B
    llama_st = MODEL_STYLE[PRIMARY_MODEL]
    data = _load_profile(PRIMARY_MODEL)
    if data is not None:
        n_layers = len(data[0]["profile"]["mean_curv"])
        xs_norm  = np.linspace(0, 1, n_layers)
        mu_c = np.array([r["profile"]["mean_curv"] for r in data]).mean(0)
        se_c = scipy_sem(np.array([r["profile"]["mean_curv"] for r in data]), axis=0)
        ax_A.plot(xs_norm, mu_c, color=llama_st["color"], ls=llama_st["ls"], lw=1.8, label=llama_st["label"])
        ax_A.fill_between(xs_norm, mu_c - se_c, mu_c + se_c, color=llama_st["color"], alpha=0.15, lw=0)

    cs_prim = _load_cross_space_dur8040(PRIMARY_MODEL)
    if cs_prim:
        n_l_p = cs_prim[0]["n_layers"]
        xs_p  = np.linspace(0, 1, n_l_p)
        mat_p = np.array([r["r_curv_layers_ch"] for r in cs_prim if "r_curv_layers_ch" in r], dtype=float)
        mu_p  = np.nanmean(mat_p, axis=0)
        se_p  = scipy_sem(mat_p, axis=0, nan_policy="omit")
        ax_B.plot(xs_p, mu_p, color=llama_st["color"], ls=llama_st["ls"], lw=1.8, label="_nolegend_")
        ax_B.fill_between(xs_p, mu_p - se_p, mu_p + se_p, color=llama_st["color"], alpha=0.15, lw=0)

    # Panel B null band: word-order shuffle, averaged across all models
    null_mu_interp, null_sd_interp = [], []
    for model in MODEL_STYLE:
        cs_null = _load_cross_space_dur8040(model)
        if not cs_null:
            continue
        mat_mu = np.array([r["null_curv_layers_ch_mean"] for r in cs_null if "null_curv_layers_ch_mean" in r], dtype=float)
        mat_sd = np.array([r["null_curv_layers_ch_std"] for r in cs_null if "null_curv_layers_ch_std" in r], dtype=float)
        if not len(mat_mu) or not len(mat_sd):
            continue
        mu_null = np.nanmean(mat_mu, axis=0)
        sd_null = np.nanmean(mat_sd, axis=0)
        xs_null = np.linspace(0, 1, len(mu_null))
        null_mu_interp.append(np.interp(GRID, xs_null, np.nan_to_num(mu_null)))
        null_sd_interp.append(np.interp(GRID, xs_null, np.nan_to_num(sd_null)))

    if null_mu_interp:
        nb_mu = np.nanmean(np.array(null_mu_interp), axis=0)
        nb_sd = np.nanmean(np.array(null_sd_interp), axis=0)
        ax_B.fill_between(GRID, nb_mu - nb_sd, nb_mu + nb_sd, color="#777777", alpha=0.16, lw=0, zorder=0, label="_nolegend_")
        ax_B.plot(GRID, nb_mu, color="#555555", ls=(0, (1.2, 1.2)), lw=1.2, zorder=1, label="Word-order shuffle null")

    ax_A.set_xlabel("Normalized layer position")
    ax_A.set_ylabel("Mean curvature (°)")
    ax_A.set_title("$\\mathbf{A}$  Mean semantic curvature · all models", loc="left", pad=5, fontsize=11.5)
    ax_A.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax_A.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax_A.legend(loc="best", fontsize=6.5, framealpha=0.8, edgecolor="#DDDDDD", ncol=2)

    ax_B.axhline(0, color="k", lw=0.5, ls=":")
    ax_B.set_xlabel("Normalized layer position")
    ax_B.set_ylabel("Pearson $r$ (curvature)")
    ax_B.set_title("$\\mathbf{B}$  Neural-LLM alignment · dur80/40, rate-normalized\n15-word chunks", loc="left", pad=5, fontsize=11.5)
    ax_B.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax_B.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax_B.legend(loc="best", fontsize=7, framealpha=0.8, edgecolor="#DDDDDD")

    os.makedirs(FIGURES, exist_ok=True)
    stem = "fig1_AB_podcast_pca20_dur8040"
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
