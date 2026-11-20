"""
make_podcast_chunk_context_sweep_fig.py
=========================================
Shows mean r(LLM, neural) vs chunk size for curvature and displacement,
one line per context length (200/100/50/10/3), gpt2-xl, podcast.
Answers: does the context-length effect survive chunk-averaging, or fade
out as chunks get larger?

Run:
    conda run -n gpt2_embed python3 make_podcast_chunk_context_sweep_fig.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem as scipy_sem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(BASE_DIR, "results")
FIGURES  = os.path.join(BASE_DIR, "figures")

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size":         13,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

CONTEXT_STYLE = {
    "200": dict(color="#1F618D", label="ctx200"),
    "100": dict(color="#2980B9", label="ctx100"),
    "50":  dict(color="#16A085", label="ctx50"),
    "10":  dict(color="#E67E22", label="ctx10"),
    "3":   dict(color="#C0392B", label="ctx3"),
}

METRICS = [
    dict(key="curv", label="Curvature", panel="A"),
    dict(key="disp", label="Displacement", panel="B"),
]


def _save(fig, name):
    for ext in ("pdf", "png", "svg"):
        p = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(p)
        if ext == "svg":
            txt = open(p).read().replace("TeX Gyre Termes", "Times New Roman")
            open(p, "w").write(txt)
        print(f"  -> {p}")
    plt.close(fig)


def main():
    sweep_path = os.path.join(RESULTS, "podcast_chunk_context_sweep.json")
    data = json.load(open(sweep_path))
    contexts = [c for c in ["200", "100", "50", "10", "3"] if c in data]
    chunk_sizes = sorted(int(k) for k in data[contexts[0]].keys())
    xs = np.array(chunk_sizes)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.8, wspace=0.35)

    for ax, metric in zip(axes, METRICS):
        mk = metric["key"]
        for ctx in contexts:
            st = CONTEXT_STYLE[ctx]
            mus, sems = [], []
            for cs in chunk_sizes:
                vals = data[ctx][str(cs)].get(mk, [])
                vals = np.array([v for v in vals if v is not None and not np.isnan(v)])
                mus.append(vals.mean() if len(vals) else np.nan)
                sems.append(scipy_sem(vals) if len(vals) > 1 else np.nan)
            mus, sems = np.array(mus), np.array(sems)
            ax.plot(xs, mus, color=st["color"], lw=1.8, marker="o", ms=4, label=st["label"])
            ax.fill_between(xs, mus - sems, mus + sems, color=st["color"], alpha=0.15)

        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xlabel("Chunk size (words)")
        ax.set_ylabel("Mean r  (LLM <-> Neural)")
        ax.set_title(f"{metric['panel']}  {metric['label']}\ngpt2-xl . podcast . hippocampus . avg layers",
                     loc="left", fontsize=10)
        ax.set_xticks(chunk_sizes)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle("Cross-space coupling vs chunk size, by context length  ·  Podcast",
                 fontsize=10, fontweight="bold", y=0.94)

    _save(fig, "fig_podcast_chunk_context_sweep")
    print("Done.")


if __name__ == "__main__":
    main()
