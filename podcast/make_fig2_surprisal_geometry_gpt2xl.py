"""
make_fig2_surprisal_geometry_gpt2xl.py
====================================
make_geometry_figures_podcast_pca20.py's fig2_surprisal_geometry(), with
PRIMARY_MODEL switched to gpt2-xl instead of llama-3.1-8b. Requires
scatter_data_podcast_pca20_gpt2-xl_hippocampus.json (built by
compute_scatter_data_podcast_pca20.py with MODEL/N_LAYERS overridden to
gpt2-xl/49) and geometry_podcast_pca20_gpt2-xl_ctx200_hippocampus.json
(already built via run_podcast_geometry_analysis_pca20.py --models gpt2-xl).

Saves: figures/fig2_surprisal_geometry_pca20_podcast_gpt2xl.{pdf,png,svg}
"""
import os
import make_geometry_figures_podcast_pca20 as gf

gf.PRIMARY_MODEL = "gpt2-xl"

_orig_save = gf._save
def _save_renamed(fig, name):
    if name == "fig2_surprisal_geometry_pca20_podcast":
        name = "fig2_surprisal_geometry_pca20_podcast_gpt2xl"
    return _orig_save(fig, name)
gf._save = _save_renamed

if __name__ == "__main__":
    gf.fig2_surprisal_geometry()
