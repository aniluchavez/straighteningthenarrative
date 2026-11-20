"""
make_fig2_surprisal_geometry_podcast_llama2.py
================================================
Runs make_geometry_figures_podcast_pca20.fig2_surprisal_geometry() with
PRIMARY_MODEL patched to "llama-2-7b" instead of the default "llama-3.1-8b",
and saves under a "_llama2" suffix so it doesn't clobber the original.

Requires results/scatter_data_podcast_pca20_llama-2-7b_hippocampus.json
(from compute_scatter_data_podcast_pca20_llama2.py) and
results/geometry_podcast_pca20_llama-2-7b_ctx200_hippocampus.json to exist.

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u make_fig2_surprisal_geometry_podcast_llama2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_geometry_figures_podcast_pca20 as m

m.PRIMARY_MODEL = "llama-2-7b"

_orig_save = m._save
def _save_suffixed(fig, name):
    _orig_save(fig, f"{name}_llama2")
m._save = _save_suffixed

if __name__ == "__main__":
    m.fig2_surprisal_geometry()
