"""
compute_scatter_data_podcast_pca20_llama2.py
==============================================
Same as compute_scatter_data_podcast_pca20.py but for llama-2-7b instead of
llama-3.1-8b (also 33 layers). Saves
results/scatter_data_podcast_pca20_llama-2-7b_hippocampus.json.

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u compute_scatter_data_podcast_pca20_llama2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_scatter_data_podcast_pca20 as m

m.MODEL = "llama-2-7b"
m.N_LAYERS = 33

if __name__ == "__main__":
    m.main()
