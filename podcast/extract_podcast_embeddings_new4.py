"""
extract_podcast_embeddings_new4.py
====================================
Same extraction pipeline as extract_podcast_embeddings.py (all 11 models,
ctx200, saves to podcast/EmbedCache/), but for the 4 podcast patients added
2026-09 (YEX, YFB, YFX, YFU) whose transcripts don't live in the standard
PODCAST_ROOT/{pid}/transcript/ folder -- only load_words() is overridden to
pull text from each patient's actual source:
  YEX, YFB : HPC_YEXthurYFP_WordFRMatrix_Duration80msDelay.mat ({short}_word)
  YFX      : PTYFX_EMU-0072_podcast.xlsx ("words" column)
  YFU      : YFU_bundles/PTYFU_podcast.mat ("words" field)

Usage:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed --no-capture-output python3 -u extract_podcast_embeddings_new4.py
"""
import os, sys, re
import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_podcast_embeddings as base

WORDFR_PATH = "/scratch/aniluchavez/semantic_syntactic_emb_project_for_Chavez/HPC_YEXthurYFP_WordFRMatrix_Duration80msDelay.mat"
YFX_XLSX = "/scratch/aniluchavez/semantic_syntactic_emb_project_for_Chavez/PTYFX_EMU-0072_podcast.xlsx"
YFU_MAT = "/scratch/aniluchavez/semantic_syntactic_emb_project_for_Chavez/YFU_bundles/PTYFU_podcast.mat"

_wordfr_cache = None
_clean = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_words_new4(pid):
    global _wordfr_cache
    if pid in ("YEX", "YFB"):
        if _wordfr_cache is None:
            _wordfr_cache = sio.loadmat(WORDFR_PATH, simplify_cells=True)
        words = _wordfr_cache[f"{pid.lower()}_word"]
        return [_clean(w["text"]) for w in words]
    elif pid == "YFX":
        df = pd.read_excel(YFX_XLSX)
        return [_clean(w) for w in df["words"].fillna("").astype(str).tolist()]
    elif pid == "YFU":
        d = sio.loadmat(YFU_MAT, simplify_cells=True)
        return [_clean(w) for w in d["words"]]
    else:
        return base.load_words(pid)


base.load_words = load_words_new4
base.PODCAST_PATIENTS = ["YEX", "YFB", "YFX", "YFU"]

if __name__ == "__main__":
    sys.argv = [sys.argv[0]]  # ignore any accidental extra args, use defaults (all 11 models, ctx200)
    base.main()
