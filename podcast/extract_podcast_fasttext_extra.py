"""
extract_podcast_fasttext_extra.py
==================================
Extends extract_podcast_fasttext.py to the 4 EXTRA_SOURCES patients
(YEX, YFB, YFX, YFU) added to PODCAST_PATIENTS on 2026-09-04, which
extract_podcast_fasttext.py can't reach directly since it sources words via
find_transcript() (old PODCAST_ROOT-only transcript loader). Instead, this
uses podcast_loader.load_podcast_patient(), which already handles both the
regular and EXTRA_SOURCES transcript formats uniformly and returns the same
cleaned `words` convention (lowercase, non-alphanumeric stripped) that
extract_podcast_fasttext.py uses -- so the resulting cache is identical in
format and lookup logic to the original 9-patient run.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_podcast_fasttext import word_vector, load_fasttext_kv, CACHE_DIR, MODEL_TAG
from podcast_loader import load_podcast_patient

EXTRA_PIDS = ["YEX", "YFB", "YFX", "YFU"]


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    kv = load_fasttext_kv()
    dim = kv.vector_size

    for pid in EXTRA_PIDS:
        out_path = os.path.join(CACHE_DIR, f"{pid}_{MODEL_TAG}_word_emb_layers.npy")
        if os.path.exists(out_path):
            print(f"[SKIP] {pid} -- already exists", flush=True)
            continue

        data = load_podcast_patient(pid, region="hippocampus", model_tag="llama-3.1-8b", context_words=200)
        if data is None:
            print(f"[SKIP] {pid} -- could not load via podcast_loader", flush=True)
            continue
        words = data["words"]
        n_words = len(words)

        emb = np.zeros((1, n_words, dim), dtype=np.float32)
        n_oov = 0
        for i, w in enumerate(words):
            vec, found = word_vector(kv, w, dim)
            emb[0, i] = vec
            if not found:
                n_oov += 1

        np.save(out_path, emb)
        print(f"  {pid}: {n_words} words, {n_oov} OOV ({100*n_oov/n_words:.1f}%)  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
