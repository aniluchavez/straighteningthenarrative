"""
extract_podcast_fasttext.py
=============================
Podcast-dataset version of extract_embeddings_fasttext.py: static fastText
lookup, one vector per word regardless of context (so "context window"
doesn't apply -- this is the fixed baseline used to contrast against the
contextual models in extract_podcast_embeddings.py).

Reuses the same cached wiki-news-300d-1M KeyedVectors as the convo pipeline.
"""
import os, string
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_podcast_embeddings import PODCAST_PATIENTS, find_transcript

CACHE_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EmbedCache")
KV_CACHE_PATH  = "/scratch/aniluchavez/ConvoDATAS/StaticEmbeds/wiki-news-300d-1M.kv"
MODEL_TAG      = "fasttext-wiki"

_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "can not", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "what's": "what is", "who's": "who is", "let's": "let us",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will",
    "they'll": "they will", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "we'd": "we would",
    "they'd": "they would",
}


def load_words(pid):
    tx = find_transcript(pid)
    df = pd.read_excel(tx, sheet_name="Sheet1")
    raw = df["text"].fillna("").astype(str)
    # Match convo's CleanedWord convention (lowercase, strip non-alphanumeric)
    # so podcast fasttext vectors are looked up consistently with every other model.
    return raw.str.lower().str.replace(r"[^a-z0-9]", "", regex=True).tolist()


def load_fasttext_kv():
    from gensim.models import KeyedVectors
    print(f"Loading cached binary KeyedVectors: {KV_CACHE_PATH}", flush=True)
    return KeyedVectors.load(KV_CACHE_PATH)


def _lookup(kv, w):
    if w in kv.key_to_index:
        return kv[w]
    lw = w.lower()
    if lw in kv.key_to_index:
        return kv[lw]
    return None


def word_vector(kv, word, dim):
    v = _lookup(kv, word)
    if v is not None:
        return v, True
    stripped = word.strip(string.punctuation)
    v = _lookup(kv, stripped)
    if v is not None:
        return v, True
    lw = stripped.lower()
    if lw in _CONTRACTIONS:
        parts = _CONTRACTIONS[lw].split()
        vecs = [_lookup(kv, p) for p in parts]
        vecs = [v for v in vecs if v is not None]
        if vecs:
            return np.mean(vecs, axis=0), True
    if "'" in stripped:
        stem = stripped.split("'")[0]
        v = _lookup(kv, stem)
        if v is not None:
            return v, True
    return np.zeros(dim, dtype=np.float32), False


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    kv = load_fasttext_kv()
    dim = kv.vector_size

    for pid in PODCAST_PATIENTS:
        out_path = os.path.join(CACHE_DIR, f"{pid}_{MODEL_TAG}_word_emb_layers.npy")
        if os.path.exists(out_path):
            print(f"[SKIP] {pid} -- already exists", flush=True)
            continue

        words = load_words(pid)
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
