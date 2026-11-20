"""
extract_tinyllama_all13.py
====================================
Extract TinyLlama-1.1B (RoPE architecture, small scale) embeddings for all 13
podcast patients -- disentangling "RoPE positional encoding" from "large
model scale" as the explanation for why LLaMA/Mistral/Gemma favor
displacement over curvature while GPT-2/BERT/RoBERTa/OPT favor curvature.
"""
import sys, os
sys.path.insert(0, ".")
import extract_podcast_embeddings as base

ORIGINAL9 = ["YEY", "YEZ", "YFA", "YFC", "YFD", "YFF", "YFG", "YFI", "YFK"]
NEW4 = ["YEX", "YFB", "YFX", "YFU"]
ALL13 = ORIGINAL9 + NEW4

# Capture the TRUE original load_words before importing extract_podcast_embeddings_new4,
# whose import has the side effect of patching base.load_words itself.
_orig_load_words = base.load_words
import extract_podcast_embeddings_new4 as new4mod

def load_words_all13(pid):
    if pid in NEW4:
        return new4mod.load_words_new4(pid)
    return _orig_load_words(pid)

base.load_words = load_words_all13
base.PODCAST_PATIENTS = ALL13
base.MODELS = [{"tag": "tinyllama-1.1b", "hf_id": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"}]

if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    base.main()
