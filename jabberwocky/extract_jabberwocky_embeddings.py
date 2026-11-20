"""
extract_jabberwocky_embeddings.py
====================================
Jabberwocky-dataset version of podcast/extract_podcast_embeddings.py:
fixed causal context window of `--context_words` preceding words + current
word, extracted at the FIRST subword of the current word, all hidden-state
layers, for the "top models" (llama-3.1-8b, gpt2-medium) per this
investigation's convention.

Differences from the podcast script:
  - Reads words from Jabberwocky/{pid}_jabberwocky.xlsx directly ("text"
    column), no subdirectory.
  - Saves to jabberwocky/EmbedCache/{pid}_{tag}_ctx{N}_word_emb_layers.npy.

Usage:
    cd .../geometry_paper/jabberwocky
    conda run -n gpt2_embed --no-capture-output python3 -u extract_jabberwocky_embeddings.py \
        --context_words 200 --max_tokens 512
"""
import os, sys, gc, glob, argparse
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")
os.environ.setdefault("HF_HUB_OFFLINE",        "1")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

JABBERWOCKY_ROOT = "/scratch/aniluchavez/semantic_syntactic_emb_project_for_Chavez/Jabberwocky"
CACHE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EmbedCache")
BATCH_SIZE   = 16

JABBERWOCKY_PATIENTS = [
    "PTYFC_task29", "PTYFD_task15", "PTYFF_task16", "PTYFG_task17",
    "PTYFI_task92", "PTYFK_task63", "PTYFP_task162",
]

MODELS = [
    {"tag": "gpt2",            "hf_id": "gpt2"},
    {"tag": "gpt2-medium",     "hf_id": "gpt2-medium"},
    {"tag": "gpt2-large",      "hf_id": "gpt2-large"},
    {"tag": "gpt2-xl",         "hf_id": "openai-community/gpt2-xl"},
    {"tag": "bert-base",       "hf_id": "bert-base-uncased"},
    {"tag": "roberta-base",    "hf_id": "roberta-base"},
    {"tag": "opt-350m",        "hf_id": "facebook/opt-350m"},
    {"tag": "llama-2-7b",      "hf_id": "meta-llama/Llama-2-7b-hf",   "bfloat16": True},
    {"tag": "llama-3.1-8b",    "hf_id": "meta-llama/Meta-Llama-3.1-8B", "bfloat16": True},
    {"tag": "mistral-7b",      "hf_id": "mistralai/Mistral-7B-v0.3",  "bfloat16": True},
    {"tag": "gemma-2-9b",      "hf_id": "google/gemma-2-9b",          "bfloat16": True},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cli_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patients", type=str, default=None)
    p.add_argument("--models", type=str, default=None)
    p.add_argument("--context_words", type=int, default=200)
    p.add_argument("--context_tag", type=str, default=None)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def find_transcript(pid):
    cands = [f for f in glob.glob(os.path.join(JABBERWOCKY_ROOT, f"{pid}_jabberwocky.xlsx"))
             if not os.path.basename(f).startswith(("._", "~$"))]
    assert len(cands) == 1, f"{pid}: expected exactly 1 transcript xlsx, found {cands}"
    return cands[0]


def load_words(pid):
    tx = find_transcript(pid)
    df = pd.read_excel(tx, sheet_name="Sheet1")
    raw = df["text"].fillna("").astype(str)
    return raw.str.lower().str.replace(r"[^a-z0-9]", "", regex=True).tolist()


def load_model(cfg):
    hf_id = cfg["hf_id"]
    bf16 = cfg.get("bfloat16", False)
    dtype_kw = {"torch_dtype": torch.bfloat16} if bf16 else {}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(hf_id, **dtype_kw)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def get_n_layers_embed_dim(model):
    cfg = model.config
    embed_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    probe = torch.zeros((1, 2), dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        out = model(probe, output_hidden_states=True)
    n_layers = sum(1 for hs in out.hidden_states if hs.shape[-1] == embed_dim)
    return n_layers, embed_dim


def _detect_trailing_special(tokenizer):
    probe_ids = tokenizer("test", add_special_tokens=True)["input_ids"]
    if len(probe_ids) < 2:
        return 0
    last_id = probe_ids[-1]
    special_ids = {tokenizer.sep_token_id, tokenizer.eos_token_id} - {None}
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id == tokenizer.bos_token_id:
        special_ids.discard(tokenizer.eos_token_id)
    return 1 if last_id in special_ids else 0


@torch.no_grad()
def embed_patient(words, tokenizer, model, n_layers, embed_dim, trailing_special,
                  context_words, max_tokens):
    n_words = len(words)
    word_emb = np.zeros((n_layers, n_words, embed_dim), dtype=np.float32)

    for batch_start in range(0, n_words, BATCH_SIZE):
        batch_idx = list(range(batch_start, min(batch_start + BATCH_SIZE, n_words)))

        context_strings = []
        for t in batch_idx:
            ctx_start = max(0, t - context_words)
            context_strings.append(" ".join(words[ctx_start:t + 1]))

        tokens = tokenizer(
            context_strings, return_tensors="pt", truncation=True,
            max_length=max_tokens, padding=True,
        ).to(device)

        real_lens = tokens["attention_mask"].sum(dim=1).tolist()

        word_strs = [(" " if t > 0 else "") + words[t] for t in batch_idx]
        word_n_toks = [len(tokenizer(w, add_special_tokens=False)["input_ids"]) for w in word_strs]
        first_idxs = [max(0, int(rl) - trailing_special - wn) for rl, wn in zip(real_lens, word_n_toks)]

        out = model(**tokens, output_hidden_states=True)
        hs_list = [hs for hs in out.hidden_states if hs.shape[-1] == embed_dim]

        for li, hs in enumerate(hs_list):
            for j, (t, idx) in enumerate(zip(batch_idx, first_idxs)):
                word_emb[li, t] = hs[j, idx, :].float().cpu().numpy()

        del out, tokens
        torch.cuda.empty_cache()

    return word_emb


def main():
    args = _cli_args()
    patients = args.patients.split(",") if args.patients else JABBERWOCKY_PATIENTS
    model_tags = set(args.models.split(",")) if args.models else None
    models = [m for m in MODELS if model_tags is None or m["tag"] in model_tags]
    context_tag = args.context_tag or f"_ctx{args.context_words}"

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name}  {p.total_memory // 1024**2} MB VRAM")

    for model_cfg in models:
        tag = model_cfg["tag"]
        missing = [p for p in patients
                   if args.overwrite or not os.path.exists(
                       os.path.join(CACHE_DIR, f"{p}_{tag}{context_tag}_word_emb_layers.npy"))]
        if not missing:
            print(f"\n{tag}: all requested patients cached -- skipping")
            continue

        print(f"\n{'#'*70}\nMODEL: {tag}  ({model_cfg['hf_id']})")
        print(f"context_words={args.context_words}  context_tag={context_tag}  max_tokens={args.max_tokens}")
        print(f"Missing: {len(missing)} patients: {missing}\n{'#'*70}")

        print("Loading model...", flush=True)
        tokenizer, lm = load_model(model_cfg)
        n_layers, embed_dim = get_n_layers_embed_dim(lm)
        trailing_special = _detect_trailing_special(tokenizer)
        print(f"  n_layers={n_layers}  embed_dim={embed_dim}  trailing_special={trailing_special}", flush=True)

        for pid in missing:
            cache = os.path.join(CACHE_DIR, f"{pid}_{tag}{context_tag}_word_emb_layers.npy")
            print(f"\n  {pid}...", flush=True)
            words = load_words(pid)
            print(f"    {len(words)} words, context<={args.context_words} words, token cap<={args.max_tokens}",
                  flush=True)
            emb = embed_patient(words, tokenizer, lm, n_layers, embed_dim, trailing_special,
                                args.context_words, args.max_tokens)
            np.save(cache, emb)
            print(f"    saved {cache}  shape={emb.shape}", flush=True)

        del lm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n{tag} done.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
