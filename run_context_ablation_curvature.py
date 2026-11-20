"""Test whether LLM context-induced deflection predicts neural curvature.

This script builds a counterfactual hidden state for each word by rerunning the
LLM under an ablated context, then compares that state to the normal full-context
embedding cache used by the geometry-paper analyses.

Main question:

    Do words whose LLM representation depends more strongly on context also
    produce stronger hippocampal / ACC trajectory curvature?

Outputs are patient-level summaries plus a word table analogous to the clean
Figure 4 tables.
"""

from __future__ import annotations

# Keep model loading offline by default, matching the p6 embedding scripts.
import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, ttest_1samp
from transformers import AutoModel, AutoTokenizer

import run_clean_fig4_tests as clean
import run_geometry_analysis as geom


MODEL_CONFIGS = {
    "gpt2": {
        "hf_id": "gpt2",
        "max_ctx": 1024,
        "special_tokens": False,
    },
    "gpt2-medium": {
        "hf_id": "gpt2-medium",
        "max_ctx": 1024,
        "special_tokens": False,
    },
    "bert-base": {
        "hf_id": "bert-base-uncased",
        "max_ctx": 512,
        "special_tokens": True,
    },
    "roberta-base": {
        "hf_id": "roberta-base",
        "max_ctx": 512,
        "special_tokens": True,
    },
    "opt-350m": {
        "hf_id": "facebook/opt-350m",
        "max_ctx": 2048,
        "special_tokens": False,
    },
    "llama-2-7b": {
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "max_ctx": 4096,
        "special_tokens": False,
        "torch_dtype": "bfloat16",
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B",
        "max_ctx": 8192,
        "special_tokens": False,
        "torch_dtype": "bfloat16",
    },
    "gemma-2-9b": {
        "hf_id": "google/gemma-2-9b",
        "max_ctx": 8192,
        "special_tokens": False,
        "torch_dtype": "bfloat16",
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.3",
        "max_ctx": 8192,
        "special_tokens": False,
        "torch_dtype": "bfloat16",
    },
}


def sem(values: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(x[mask], y[mask])
    return float(r), float(p)


def cosine_deflection(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    denom = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    numer = np.sum(A * B, axis=1)
    out = np.full(A.shape[0], np.nan, dtype=float)
    valid = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1) & (denom > 0)
    out[valid] = 1.0 - np.clip(numer[valid] / denom[valid], -1.0, 1.0)
    return out


def l2_delta(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    valid = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    out = np.full(A.shape[0], np.nan, dtype=float)
    out[valid] = np.linalg.norm(A[valid] - B[valid], axis=1)
    return out


def trajectory_curvature_nan_safe(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 3:
        return np.zeros(0, dtype=float)
    out = np.full(X.shape[0] - 2, np.nan, dtype=float)
    valid_rows = np.isfinite(X).all(axis=1)
    for idx in range(len(out)):
        if not (valid_rows[idx] and valid_rows[idx + 1] and valid_rows[idx + 2]):
            continue
        v1 = X[idx + 1] - X[idx]
        v2 = X[idx + 2] - X[idx + 1]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        out[idx] = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))) if denom > 0 else 0.0
    return out


def load_layers(model: str, region: str) -> Dict[str, int]:
    path = Path(geom.RESULTS) / f"geometry_{model}_{region}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_geometry_analysis.py first.")
    with path.open() as f:
        records = json.load(f)
    return {rec["pid"]: int(rec["loo_layer_neu"]) for rec in records}


def load_model_and_tokenizer(model_tag: str, device: torch.device):
    cfg = MODEL_CONFIGS[model_tag]
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tokenizer.pad_token_id is None:
        fallback = tokenizer.eos_token or tokenizer.unk_token or tokenizer.cls_token
        if fallback is not None:
            tokenizer.pad_token = fallback

    load_kw = {}
    dtype_name = cfg.get("torch_dtype")
    if dtype_name and device.type == "cuda":
        load_kw["torch_dtype"] = getattr(torch, dtype_name)

    model = AutoModel.from_pretrained(cfg["hf_id"], **load_kw).to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.eval()
    return tokenizer, model


def word_token_ids(tokenizer, word: str) -> List[int]:
    unk = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    ids = tokenizer.encode(" " + str(word), add_special_tokens=False)
    return ids if ids else [unk]


def make_context_words(
    words: Sequence[str],
    word_idx: int,
    ablation: str,
    local_words: int,
    sent_id: Optional[np.ndarray],
    rng: np.random.Generator,
) -> List[str]:
    if ablation == "isolated":
        return []
    if ablation in {"local", "shuffle_prev"}:
        start = max(0, word_idx - local_words)
        context = list(words[start:word_idx])
        if ablation == "shuffle_prev" and len(context) > 1:
            context = [context[idx] for idx in rng.permutation(len(context))]
        return context
    if ablation == "sentence":
        if sent_id is None:
            return []
        sid = sent_id[word_idx]
        start = word_idx
        while start > 0 and sent_id[start - 1] == sid:
            start -= 1
        return list(words[start:word_idx])
    raise ValueError(f"Unknown ablation: {ablation}")


def build_prompt_content(
    tokenizer,
    words: Sequence[str],
    word_idx: int,
    ablation: str,
    local_words: int,
    sent_id: Optional[np.ndarray],
    content_per_window: int,
    rng: np.random.Generator,
) -> Tuple[List[int], Tuple[int, int]]:
    context_words = make_context_words(
        words=words,
        word_idx=word_idx,
        ablation=ablation,
        local_words=local_words,
        sent_id=sent_id,
        rng=rng,
    )
    context_ids: List[int] = []
    for word in context_words:
        context_ids.extend(word_token_ids(tokenizer, word))
    target_ids = word_token_ids(tokenizer, words[word_idx])

    max_context_tokens = max(0, content_per_window - len(target_ids))
    if len(context_ids) > max_context_tokens:
        context_ids = context_ids[-max_context_tokens:]

    target_start = len(context_ids)
    content_ids = context_ids + target_ids
    target_span = (target_start, target_start + len(target_ids))
    return content_ids, target_span


def batch_iter(items: Sequence[Tuple[List[int], Tuple[int, int]]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "outofmemory" in name
        or ("acceleratorerror" in name and "cuda" in text)
    )


def clear_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def format_model_oom_hint() -> str:
    return (
        "CUDA ran out of memory even for a single prompt. Try rerunning with "
        "`--batch-size 1`, a smaller `--local-words` value such as 8 or 16, "
        "one ablation at a time, or `--device cpu` if the GPU cannot fit the model."
    )


def encode_prompt_batch(
    seqs: Sequence[List[int]],
    spans: Sequence[Tuple[int, int]],
    pad_id: int,
    model,
    layer: int,
    embed_dim: int,
    device: torch.device,
) -> np.ndarray:
    max_len = max(len(seq) for seq in seqs)
    input_arr = np.full((len(seqs), max_len), pad_id, dtype=np.int64)
    attn_arr = np.zeros((len(seqs), max_len), dtype=np.int64)
    for row_idx, seq in enumerate(seqs):
        input_arr[row_idx, : len(seq)] = seq
        attn_arr[row_idx, : len(seq)] = 1

    input_ids = torch.tensor(input_arr, dtype=torch.long, device=device)
    attn = torch.tensor(attn_arr, dtype=torch.long, device=device)
    with torch.inference_mode():
        model_out = model(input_ids, attention_mask=attn, output_hidden_states=True)

    hidden_states = [
        hs
        for hs in model_out.hidden_states
        if hs.shape[-1] == embed_dim
    ]
    if layer >= len(hidden_states):
        raise ValueError(f"Requested layer {layer}, but model returned {len(hidden_states)} hidden states.")
    hidden = hidden_states[layer].detach().float().cpu().numpy()

    pooled = np.zeros((len(seqs), embed_dim), dtype=np.float32)
    for offset, (span_start, span_end) in enumerate(spans):
        pooled[offset] = hidden[offset, span_start:span_end].mean(axis=0)

    del input_ids, attn, model_out, hidden_states
    clear_accelerator_cache()
    return pooled


def compute_ablation_layer_embeddings(
    words: Sequence[str],
    tokenizer,
    model,
    layer: int,
    embed_dim: int,
    model_tag: str,
    ablation: str,
    local_words: int,
    sent_id: Optional[np.ndarray],
    batch_size: int,
    random_state: int,
    device: torch.device,
    max_words: Optional[int] = None,
) -> np.ndarray:
    cfg = MODEL_CONFIGS[model_tag]
    special_tokens = bool(cfg["special_tokens"])
    content_per_window = int(cfg["max_ctx"]) - (2 if special_tokens else 0)
    n_words = len(words) if max_words is None else min(len(words), int(max_words))
    out = np.full((len(words), embed_dim), np.nan, dtype=np.float32)
    rng = np.random.default_rng(random_state)

    prompts = [
        build_prompt_content(
            tokenizer=tokenizer,
            words=words,
            word_idx=word_idx,
            ablation=ablation,
            local_words=local_words,
            sent_id=sent_id,
            content_per_window=content_per_window,
            rng=rng,
        )
        for word_idx in range(n_words)
    ]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for start, batch in batch_iter(prompts, batch_size=batch_size):
        if start == 0 or start % 500 == 0:
            print(f"    embedding ablated words {start}-{min(start + len(batch), n_words) - 1} / {n_words}", flush=True)
        seqs = []
        spans = []
        for content_ids, target_span in batch:
            if special_tokens:
                input_ids = [tokenizer.cls_token_id] + content_ids + [tokenizer.sep_token_id]
                spans.append((target_span[0] + 1, target_span[1] + 1))
            else:
                input_ids = content_ids
                spans.append(target_span)
            seqs.append(input_ids)

        try:
            pooled = encode_prompt_batch(
                seqs=seqs,
                spans=spans,
                pad_id=pad_id,
                model=model,
                layer=layer,
                embed_dim=embed_dim,
                device=device,
            )
        except Exception as exc:
            if not is_cuda_oom(exc) or len(seqs) == 1:
                if is_cuda_oom(exc):
                    raise RuntimeError(format_model_oom_hint()) from exc
                raise
            print(
                f"    CUDA OOM at words {start}-{start + len(seqs) - 1}; "
                "retrying this batch one word at a time.",
                flush=True,
            )
            clear_accelerator_cache()
            pooled = np.zeros((len(seqs), embed_dim), dtype=np.float32)
            for offset, (seq, span) in enumerate(zip(seqs, spans)):
                try:
                    pooled[offset : offset + 1] = encode_prompt_batch(
                        seqs=[seq],
                        spans=[span],
                        pad_id=pad_id,
                        model=model,
                        layer=layer,
                        embed_dim=embed_dim,
                        device=device,
                    )
                except Exception as single_exc:
                    if is_cuda_oom(single_exc):
                        raise RuntimeError(format_model_oom_hint()) from single_exc
                    raise

        out[start : start + len(seqs)] = pooled

    return out


def ablation_cache_path(
    output_dir: Path,
    pid: str,
    model: str,
    ablation: str,
    local_words: int,
    layer: int,
    random_state: int,
) -> Path:
    tag = f"{ablation}_local{local_words}"
    if ablation == "shuffle_prev":
        tag = f"{tag}_seed{random_state}"
    return output_dir / "ablation_embedding_cache" / f"{pid}_{model}_{tag}_layer{layer}_word_emb.npy"


def get_ablation_embeddings(
    output_dir: Path,
    pid: str,
    model_tag: str,
    words: Sequence[str],
    tokenizer,
    model,
    layer: int,
    embed_dim: int,
    ablation: str,
    local_words: int,
    sent_id: Optional[np.ndarray],
    batch_size: int,
    random_state: int,
    device: torch.device,
    max_words: Optional[int],
    use_cache: bool,
) -> np.ndarray:
    cache_path = ablation_cache_path(output_dir, pid, model_tag, ablation, local_words, layer, random_state)
    if use_cache and max_words is None and cache_path.exists():
        return np.load(cache_path, mmap_mode="r")

    emb = compute_ablation_layer_embeddings(
        words=words,
        tokenizer=tokenizer,
        model=model,
        layer=layer,
        embed_dim=embed_dim,
        model_tag=model_tag,
        ablation=ablation,
        local_words=local_words,
        sent_id=sent_id,
        batch_size=batch_size,
        random_state=random_state,
        device=device,
        max_words=max_words,
    )
    if use_cache and max_words is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb


BASE_CONTROLS = [
    "surprisal",
    "sent_pos",
    "sentence_start",
    "turn_boundary",
    "is_speaker",
    "word_len",
    "log_freq",
    "dist_turn",
]


def build_patient_table(
    pid: str,
    region: str,
    model_tag: str,
    layer: int,
    ablation: str,
    local_words: int,
    data: Dict,
    full_layer: np.ndarray,
    ablated_layer: np.ndarray,
    deflection_lag: int,
    surprisal_lag: int,
) -> Optional[pd.DataFrame]:
    n = int(data["n_words"])
    Y_neu, n_pcs = geom.fit_neural_pca(data["neu_vecs"], n)
    if Y_neu is None:
        return None

    full_layer = np.asarray(full_layer[:n], dtype=np.float64)
    ablated_layer = np.asarray(ablated_layer[:n], dtype=np.float64)

    c_sem = geom.trajectory_curvature(full_layer)
    d_sem = clean.centered_displacement(full_layer)
    c_ablate = trajectory_curvature_nan_safe(ablated_layer)
    c_neu = geom.trajectory_curvature(Y_neu)
    d_neu = clean.centered_displacement(Y_neu)
    context_deflection = cosine_deflection(full_layer, ablated_layer)
    context_l2_delta = l2_delta(full_layer, ablated_layer)

    m = min(len(c_sem), len(d_sem), len(c_ablate), len(c_neu), len(d_neu))
    center_idx = np.arange(1, 1 + m)
    deflection_idx = center_idx + int(deflection_lag)
    surprisal_idx = center_idx + int(surprisal_lag)
    valid = (
        (deflection_idx >= 0)
        & (deflection_idx < n)
        & (surprisal_idx >= 0)
        & (surprisal_idx < n)
    )

    center_idx = center_idx[valid]
    deflection_idx = deflection_idx[valid]
    surprisal_idx = surprisal_idx[valid]

    sent_pos = np.asarray(data["sent_pos"])
    is_turn = np.asarray(data["is_turn"], dtype=bool)
    is_speaker = np.asarray(data["is_speaker"], dtype=bool)
    word_len = np.asarray(data["word_len"], dtype=float)
    log_freq = np.asarray(data["log_freq"], dtype=float)
    dist_turn = clean.distance_to_turn(is_turn)
    surp = np.asarray(data["surp"], dtype=float)

    return pd.DataFrame(
        {
            "pid": pid,
            "region": region,
            "model": model_tag,
            "loo_layer_neu": int(layer),
            "ablation": ablation,
            "local_words": int(local_words),
            "n_neural_pcs": int(n_pcs),
            "center_word_index": center_idx,
            "deflection_word_index": deflection_idx,
            "surprisal_word_index": surprisal_idx,
            "deflection_lag": int(deflection_lag),
            "surprisal_lag": int(surprisal_lag),
            "surprisal": surp[surprisal_idx],
            "context_deflection": context_deflection[deflection_idx],
            "context_l2_delta": context_l2_delta[deflection_idx],
            "c_sem": c_sem[:m][valid],
            "c_ablate": c_ablate[:m][valid],
            "delta_c_context": c_sem[:m][valid] - c_ablate[:m][valid],
            "d_sem": d_sem[:m][valid],
            "c_neu": c_neu[:m][valid],
            "d_neu": d_neu[:m][valid],
            "sent_pos": sent_pos[surprisal_idx].astype(float),
            "sentence_start": (sent_pos[surprisal_idx] == 0).astype(float),
            "near_sentence_start": (sent_pos[surprisal_idx] <= 1).astype(float),
            "turn_boundary": is_turn[surprisal_idx].astype(float),
            "turn_boundary_center": is_turn[center_idx].astype(float),
            "is_speaker": is_speaker[surprisal_idx].astype(float),
            "word_len": word_len[surprisal_idx],
            "log_freq": log_freq[surprisal_idx],
            "dist_turn": dist_turn[surprisal_idx],
        }
    )


def summarize_patient(table: pd.DataFrame) -> Dict[str, object]:
    base_predictors = {
        "c_sem": table["c_sem"],
        "d_sem": table["d_sem"],
        **{control: table[control] for control in BASE_CONTROLS},
    }
    m_base = clean.ols_clean(table["c_neu"], base_predictors)
    m_deflection = clean.ols_clean(
        table["c_neu"],
        {"context_deflection": table["context_deflection"], **base_predictors},
    )
    m_l2 = clean.ols_clean(
        table["c_neu"],
        {"context_l2_delta": table["context_l2_delta"], **base_predictors},
    )
    m_delta_curv = clean.ols_clean(
        table["c_neu"],
        {"delta_c_context": table["delta_c_context"], **base_predictors},
    )
    m_surp_deflection = clean.ols_clean(
        table["surprisal"],
        {
            "context_deflection": table["context_deflection"],
            "c_sem": table["c_sem"],
            "d_sem": table["d_sem"],
            **{control: table[control] for control in BASE_CONTROLS if control != "surprisal"},
        },
    )

    r_def_neu, p_def_neu = safe_pearson(table["context_deflection"], table["c_neu"])
    r_def_sem, p_def_sem = safe_pearson(table["context_deflection"], table["c_sem"])
    r_delta_neu, p_delta_neu = safe_pearson(table["delta_c_context"], table["c_neu"])

    def beta(model_dict, name):
        return float(model_dict["betas"].get(name, np.nan))

    def pval(model_dict, name):
        return float(model_dict["pvals"].get(name, np.nan))

    return {
        "pid": str(table["pid"].iloc[0]),
        "region": str(table["region"].iloc[0]),
        "model": str(table["model"].iloc[0]),
        "loo_layer_neu": int(table["loo_layer_neu"].iloc[0]),
        "ablation": str(table["ablation"].iloc[0]),
        "local_words": int(table["local_words"].iloc[0]),
        "deflection_lag": int(table["deflection_lag"].iloc[0]),
        "surprisal_lag": int(table["surprisal_lag"].iloc[0]),
        "n": int(len(table)),
        "n_finite_deflection": int(np.isfinite(table["context_deflection"]).sum()),
        "mean_context_deflection": float(np.nanmean(table["context_deflection"])),
        "mean_context_l2_delta": float(np.nanmean(table["context_l2_delta"])),
        "mean_delta_c_context": float(np.nanmean(table["delta_c_context"])),
        "r_deflection_cneu": r_def_neu,
        "p_deflection_cneu": p_def_neu,
        "r_deflection_csem": r_def_sem,
        "p_deflection_csem": p_def_sem,
        "r_delta_context_cneu": r_delta_neu,
        "p_delta_context_cneu": p_delta_neu,
        "r2_base_cneu_model": float(m_base["r2"]),
        "r2_deflection_cneu_model": float(m_deflection["r2"]),
        "delta_r2_deflection_cneu": float(m_deflection["r2"] - m_base["r2"]),
        "beta_deflection_to_cneu": beta(m_deflection, "context_deflection"),
        "p_deflection_to_cneu": pval(m_deflection, "context_deflection"),
        "r2_l2_cneu_model": float(m_l2["r2"]),
        "delta_r2_l2_cneu": float(m_l2["r2"] - m_base["r2"]),
        "beta_l2_to_cneu": beta(m_l2, "context_l2_delta"),
        "p_l2_to_cneu": pval(m_l2, "context_l2_delta"),
        "r2_delta_context_cneu_model": float(m_delta_curv["r2"]),
        "delta_r2_delta_context_cneu": float(m_delta_curv["r2"] - m_base["r2"]),
        "beta_delta_context_to_cneu": beta(m_delta_curv, "delta_c_context"),
        "p_delta_context_to_cneu": pval(m_delta_curv, "delta_c_context"),
        "r2_deflection_surp_model": float(m_surp_deflection["r2"]),
        "beta_deflection_to_surp": beta(m_surp_deflection, "context_deflection"),
        "p_deflection_to_surp": pval(m_surp_deflection, "context_deflection"),
    }


def group_summary(patient_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "r_deflection_cneu",
        "r_deflection_csem",
        "r_delta_context_cneu",
        "beta_deflection_to_cneu",
        "delta_r2_deflection_cneu",
        "beta_l2_to_cneu",
        "delta_r2_l2_cneu",
        "beta_delta_context_to_cneu",
        "delta_r2_delta_context_cneu",
        "beta_deflection_to_surp",
    ]
    rows: List[Dict[str, object]] = []
    for (region, ablation), group in patient_df.groupby(["region", "ablation"], dropna=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            if len(values) >= 2:
                test = ttest_1samp(values, popmean=0.0, nan_policy="omit")
                t_stat = float(test.statistic)
                p_value = float(test.pvalue)
            else:
                t_stat = p_value = float("nan")
            rows.append(
                {
                    "region": region,
                    "ablation": ablation,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sem": sem(values),
                    "median": float(np.median(values)),
                    "t_vs_zero": t_stat,
                    "p_vs_zero": p_value,
                }
            )
    return pd.DataFrame(rows)


def run(
    model_tag: str,
    regions: Iterable[str],
    patients: Optional[Iterable[str]],
    ablations: Iterable[str],
    local_words: int,
    batch_size: int,
    deflection_lag: int,
    surprisal_lag: int,
    max_words: Optional[int],
    output_dir: Path,
    random_state: int,
    device_name: str,
    use_cache: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_patients = set(patients) if patients else None
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer, lm = load_model_and_tokenizer(model_tag, device)

    patient_rows: List[Dict[str, object]] = []
    word_tables: List[pd.DataFrame] = []

    try:
        for region in regions:
            layers = load_layers(model_tag, region)
            for pid in geom.PATIENTS:
                if requested_patients is not None and pid not in requested_patients:
                    continue
                if region not in geom.PATIENTS[pid] or pid not in layers:
                    continue

                data = geom.load_patient(pid, geom.PATIENTS[pid], region, model_tag)
                if data is None or data.get("neu_vecs") is None:
                    continue

                n = int(data["n_words"])
                layer = int(layers[pid])
                emb_mm = np.load(data["emb_path"], mmap_mode="r")
                n_layers, _, embed_dim = emb_mm.shape
                if layer >= n_layers:
                    print(f"Skipping {pid} {region}: layer {layer} outside cache shape {emb_mm.shape}", flush=True)
                    continue
                full_layer = np.asarray(emb_mm[layer, :n], dtype=np.float32)
                del emb_mm

                words = [str(word) for word in data["words"][:n]]
                sent_id = np.asarray(data.get("sent_id", np.zeros(n, dtype=int)))[:n]

                print(
                    f"{model_tag} {region} {pid} layer={layer} n_words={n} embed_dim={embed_dim}",
                    flush=True,
                )
                for ablation in ablations:
                    print(f"  ablation={ablation} local_words={local_words}", flush=True)
                    ablated = get_ablation_embeddings(
                        output_dir=output_dir,
                        pid=pid,
                        model_tag=model_tag,
                        words=words,
                        tokenizer=tokenizer,
                        model=lm,
                        layer=layer,
                        embed_dim=int(embed_dim),
                        ablation=ablation,
                        local_words=local_words,
                        sent_id=sent_id,
                        batch_size=batch_size,
                        random_state=random_state,
                        device=device,
                        max_words=max_words,
                        use_cache=use_cache,
                    )
                    table = build_patient_table(
                        pid=pid,
                        region=region,
                        model_tag=model_tag,
                        layer=layer,
                        ablation=ablation,
                        local_words=local_words,
                        data=data,
                        full_layer=full_layer,
                        ablated_layer=ablated,
                        deflection_lag=deflection_lag,
                        surprisal_lag=surprisal_lag,
                    )
                    if table is None or table.empty:
                        continue
                    word_tables.append(table)
                    patient_rows.append(summarize_patient(table))
    finally:
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    patient_df = pd.DataFrame(patient_rows)
    word_df = pd.concat(word_tables, ignore_index=True) if word_tables else pd.DataFrame()
    group_df = group_summary(patient_df) if not patient_df.empty else pd.DataFrame()

    ablation_tag = "-".join(ablations)
    region_tag = "-".join(regions)
    patient_tag = "selected" if requested_patients is not None else "all-patients"
    max_tag = f"_maxwords{max_words}" if max_words is not None else ""
    lag_tag = (
        f"deflag{deflection_lag:+d}_surplag{surprisal_lag:+d}"
    ).replace("+", "plus").replace("-", "minus")
    stem = (
        f"context_ablation_{model_tag}_{region_tag}_{patient_tag}_{ablation_tag}"
        f"_local{local_words}_{lag_tag}{max_tag}"
    )

    patient_path = output_dir / f"{stem}_patient_summary.csv"
    word_path = output_dir / f"{stem}_word_table.csv"
    group_path = output_dir / f"{stem}_group_summary.csv"
    patient_df.to_csv(patient_path, index=False)
    word_df.to_csv(word_path, index=False)
    group_df.to_csv(group_path, index=False)

    print(f"\nSaved patient summary: {patient_path}")
    print(f"Saved word table:      {word_path}")
    print(f"Saved group summary:   {group_path}")
    return patient_df, word_df, group_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM context ablations and test context-induced deflection against neural curvature.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="llama-3.1-8b", choices=list(MODEL_CONFIGS))
    parser.add_argument("--regions", nargs="+", default=["hippocampus"], choices=geom.REGIONS)
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["shuffle_prev"],
        choices=["local", "shuffle_prev", "sentence", "isolated"],
    )
    parser.add_argument("--local-words", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--deflection-lag",
        type=int,
        default=0,
        help="Use context deflection at center_word + lag to predict neural curvature at center_word.",
    )
    parser.add_argument(
        "--surprisal-lag",
        type=int,
        default=1,
        help="Surprisal/control target index relative to curvature center.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Debug mode: only compute ablated embeddings for the first N words.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument(
        "--no-cache-embeddings",
        action="store_true",
        help="Do not cache selected-layer ablated embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(geom.RESULTS) / "context_ablation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    patient_df, _, group_df = run(
        model_tag=args.model,
        regions=args.regions,
        patients=args.patients,
        ablations=args.ablations,
        local_words=args.local_words,
        batch_size=args.batch_size,
        deflection_lag=args.deflection_lag,
        surprisal_lag=args.surprisal_lag,
        max_words=args.max_words,
        output_dir=args.output_dir,
        random_state=args.random_state,
        device_name=args.device,
        use_cache=not args.no_cache_embeddings,
    )
    if not patient_df.empty:
        print("\nPatient summary:")
        print(patient_df.to_string(index=False))
    if not group_df.empty:
        print("\nGroup summary:")
        print(group_df.to_string(index=False))


if __name__ == "__main__":
    main()
