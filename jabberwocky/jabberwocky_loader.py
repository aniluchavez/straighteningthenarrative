"""
jabberwocky_loader.py
======================
Loads word-level neural + surprisal data for the Jabberwocky dataset
(patient passively listening to jabberwocky sentences -- real function
words + nonce content words, preserved syntax), mirroring
podcast_loader.load_podcast_patient() closely enough that the same
geometry helpers (fit_neural_pca, trajectory_curvature, trajectory_displacement)
work unchanged on its output.

Differences from podcast:
  - Transcript is a flat xlsx per patient (text/onset/offset/Duration columns,
    same shape as podcast's), found directly in JABBERWOCKY_ROOT (no subdir).
  - Spikes are old-style key naming (chan/qual/spikes), filename lowercase
    "pt" prefix (ptYFx_taskN_spikes.mat) vs transcript's uppercase "PT".
  - Channel-region maps are reused from run_geometry_analysis.PATIENTS (same
    patient code, different task/session number -- electrodes don't move
    between sessions) or podcast's PODCAST_ONLY_REGIONS for patients with no
    convo recording (PTYFD).
  - No STATIC_MODELS handling needed yet (fasttext-wiki), only contextual
    LLM embeddings extracted via extract_jabberwocky_embeddings.py.
"""
from __future__ import annotations
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "podcast"))
from run_geometry_analysis import PATIENTS, ACCEPTED_QUAL, WIN_MS, LISTEN_OFFSET_MS, get_log_freq
from podcast_loader import PODCAST_ONLY_REGIONS, _sent_positions

JABBERWOCKY_ROOT = "/scratch/aniluchavez/semantic_syntactic_emb_project_for_Chavez/Jabberwocky"
EMBED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EmbedCache")
SURPRISAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Surprisal")
REGION = "hippocampus"

# pid -> (task_dir_code_for_files)
JABBERWOCKY_PATIENTS = [
    "PTYFC_task29", "PTYFD_task15", "PTYFF_task16", "PTYFG_task17",
    "PTYFI_task92", "PTYFK_task63", "PTYFP_task162",
]


def _short_code(pid):
    """PTYFC_task29 -> YFC (matches convo PATIENTS / podcast short-code convention)."""
    return pid.split("_")[0].upper()[2:]


def _convo_regions(pid):
    short = _short_code(pid)
    if short in PODCAST_ONLY_REGIONS:
        return PODCAST_ONLY_REGIONS[short]
    matches = [p for p in PATIENTS if p.split("_")[0].upper()[2:] == short]
    if len(matches) != 1:
        return None
    return PATIENTS[matches[0]]


def _find_one(pattern):
    cands = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith(("._", "~$"))]
    assert len(cands) == 1, f"expected exactly 1 match for {pattern}, found {cands}"
    return cands[0]


def find_transcript(pid):
    return _find_one(os.path.join(JABBERWOCKY_ROOT, f"{pid}_jabberwocky.xlsx"))


def find_spikes(pid):
    # filename uses lowercase "pt" prefix, transcript uses uppercase "PT"
    lower_pid = "pt" + pid[2:]
    return _find_one(os.path.join(JABBERWOCKY_ROOT, f"{lower_pid}_spikes.mat"))


def load_words(pid):
    df = pd.read_excel(find_transcript(pid), sheet_name="Sheet1")
    raw = df["text"].fillna("").astype(str)
    # Same cleaning convention as convo/podcast: lowercase, strip non-alphanumeric.
    return raw.str.lower().str.replace(r"[^a-z0-9]", "", regex=True).tolist()


def load_jabberwocky_patient(pid, region=REGION, model_tag=None, context_words=200):
    """Returns dict(n_words, surp, neu_vecs, emb_path, words, sent_pos, sent_id,
    is_turn, is_speaker, word_len, log_freq) or None if unavailable."""
    preg = _convo_regions(pid)
    if preg is None or region not in preg:
        return None

    df = pd.read_excel(find_transcript(pid), sheet_name="Sheet1")
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).reset_index(drop=True)
    n_words = len(df)

    raw = df["text"].fillna("").astype(str)
    words = raw.str.lower().str.replace(r"[^a-z0-9]", "", regex=True).tolist()

    sent_pos = _sent_positions(raw.tolist())
    sent_id = np.cumsum(sent_pos == 0) - 1
    is_turn = np.zeros(n_words, dtype=bool)       # no turn-taking (pure listening)
    is_speaker = np.zeros(n_words, dtype=bool)    # patient never speaks
    word_len = np.array([len(w) for w in words])
    log_freq = get_log_freq(words)

    surp_path = os.path.join(SURPRISAL_DIR, f"{pid}_surprisal.csv")
    if not os.path.exists(surp_path):
        return None
    surp_df = pd.read_csv(surp_path)
    if len(surp_df) < n_words:
        return None
    surp = surp_df["surprisal"].to_numpy()[:n_words]

    emb_path = None
    if model_tag is not None:
        ctx_tag = f"_ctx{context_words}"
        cache = os.path.join(EMBED_DIR, f"{pid}_{model_tag}{ctx_tag}_word_emb_layers.npy")
        if not os.path.exists(cache):
            return None
        emb_path = cache

    neu_vecs = None
    spikes_path = find_spikes(pid)
    with h5py.File(spikes_path, "r") as f:
        qual_key = "qual" if "qual" in f else "quality"
        chan_key = "chan" if "chan" in f else "channelIds"
        spikes = scipy.sparse.csr_matrix((
            f["spikes"]["data"][()], f["spikes"]["ir"][()], f["spikes"]["jc"][()]
        )).toarray()
        qual = np.asarray(f[qual_key][()]).ravel()
        chan = np.asarray(f[chan_key][()]).ravel()
    T = spikes.shape[0]
    neuron_idx = [
        idx for lo, hi in preg[region]
        for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()
    ]
    reg_sp = spikes[:, neuron_idx]

    vecs = []
    for wi in range(n_words):
        onset_ms = int(df.iloc[wi]["onset"])
        t0 = max(0, onset_ms + LISTEN_OFFSET_MS)
        t1 = min(T, t0 + WIN_MS)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    neu_vecs = np.array(vecs)

    return dict(
        n_words=n_words, surp=surp, neu_vecs=neu_vecs, emb_path=emb_path,
        words=words, sent_pos=sent_pos, sent_id=sent_id, is_turn=is_turn,
        is_speaker=is_speaker, word_len=word_len, log_freq=log_freq,
    )
