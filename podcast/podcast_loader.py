"""
podcast_loader.py
==================
Loads word-level neural + surprisal data for the podcast_comparison dataset
(single-narrator podcast listening, not conversation), mirroring
run_geometry_analysis.load_patient() closely enough that the same geometry
helpers (fit_neural_pca, trajectory_curvature, trajectory_displacement) work
unchanged on its output.

Differences from the convo loader:
  - Transcript has no speaker columns (podcast = patient always listening),
    so every word uses the LISTEN_OFFSET_MS branch (no is_speaker check).
  - Spikes are read directly from the .mat (chan/qual/spikes), same as
    convo's old-style key naming -- restricted to each patient's
    hippocampal channel ranges taken from run_geometry_analysis.PATIENTS
    (same channel maps as the convo recordings for the same patient).
  - Surprisal is precomputed here (compute_podcast_surprisal.py, gpt2-large)
    rather than reused from ConvoDATAS/Surprisal.
"""
from __future__ import annotations
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_geometry_analysis import PATIENTS, ACCEPTED_QUAL, WIN_MS, LISTEN_OFFSET_MS, get_log_freq

CHAVEZ_DIR = "/scratch/aniluchavez/semantic_syntactic_emb_project_for_Chavez"

# Patients whose podcast text+spikes live outside PODCAST_ROOT, in bespoke
# per-patient formats (2026-09-07, per user-supplied source paths). Each of
# these still uses the standard PODCAST_ONLY_REGIONS/AP_ONLY_REGIONS channel
# maps below for hippocampal restriction -- only the raw text/spike *source*
# differs from the podcast_comparison/{short}/{transcript,spikes}/ layout.
EXTRA_SOURCES = {
    # YEX words 0-2205 (podcast stories 1-2) are on the task-96 clock, but
    # ptYEX_82neurons.mat is task 99 then task 96 stitched (join ~2698 s) and
    # the offset is unconfirmed -> NaN (held out), as in the semantics/syntax
    # paper (semantic_syntactic_emb_project_for_Chavez/ConvoData/
    # build_podcast_counts.py, YEX_HELD_OUT). Added 2026-11-11.
    "YEX": dict(kind="wordfr_mat", mat_key="yex_word", held_out_words=np.arange(0, 2206),
                spikes_path=os.path.join(CHAVEZ_DIR, "podcast_mats_partial", "ptYEX_82neurons.mat")),
    "YFB": dict(kind="wordfr_mat", mat_key="yfb_word",
                spikes_path=os.path.join(CHAVEZ_DIR, "podcast_mats_partial", "ptYFB.mat")),
    # YFX transcript onsets are on a different clock from the spikes: the
    # recording is silent for its first ~574 s, and cross-correlating the
    # recorded audio envelope with transcript word timing gives
    # recording_time = transcript_time + 574.9 s (2026-10-21).
    "YFX": dict(kind="xlsx",
                tx_path=os.path.join(CHAVEZ_DIR, "PTYFX_EMU-0072_podcast.xlsx"),
                spikes_path=os.path.join(CHAVEZ_DIR, "ptYFX_task72_spikes.mat"),
                clock_offset_ms=574900.0),
    "YFU": dict(kind="selfcontained_mat",
                path=os.path.join(CHAVEZ_DIR, "YFU_bundles", "PTYFU_podcast.mat")),
}
_WORDFR_MAT_PATH = os.path.join(CHAVEZ_DIR, "HPC_YEXthurYFP_WordFRMatrix_Duration80msDelay.mat")

PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"
SURPRISAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Surprisal")
EMBED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EmbedCache")
REGION = "hippocampus"

# Static (non-contextual) embedders -- cache file has no "_ctx200" suffix,
# same convention as run_geometry_analysis.STATIC_MODELS.
STATIC_MODELS = {"fasttext-wiki"}


def _sent_positions(words):
    """Position within sentence, resetting after '.', '?', '!' -- podcast has
    a single 'text' word column (no speaker columns like convo), so this
    mirrors run_geometry_analysis.get_sent_positions() but reads straight
    off the word list."""
    sent_pos = np.zeros(len(words), dtype=int)
    pos = 0
    for i, w in enumerate(words):
        sent_pos[i] = pos
        pos = 0 if str(w).strip().endswith((".", "?", "!")) else pos + 1
    return sent_pos

PODCAST_PATIENTS = ["YEY", "YEZ", "YFA", "YFC", "YFD", "YFF", "YFG", "YFI", "YFK",
                     "YEX", "YFB", "YFX", "YFU"]

# Podcast-only patients with no matching convo recording (no entry in
# run_geometry_analysis.PATIENTS) -- channel ranges supplied directly.
# YEX/YFB/YFX/YFU added 2026-09-04: raw spikes+transcript live outside
# PODCAST_ROOT (see podcast_mats_partial / HPC_YEXthurYFP_WordFRMatrix in
# semantic_syntactic_emb_project_for_Chavez) -- no per-patient PODCAST_ROOT
# folder or GPT2 embeddings/surprisal exist for them yet.
PODCAST_ONLY_REGIONS = {
    "YFD": {"hippocampus": [(1, 8), (17, 24)]},
    "YEX": {"hippocampus": [(1, 8), (33, 40), (41, 48)]},
    "YFB": {"hippocampus": [(1, 16), (25, 40)]},
    "YFX": {"hippocampus": [(49, 64)]},
    "YFU": {"hippocampus": [(17, 32), (41, 56)]},
}

# Anterior/posterior split within the above hippocampus ranges (MNI152 y<-21
# = posterior), for the AP curvature analysis. YEX has no projectworlds
# anatomy on file -- ant/post assigned directly per user specification
# (2026-09-04). YFB's 4 bundles are all y>-21 (-12.0,-12.4,-12.5,-18.1) so it
# contributes anterior units only. YFU's mRT2dHb bundle (49-56) has 0 units
# in ptYFU_timefiles_spikes.mat (chan max=48) despite being anatomically
# posterior -- excluded from post here since it yields no data.
AP_ONLY_REGIONS = {
    "YEX": {"ant": [(1, 8), (33, 40)], "post": [(41, 48)]},
    "YFB": {"ant": [(1, 8), (9, 16), (25, 32), (33, 40)], "post": []},
    "YFX": {"ant": [(57, 64)], "post": [(49, 56)]},
    "YFU": {"ant": [(17, 24), (41, 48)], "post": [(25, 32)]},
}


def _convo_regions(short):
    """Find the matching PTxxx_taskNNN key in run_geometry_analysis.PATIENTS for this short code.
    Falls back to PODCAST_ONLY_REGIONS; returns None if neither has a channel map."""
    if short in PODCAST_ONLY_REGIONS:
        return PODCAST_ONLY_REGIONS[short]
    matches = [pid for pid in PATIENTS if pid.split("_")[0].upper()[2:] == short]
    if len(matches) != 1:
        return None
    return PATIENTS[matches[0]]


def _find_one(pattern):
    cands = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith(("._", "~$"))]
    assert len(cands) == 1, f"expected exactly 1 match for {pattern}, found {cands}"
    return cands[0]


def _load_words_extra(short):
    """words_raw (list[str]) + onset_ms (float array) for the 4 EXTRA_SOURCES
    patients, whose transcripts don't live under PODCAST_ROOT."""
    words_raw, onset_ms, _ = _load_words_offsets_extra(short)
    return words_raw, onset_ms


def _load_words_offsets_extra(short):
    """words_raw + onset_ms + offset_ms for the 4 EXTRA_SOURCES patients."""
    src = EXTRA_SOURCES[short]
    if src["kind"] == "wordfr_mat":
        m = scipy.io.loadmat(_WORDFR_MAT_PATH, simplify_cells=True)
        entries = m[src["mat_key"]]
        words_raw = [str(e["text"]) for e in entries]
        onset_ms = np.array([float(e["onset"]) for e in entries])
        offset_ms = np.array([float(e["offset"]) for e in entries])
        held = src.get("held_out_words")
        if held is not None:
            onset_ms[held] = np.nan
            offset_ms[held] = np.nan
        return words_raw, onset_ms, offset_ms
    if src["kind"] == "xlsx":
        df = pd.read_excel(src["tx_path"])
        words_raw = df["words"].astype(str).tolist()
        onset_ms = pd.to_numeric(df["onset"], errors="coerce").values + src.get("clock_offset_ms", 0.0)
        offset_ms = pd.to_numeric(df["offset"], errors="coerce").values + src.get("clock_offset_ms", 0.0)
        return words_raw, onset_ms, offset_ms
    if src["kind"] == "selfcontained_mat":
        m = scipy.io.loadmat(src["path"], simplify_cells=True)
        words_raw = [str(w) for w in m["words"]]
        onset_ms = np.asarray(m["word_onset_ms"], dtype=float).ravel()
        offset_ms = np.asarray(m["word_offset_ms"], dtype=float).ravel()
        return words_raw, onset_ms, offset_ms
    raise ValueError(f"unknown EXTRA_SOURCES kind for {short}")


def load_words_onset_offset(short):
    """words_raw + onset_ms + offset_ms for ANY podcast patient (standard 9
    under PODCAST_ROOT, or the 4 EXTRA_SOURCES). Unified entry point for
    duration-window (onset+pad/offset+pad) neural windowing."""
    if short in EXTRA_SOURCES:
        return _load_words_offsets_extra(short)
    pdir = os.path.join(PODCAST_ROOT, short)
    tx_path = _find_one(os.path.join(pdir, "transcript", "*.xlsx"))
    df = pd.read_excel(tx_path, sheet_name="Sheet1")
    words_raw = df["text"].astype(str).tolist()
    onset_ms = pd.to_numeric(df["onset"], errors="coerce").values
    offset_ms = pd.to_numeric(df["offset"], errors="coerce").values if "offset" in df.columns else np.full(len(df), np.nan)
    return words_raw, onset_ms, offset_ms


def _dense_spike_source(mat_path, region, preg):
    with h5py.File(mat_path, "r") as f:
        spikes = scipy.sparse.csr_matrix((
            f["spikes"]["data"][()], f["spikes"]["ir"][()], f["spikes"]["jc"][()]
        )).toarray()
        chan_key = "chan" if "chan" in f else "channelIds"
        qual_key = "qual" if "qual" in f else "quality"
        qual = np.asarray(f[qual_key][()]).ravel()
        chan = np.asarray(f[chan_key][()]).ravel()
    T = spikes.shape[0]
    neuron_idx = [
        idx for lo, hi in preg[region]
        for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()
    ]
    if len(neuron_idx) < 3:
        return None
    return ("dense", spikes[:, neuron_idx], T)


def _units_spike_source(path, region, preg):
    m = scipy.io.loadmat(path, simplify_cells=True)
    unit_chan = np.asarray(m["unit_chan"], dtype=float).ravel()
    unit_quality = np.asarray(m["unit_quality"], dtype=float).ravel()
    spike_times = m["spike_times_ms"]
    neuron_idx = [
        idx for lo, hi in preg[region]
        for idx in np.where((unit_chan >= lo) & (unit_chan <= hi) & np.isin(unit_quality, ACCEPTED_QUAL))[0].tolist()
    ]
    if len(neuron_idx) < 3:
        return None
    return ("units", [np.sort(np.asarray(spike_times[idx], dtype=float).ravel()) for idx in neuron_idx])


def windowed_neu_vecs(short, region, preg, onset_ms, offset_ms, pad_onset_ms, pad_offset_ms, rate=False):
    """neu_vecs using a per-word [onset+pad_onset_ms, offset+pad_offset_ms]
    window (falls back to onset+100ms when offset is missing/invalid), for
    ANY podcast patient -- standard 9 (dense chan/qual/spikes under
    PODCAST_ROOT) or the 4 EXTRA_SOURCES (dense for YEX/YFB/YFX, per-unit
    spike-time lists for YFU). rate=True divides by window duration (s)
    instead of returning raw spike counts."""
    n_words = len(onset_ms)
    if short in EXTRA_SOURCES:
        src = EXTRA_SOURCES[short]
        if src["kind"] in ("wordfr_mat", "xlsx"):
            res = _dense_spike_source(src["spikes_path"], region, preg)
        else:
            res = _units_spike_source(src["path"], region, preg)
    else:
        pdir = os.path.join(PODCAST_ROOT, short)
        mat_path = _find_one(os.path.join(pdir, "spikes", "*_spikes.mat"))
        res = _dense_spike_source(mat_path, region, preg)
    if res is None:
        return None

    if res[0] == "dense":
        _, reg_sp, T = res
        n_ch = reg_sp.shape[1]
        vecs = np.zeros((n_words, n_ch))
        for wi in range(n_words):
            if not np.isfinite(onset_ms[wi]):      # missing / held-out word -> NaN (was: windowed at t=0)
                vecs[wi] = np.nan
                continue
            o = onset_ms[wi]
            of = offset_ms[wi] if (np.isfinite(offset_ms[wi]) and offset_ms[wi] > o) else o + 100
            t0 = max(0, int(o + pad_onset_ms))
            t1 = min(T, int(of + pad_offset_ms))
            if t1 > t0:
                seg = reg_sp[t0:t1].sum(axis=0).astype(np.float64)
                vecs[wi] = seg / ((t1 - t0) / 1000.0) if rate else seg
    else:
        times_sorted = res[1]
        n_ch = len(times_sorted)
        vecs = np.zeros((n_words, n_ch))
        for wi in range(n_words):
            if not np.isfinite(onset_ms[wi]):      # missing / held-out word -> NaN (was: windowed at t=0)
                vecs[wi] = np.nan
                continue
            o = onset_ms[wi]
            of = offset_ms[wi] if (np.isfinite(offset_ms[wi]) and offset_ms[wi] > o) else o + 100
            t0, t1 = o + pad_onset_ms, of + pad_offset_ms
            if t1 > t0:
                for ci, ts in enumerate(times_sorted):
                    lo_i = np.searchsorted(ts, t0, side="left")
                    hi_i = np.searchsorted(ts, t1, side="left")
                    cnt = hi_i - lo_i
                    vecs[wi, ci] = cnt / ((t1 - t0) / 1000.0) if rate else cnt
    return vecs


def _load_spikes_extra(short, region, preg):
    """Windowed neu_vecs for the 4 EXTRA_SOURCES patients. YEX/YFB/YFX use the
    same chan/qual/sparse-spikes .mat layout as the standard loader (just a
    different file path); YFU is self-contained with per-unit spike time
    lists instead of a dense channel x time matrix, so it needs its own
    windowing. Returns (neu_vecs_builder) -- a function(onset_ms, n_words)
    -> neu_vecs array, or None if too few qualifying units."""
    src = EXTRA_SOURCES[short]
    if src["kind"] in ("wordfr_mat", "xlsx"):
        with h5py.File(src["spikes_path"], "r") as f:
            spikes = scipy.sparse.csr_matrix((
                f["spikes"]["data"][()], f["spikes"]["ir"][()], f["spikes"]["jc"][()]
            )).toarray()
            chan_key = "chan" if "chan" in f else "channelIds"
            qual_key = "qual" if "qual" in f else "quality"
            qual = np.asarray(f[qual_key][()]).ravel()
            chan = np.asarray(f[chan_key][()]).ravel()
        T = spikes.shape[0]
        neuron_idx = [
            idx for lo, hi in preg[region]
            for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()
        ]
        if len(neuron_idx) < 3:
            return None
        reg_sp = spikes[:, neuron_idx]

        def build(onset_ms, n_words):
            vecs = []
            for wi in range(n_words):
                o = int(onset_ms[wi]) if np.isfinite(onset_ms[wi]) else 0
                t0 = max(0, o + LISTEN_OFFSET_MS)
                t1 = min(T, t0 + WIN_MS)
                vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
            return np.array(vecs)
        return build

    if src["kind"] == "selfcontained_mat":
        m = scipy.io.loadmat(src["path"], simplify_cells=True)
        unit_chan = np.asarray(m["unit_chan"], dtype=float).ravel()
        unit_quality = np.asarray(m["unit_quality"], dtype=float).ravel()
        spike_times = m["spike_times_ms"]
        neuron_idx = [
            idx for lo, hi in preg[region]
            for idx in np.where((unit_chan >= lo) & (unit_chan <= hi) & np.isin(unit_quality, ACCEPTED_QUAL))[0].tolist()
        ]
        if len(neuron_idx) < 3:
            return None
        times_sorted = [np.sort(np.asarray(spike_times[idx], dtype=float).ravel()) for idx in neuron_idx]

        def build(onset_ms, n_words):
            vecs = np.zeros((n_words, len(neuron_idx)), dtype=np.float64)
            for wi in range(n_words):
                o = onset_ms[wi] if np.isfinite(onset_ms[wi]) else 0.0
                t0, t1 = o + LISTEN_OFFSET_MS, o + LISTEN_OFFSET_MS + WIN_MS
                for ci, ts in enumerate(times_sorted):
                    lo_i = np.searchsorted(ts, t0, side="left")
                    hi_i = np.searchsorted(ts, t1, side="left")
                    vecs[wi, ci] = hi_i - lo_i
            return vecs
        return build

    raise ValueError(f"unknown EXTRA_SOURCES kind for {short}")


def load_podcast_patient(short, region=REGION, model_tag=None, context_words=200):
    """Returns dict(n_words, surp, neu_vecs, emb_path, words, sent_pos, sent_id,
    is_turn, is_speaker, word_len, log_freq) or None if unavailable.

    model_tag: which cached embedding to point emb_path at. None (default)
    keeps the original lab-provided word_embeddings_gpt2.npy (words-first
    axis order, use with care -- see make_convo_podcast_curvature_comparison.py's
    earlier bug). Any tag matching an EmbedCache file
    ({short}_{tag}_ctx{context_words}_word_emb_layers.npy, or no ctx suffix for
    STATIC_MODELS) uses that instead -- (n_layers, n_words, dim) axis order,
    same convention as convo.
    context_words: which ctx-sweep cache to use (default 200, matching convo).
    Ignored for STATIC_MODELS (fasttext-wiki has no context).
    """
    preg = _convo_regions(short)
    if preg is None or region not in preg:
        return None

    extra = short in EXTRA_SOURCES
    if model_tag is None:
        if extra:
            return None  # lab-provided raw gpt2 embeddings only exist under PODCAST_ROOT
        pdir = os.path.join(PODCAST_ROOT, short)
        emb_path = os.path.join(pdir, "embeddings", "word_embeddings_gpt2.npy")
        emb_words_first = True
    else:
        suffix = "" if model_tag in STATIC_MODELS else f"_ctx{context_words}"
        emb_path = os.path.join(EMBED_DIR, f"{short}_{model_tag}{suffix}_word_emb_layers.npy")
        emb_words_first = False
    surp_path = os.path.join(SURPRISAL_DIR, f"{short}_surprisal.csv")
    if not (os.path.exists(emb_path) and os.path.exists(surp_path)):
        return None

    if extra:
        words_raw_full, onset_full = _load_words_extra(short)
        n_words = len(words_raw_full)
    else:
        pdir = os.path.join(PODCAST_ROOT, short)
        tx_path = _find_one(os.path.join(pdir, "transcript", "*.xlsx"))
        df = pd.read_excel(tx_path, sheet_name="Sheet1")
        df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
        n_words = len(df)

    emb_shape = np.load(emb_path, mmap_mode="r").shape
    n_words_emb = emb_shape[0] if emb_words_first else emb_shape[1]
    surp_df = pd.read_csv(surp_path)
    n_words = min(n_words, n_words_emb, len(surp_df))

    surp = np.full(n_words, np.nan)
    for _, row in surp_df.iloc[:n_words].iterrows():
        wi = int(row["RowIndex"])
        if wi < n_words:
            surp[wi] = float(row["surprisal"])

    if extra:
        words_raw = np.array(words_raw_full[:n_words])
        onsets = onset_full[:n_words]
    else:
        words_raw = df["text"].astype(str).values[:n_words]
        onsets = df["onset"].values[:n_words]
    # Sentence boundaries need the raw punctuation; compute before cleaning.
    sent_pos = _sent_positions(words_raw)
    sent_id = (sent_pos == 0).astype(int).cumsum() - 1
    is_turn = np.zeros(n_words, dtype=bool)      # single narrator -- no speaker turns
    is_speaker = np.zeros(n_words, dtype=bool)   # patient never speaks in podcast

    # Match convo's CleanedWord convention: lowercase, strip all non-alphanumeric
    # characters. Must match extract_podcast_embeddings.py's load_words() exactly,
    # since embeddings were generated from that cleaned text, not words_raw.
    words = np.array([re.sub(r"[^a-z0-9]", "", w.lower()) for w in words_raw])
    word_len = np.array([len(w) for w in words], dtype=float)
    log_freq = get_log_freq(words)

    neu_vecs = None
    if extra:
        builder = _load_spikes_extra(short, region, preg)
        if builder is not None:
            neu_vecs = builder(onsets, n_words)
    else:
        mat_path = _find_one(os.path.join(pdir, "spikes", "*_spikes.mat"))
        with h5py.File(mat_path, "r") as f:
            spikes = scipy.sparse.csr_matrix((
                f["spikes"]["data"][()], f["spikes"]["ir"][()], f["spikes"]["jc"][()]
            )).toarray()
            # newer sort-pipeline .mat files (e.g. YFU) use channelIds/quality
            # instead of chan/qual -- same sparse spikes/{data,ir,jc} layout.
            chan_key = "chan" if "chan" in f else "channelIds"
            qual_key = "qual" if "qual" in f else "quality"
            qual = np.asarray(f[qual_key][()]).ravel()
            chan = np.asarray(f[chan_key][()]).ravel()
        T = spikes.shape[0]
        neuron_idx = [
            idx for lo, hi in preg[region]
            for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()
        ]
        if len(neuron_idx) >= 3:
            reg_sp = spikes[:, neuron_idx]
            vecs = []
            for wi in range(n_words):
                onset_ms = int(onsets[wi]) if np.isfinite(onsets[wi]) else 0
                t0 = max(0, onset_ms + LISTEN_OFFSET_MS)
                t1 = min(T, t0 + WIN_MS)
                vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
            neu_vecs = np.array(vecs)

    return dict(n_words=n_words, surp=surp, neu_vecs=neu_vecs, emb_path=emb_path, words=words,
                sent_pos=sent_pos, sent_id=sent_id, is_turn=is_turn, is_speaker=is_speaker,
                word_len=word_len, log_freq=log_freq)


if __name__ == "__main__":
    for short in PODCAST_PATIENTS:
        d = load_podcast_patient(short)
        if d is None:
            print(f"{short}: SKIP")
        else:
            neu = d["neu_vecs"]
            print(f"{short}: n_words={d['n_words']}  neu_vecs={None if neu is None else neu.shape}")
