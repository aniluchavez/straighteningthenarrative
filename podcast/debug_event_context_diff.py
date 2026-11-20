"""
debug_event_context_diff.py
=============================
Checks whether "high-surprisal, unrestricted-frequency" events and
"high-surprisal, mid-frequency-band" events differ systematically in
conversational context -- specifically: distance to nearest turn boundary
(is_turn), and sentence position (sent_pos) -- for the convo dataset. If
these two event sets differ in typical turn-position, that would explain why
restricting to mid-frequency words shifts the NEURAL curvature peak lag, even
though nothing about the neural computation itself changed: the event SET
changed, and neural signals are known to be sensitive to turn-taking/speaker
transitions independent of lexical content.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PARENT_DIR, "results")
CONVO_MODEL = "llama-3.1-8b"
REGION      = "hippocampus"
HI_PCTILE = 75
MIDFREQ_LO_PCTILE = 45
MIDFREQ_HI_PCTILE = 55

sys.path.insert(0, PARENT_DIR)
from run_geometry_analysis import load_patient, PATIENTS


def dist_to_turn_boundary(is_turn, j, max_search=10):
    """Signed distance (in words) from word j to the nearest turn-boundary
    word (is_turn==True); negative = boundary is before j."""
    n = len(is_turn)
    for d in range(0, max_search + 1):
        if j - d >= 0 and is_turn[j - d]:
            return -d
        if j + d < n and is_turn[j + d]:
            return d
    return np.nan


def main():
    conv_path = os.path.join(RESULTS_DIR, f"geometry_{CONVO_MODEL}_{REGION}.json")
    cached = {r["pid"]: r["loo_layer_neu"] for r in json.load(open(conv_path))}

    unrestricted_dists, unrestricted_sentpos = [], []
    midfreq_dists, midfreq_sentpos = [], []
    n_unrestricted, n_midfreq = 0, 0

    for pid, preg in PATIENTS.items():
        if pid not in cached or REGION not in preg:
            continue
        data = load_patient(pid, preg, REGION, CONVO_MODEL)
        if data is None:
            continue
        n = data["n_words"]
        surp = data["surp"][:n]
        log_freq = data["log_freq"][:n]
        is_turn = data["is_turn"][:n] if "is_turn" in data else None
        sent_pos = data["sent_pos"][:n] if "sent_pos" in data else None
        if is_turn is None or sent_pos is None:
            print(f"  {pid}: missing is_turn/sent_pos, skipping")
            continue

        finite_surp = np.isfinite(surp)
        hi_thr_unres = np.nanpercentile(surp[finite_surp], HI_PCTILE)

        finite_freq = np.isfinite(log_freq)
        lo_edge = np.nanpercentile(log_freq[finite_freq], MIDFREQ_LO_PCTILE)
        hi_edge = np.nanpercentile(log_freq[finite_freq], MIDFREQ_HI_PCTILE)
        in_band = finite_freq & (log_freq >= lo_edge) & (log_freq <= hi_edge)
        surp_mf = np.where(in_band, surp, np.nan)
        finite_mf = np.isfinite(surp_mf)
        if finite_mf.sum() < 20:
            continue
        hi_thr_mf = np.nanpercentile(surp_mf[finite_mf], HI_PCTILE)

        for j in range(n):
            if np.isfinite(surp[j]) and surp[j] >= hi_thr_unres:
                d = dist_to_turn_boundary(is_turn, j)
                if np.isfinite(d):
                    unrestricted_dists.append(d)
                    unrestricted_sentpos.append(sent_pos[j])
                    n_unrestricted += 1
            if np.isfinite(surp_mf[j]) and surp_mf[j] >= hi_thr_mf:
                d = dist_to_turn_boundary(is_turn, j)
                if np.isfinite(d):
                    midfreq_dists.append(d)
                    midfreq_sentpos.append(sent_pos[j])
                    n_midfreq += 1
        print(f"  {pid}: done", flush=True)

    ud, md = np.array(unrestricted_dists), np.array(midfreq_dists)
    usp, msp = np.array(unrestricted_sentpos), np.array(midfreq_sentpos)

    print(f"\nUnrestricted high-surprisal events: n={n_unrestricted}")
    print(f"  mean dist to turn boundary = {ud.mean():.3f}  (sd={ud.std():.3f})")
    print(f"  mean sent_pos              = {usp.mean():.3f}  (sd={usp.std():.3f})")

    print(f"\nMid-freq-band high-surprisal events: n={n_midfreq}")
    print(f"  mean dist to turn boundary = {md.mean():.3f}  (sd={md.std():.3f})")
    print(f"  mean sent_pos              = {msp.mean():.3f}  (sd={msp.std():.3f})")

    from scipy.stats import ttest_ind
    t_d, p_d = ttest_ind(ud, md, equal_var=False, nan_policy="omit")
    t_s, p_s = ttest_ind(usp, msp, equal_var=False, nan_policy="omit")
    print(f"\nturn-boundary-distance difference: t={t_d:.3f}  p={p_d:.4g}")
    print(f"sent_pos difference:               t={t_s:.3f}  p={p_s:.4g}")


if __name__ == "__main__":
    main()
