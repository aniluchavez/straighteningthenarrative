"""
compute_curvature_wins_grid.py
=================================
For every archived model, sweep WIN_MS and, at each window, scan every layer
of that model, computing raw (unnulled) Pearson r(c_sem_layer, c_neu) and
r(d_sem_layer, d_neu) pooled (mean of per-patient r) across all convo
patients. Reports which (window, layer) cells have curvature beating
displacement.

Scope note: this is raw correlation comparison, NOT permutation-null
z-scored significance (the full z-scored grid across 11 models x 5 windows x
up to 33 layers x 3 datasets would require an impractical amount of compute
in one pass) -- treat "wins" as "which one the raw correlation favors",
not "which one is statistically significant".

Usage:
    conda run -n gpt2_embed python3 -u compute_curvature_wins_grid.py
"""
import sys, os, json, numpy as np, h5py, scipy.sparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import pearsonr
from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)

WIN_SIZES = [100, 300, 500, 1000, 2000]
REGION = "hippocampus"

MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "bert-base", "roberta-base",
          "opt-350m", "llama-2-7b", "llama-3.1-8b", "mistral-7b", "gemma-2-9b"]

ALL14 = ["PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60", "PTYFA_task25",
         "PTYFC_task28", "PTYFF_task17", "PTYFG_task18", "PTYFI_task81", "PTYFK_task40",
         "PTYFM_task104", "PTYFP_task88", "PTYFR_task91", "PTYFS_task95"]


def build_neu_vecs(pid, win_ms, model_for_meta):
    paths = build_paths(pid, model_for_meta)
    preg = PATIENTS[pid]
    data0 = load_patient(pid, preg, REGION, model_for_meta)
    if data0 is None:
        return None, None, None
    n = data0["n_words"]

    with h5py.File(paths["spikes"], "r") as f:
        qual_key = "qual" if "qual" in f else "quality"
        chan_key = "chan" if "chan" in f else "channelIds"
        spikes = scipy.sparse.csr_matrix((
            f["spikes"]["data"][()], f["spikes"]["ir"][()], f["spikes"]["jc"][()]
        )).toarray()
        qual = np.asarray(f[qual_key][()]).ravel()
        chan = np.asarray(f[chan_key][()]).ravel()
    T = spikes.shape[0]
    neuron_idx = [
        idx for lo, hi in preg[REGION]
        for idx in np.where((chan >= lo) & (chan <= hi) & np.isin(qual, ACCEPTED_QUAL))[0].tolist()
    ]
    reg_sp = spikes[:, neuron_idx]

    xl = pd.ExcelFile(paths["xlsx"])
    sheet = "Sheet1" if "Sheet1" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(paths["xlsx"], sheet_name=sheet, keep_default_na=False)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset"]).reset_index(drop=True)
    n = min(n, len(df))
    is_spk1 = df["Speaker1"].astype(str).str.strip() != ""

    vecs = []
    for wi in range(n):
        onset_ms = int(df.iloc[wi]["onset"])
        t0 = max(0, onset_ms - SPEAK_OFFSET_MS) if is_spk1.iloc[wi] else max(0, onset_ms + LISTEN_OFFSET_MS)
        t1 = min(T, t0 + win_ms)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n, data0


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                             "convo_curvature_wins_grid.json")
    all_results = {}
    for model_tag in MODELS:
        print(f"\n{'#'*70}\nMODEL: {model_tag}\n{'#'*70}", flush=True)
        model_results = {}
        for win_ms in WIN_SIZES:
            per_patient_rcurv, per_patient_rdisp = {}, {}
            n_layers = None
            for pid in ALL14:
                neu_vecs, n, data0 = build_neu_vecs(pid, win_ms, model_tag)
                if neu_vecs is None:
                    continue
                valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
                if valid.sum() < 50:
                    continue
                Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
                if Y_neu is None:
                    continue
                c_neu = trajectory_curvature(Y_neu)[valid]
                d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

                emb_mm = np.load(data0["emb_path"], mmap_mode="r")
                n_layers = emb_mm.shape[0]
                r_curv, r_disp = [], []
                for li in range(n_layers):
                    emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
                    c_l = trajectory_curvature(emb_l)[valid]
                    d_l = trajectory_displacement(emb_l)[:n - 2][valid]
                    rc, _ = pearsonr(c_l, c_neu)
                    rd, _ = pearsonr(d_l, d_neu)
                    r_curv.append(float(rc))
                    r_disp.append(float(rd))
                del emb_mm
                per_patient_rcurv[pid] = r_curv
                per_patient_rdisp[pid] = r_disp

            pids = list(per_patient_rcurv.keys())
            if not pids or n_layers is None:
                print(f"  win={win_ms}: SKIP (no usable patients)", flush=True)
                continue
            pooled_rcurv = np.nanmean(np.array([per_patient_rcurv[p] for p in pids]), axis=0)
            pooled_rdisp = np.nanmean(np.array([per_patient_rdisp[p] for p in pids]), axis=0)
            winning_layers = [li for li in range(n_layers) if pooled_rcurv[li] > pooled_rdisp[li]]
            model_results[win_ms] = dict(
                n_layers=n_layers, n_patients=len(pids),
                r_curv=pooled_rcurv.tolist(), r_disp=pooled_rdisp.tolist(),
                curv_wins_layers=winning_layers,
            )
            frac = len(winning_layers) / n_layers
            print(f"  win={win_ms:5d}  n_pat={len(pids):2d}  n_layers={n_layers:3d}  "
                  f"curv wins at {len(winning_layers)}/{n_layers} layers ({100*frac:.0f}%)  "
                  f"layers={winning_layers}", flush=True)

        all_results[model_tag] = model_results
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
