"""
compute_convo_layer_profile.py
=================================
Convo version of jabberwocky/compute_jabberwocky_window_sweep.py's LOO-layer
profile step, but fixed at a SMALL window (WIN_MS=300ms, near the
curvature-preferring end of the WIN_MS sweep found earlier this session)
instead of sweeping. For each model, per layer, pooled across all 14 convo
patients (orig10 + new4, excluding PTYFU whose transcript is broken):
  - mean semantic curvature / displacement
  - r(c_sem_layer, c_neu), r(d_sem_layer, d_neu) -- pooled leave-one-out mean,
    same LOO-selection logic as run_geometry_analysis.py / jabberwocky.

Matches the format plotted in the Jabberwocky Layer Profile artifact, so the
same chart can be rebuilt for convo at a small window.
"""
import sys, os, json, numpy as np, h5py, scipy.sparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import pearsonr
from run_geometry_analysis import (
    build_paths, PATIENTS, ACCEPTED_QUAL, SPEAK_OFFSET_MS, LISTEN_OFFSET_MS,
    trajectory_curvature, trajectory_displacement, build_valid_mask, fit_neural_pca,
    load_patient,
)

WIN_MS = 300
MODELS = ["gpt2-medium", "llama-3.1-8b"]
REGION = "hippocampus"

ALL14 = ["PTYEU_task147", "PTYEV_task37", "PTYEY_task86", "PTYEZ_task60", "PTYFA_task25",
         "PTYFC_task28", "PTYFF_task17", "PTYFG_task18", "PTYFI_task81", "PTYFK_task40",
         "PTYFM_task104", "PTYFP_task88", "PTYFR_task91", "PTYFS_task95"]


def build_neu_vecs(pid, model_for_meta):
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
        t1 = min(T, t0 + WIN_MS)
        vecs.append(reg_sp[t0:t1].sum(axis=0).astype(np.float64) if t1 > t0 else np.zeros(len(neuron_idx)))
    return np.array(vecs), n, data0


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                             "convo_layer_profile_win300.json")
    all_results = {}
    for model_tag in MODELS:
        print(f"\n{'#'*70}\nMODEL: {model_tag}  WIN_MS={WIN_MS}\n{'#'*70}")
        per_patient_curv, per_patient_disp = {}, {}
        per_patient_rcurv, per_patient_rdisp = {}, {}
        n_layers = None
        for pid in ALL14:
            neu_vecs, n, data0 = build_neu_vecs(pid, model_tag)
            if neu_vecs is None:
                print(f"  {pid}: SKIP (no data)")
                continue
            valid = build_valid_mask(n, data0["surp"][:n], data0["sent_pos"][:n], data0["is_turn"][:n])
            if valid.sum() < 50:
                print(f"  {pid}: SKIP (too few valid words)")
                continue
            Y_neu, n_pcs = fit_neural_pca(neu_vecs, n)
            if Y_neu is None:
                print(f"  {pid}: SKIP (PCA failed)")
                continue
            c_neu = trajectory_curvature(Y_neu)[valid]
            d_neu = trajectory_displacement(Y_neu)[:n - 2][valid]

            emb_mm = np.load(data0["emb_path"], mmap_mode="r")
            n_layers = emb_mm.shape[0]
            mean_curv, mean_disp, r_curv, r_disp = [], [], [], []
            for li in range(n_layers):
                emb_l = np.array(emb_mm[li, :n], dtype=np.float64)
                c_l = trajectory_curvature(emb_l)[valid]
                d_l = trajectory_displacement(emb_l)[:n - 2][valid]
                mean_curv.append(float(c_l.mean()))
                mean_disp.append(float(d_l.mean()))
                rc, _ = pearsonr(c_l, c_neu)
                rd, _ = pearsonr(d_l, d_neu)
                r_curv.append(float(rc))
                r_disp.append(float(rd))
            del emb_mm
            per_patient_curv[pid] = mean_curv
            per_patient_disp[pid] = mean_disp
            per_patient_rcurv[pid] = r_curv
            per_patient_rdisp[pid] = r_disp
            print(f"  {pid}: n={int(valid.sum())}  n_pcs={n_pcs}  n_layers={n_layers}")

        pids = list(per_patient_curv.keys())
        # LOO layer (leave-one-out mean of r_curv, same convention as jabberwocky/main pipeline)
        loo_layer = {}
        for pid in pids:
            others = [p for p in pids if p != pid]
            rc_mat = np.array([per_patient_rcurv[p] for p in others], dtype=float)
            rc_mean = np.nanmean(rc_mat, axis=0)
            loo_layer[pid] = int(np.nanargmax(np.abs(rc_mean))) if not np.all(np.isnan(rc_mean)) else 0
        print(f"\n  LOO layers: {loo_layer}")

        pooled_curv = np.nanmean(np.array([per_patient_curv[p] for p in pids]), axis=0)
        pooled_disp = np.nanmean(np.array([per_patient_disp[p] for p in pids]), axis=0)
        pooled_rcurv = np.nanmean(np.array([per_patient_rcurv[p] for p in pids]), axis=0)
        pooled_rdisp = np.nanmean(np.array([per_patient_rdisp[p] for p in pids]), axis=0)

        print("layer  mean_curv  mean_disp  r_curv   r_disp")
        for li in range(n_layers):
            print(f"  {li:3d}   {pooled_curv[li]:8.3f}  {pooled_disp[li]:9.3f}  {pooled_rcurv[li]:+.4f}  {pooled_rdisp[li]:+.4f}")

        all_results[model_tag] = dict(
            n_layers=n_layers, win_ms=WIN_MS, loo_layer=loo_layer,
            mean_curv=pooled_curv.tolist(), mean_disp=pooled_disp.tolist(),
            r_curv=pooled_rcurv.tolist(), r_disp=pooled_rdisp.tolist(),
        )
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
