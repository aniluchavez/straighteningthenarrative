"""
plot_trajectory_quantities_explainer.py
=======================================
Explainer figure: position, displacement, curvature of a word trajectory.

Top row (toy 2-D example):
  A position      x_t : a vector (arrow from the reference point to each word's point)
  B displacement  d_t = ||x_{t+1} - x_t|| : length of each step (one number per pair of words)
  C curvature     c_t = angle between step t-1->t and step t->t+1 (one number per 3 words)

Rows 2-3 (real data, one patient, N words):
  semantic: GPT-2 XL layer 29 raw activations (Hosseini & Fedorenko 2023), 1600-D
  neural  : hippocampus 2*sqrt(spike count) (Henaff/Goris 2021, sigma->0), N neurons
  left : path drawn in the first 2 PCs (centered, unscaled)
  right: over word index -> PC1-3 coordinates, position p_t = |x_t - centroid|,
         magnitude m_t = |x_t|, step lengths + symmetrised d_t, curvature c_t.
  argv[4] = "pc8": all quantities in 8 mean-subtracted PCs; position = |y_t| (distance from centroid).
         p, m, d, c computed in the FULL space, as in
         compute_curv_disp_pos_sqrt_full_gpt2xl_layer29.py.
No words shown.

    conda run -n gpt2_embed python3 -u plot_trajectory_quantities_explainer.py [PID] [START] [N]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
sys.path.insert(0, BASE_DIR)
from podcast_loader import (load_podcast_patient, _convo_regions,
                            load_words_onset_offset, windowed_neu_vecs)

MODEL, LAYER = "gpt2-xl", 29
FIG_DIR = os.path.join(BASE_DIR, "figures")
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
POS, DISP, CURV = "#2a78d6", "#eb6834", "#1baf7a"     # quantity colors, used consistently
PCS = ["#2a78d6", "#6da7ec", "#b7d3f6"]


def style(ax, grid=True):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if grid:
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def arrow(ax, p, q, color, lw=1.6, ms=10, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=3))


def angle_deg(u, v):
    return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def curv_disp(X):
    D = np.diff(X, axis=0)
    L = np.linalg.norm(D, axis=1)
    num = np.einsum("ij,ij->i", D[:-1], D[1:]); den = L[:-1] * L[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
    c[~(den > 0)] = np.nan
    return L, c


# ---------------------------------------------------------------- toy row
def toy_row(axA, axB, axC):
    P = np.array([[0.5, 0.6], [1.6, 1.4], [2.9, 1.6], [3.6, 2.9], [5.0, 2.6]])
    ref = np.array([0.0, 0.0])
    for ax in (axA, axB, axC):
        style(ax, grid=False)
        ax.set_xlim(-0.4, 5.8); ax.set_ylim(-0.4, 3.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("dimension 1", fontsize=8, color=INK2); ax.set_ylabel("dimension 2", fontsize=8, color=INK2)
        ax.plot(P[:, 0], P[:, 1], color=GRID, lw=1.2, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], s=40, color=INK, zorder=4)
        for i, p in enumerate(P):
            ax.annotate(f"word {i+1}", p, xytext=(-8, 8), textcoords="offset points", fontsize=8, color=INK)

    # A: position = vector from reference point
    axA.scatter(*ref, marker="+", s=80, color=INK, zorder=5)
    axA.annotate("reference\n(origin or centroid)", ref, xytext=(6, -2), textcoords="offset points",
                 fontsize=7, color=INK2, va="top")
    for p in P:
        arrow(axA, ref, p, POS, lw=1.2, ms=8)
    axA.annotate("x$_4$", P[3] * 0.6, xytext=(-14, 4), textcoords="offset points", color=POS, fontsize=11)
    axA.set_title("A  Position x$_t$ = a point (vector)\n    D numbers per word", loc="left", fontsize=10, color=INK)

    # B: displacement = step lengths
    for i in range(len(P) - 1):
        arrow(axB, P[i], P[i + 1], DISP, lw=2.0)
        m = (P[i] + P[i + 1]) / 2
        d = np.linalg.norm(P[i + 1] - P[i])
        axB.annotate(f"d$_{i+1}$={d:.1f}", m, xytext=(8, -14) if i < 3 else (-4, 10), textcoords="offset points", fontsize=8, color=INK)
    axB.set_title("B  Displacement d$_t$ = ‖x$_{t+1}$ − x$_t$‖\n    1 number per pair of words", loc="left", fontsize=10, color=INK)

    # C: curvature = turning angle between consecutive steps
    for i in range(len(P) - 1):
        arrow(axC, P[i], P[i + 1], INK2, lw=1.2, ms=8)
    for i in range(1, len(P) - 1):
        u, v = P[i] - P[i - 1], P[i + 1] - P[i]
        ext = P[i] + 0.55 * u / np.linalg.norm(u)          # straight-ahead continuation
        axC.plot([P[i][0], ext[0]], [P[i][1], ext[1]], ls="--", color=CURV, lw=1.2)
        a1 = np.degrees(np.arctan2(u[1], u[0])); a2 = np.degrees(np.arctan2(v[1], v[0]))
        lo, hi = sorted([a1, a2])
        if hi - lo > 180:
            lo, hi = hi, lo + 360
        axC.add_patch(Arc(P[i], 0.8, 0.8, theta1=lo, theta2=hi, color=CURV, lw=2))
        axC.annotate(f"c$_{i+1}$={angle_deg(u, v):.0f}°", P[i], xytext=[(10, -16), (12, -22), (14, -10)][i - 1],
                     textcoords="offset points", fontsize=8, color=INK)
    axC.set_title("C  Curvature c$_t$ = turn angle at word t\n    1 number per 3 words (0° = straight)",
                  loc="left", fontsize=10, color=INK)


# ---------------------------------------------------------------- real rows
def load(pid):
    if pid == "pseudopop":
        from compute_curv_disp_pos_sqrt_full_gpt2xl_layer29_pseudopop import load_pseudopop
        sem, K, _, _, patients, _ = load_pseudopop()
        return sem, 2.0 * np.sqrt(K), f"pseudopop ({len(patients)} patients, {K.shape[1]} units)"
    _, on, off = load_words_onset_offset(pid)
    data = load_podcast_patient(pid, region="hippocampus", model_tag=MODEL, context_words=200)
    n0 = min(data["n_words"], len(on))
    neu = windowed_neu_vecs(pid, "hippocampus", _convo_regions(pid), on[:n0], off[:n0], 80, 40, rate=False)
    emb = np.load(data["emb_path"], mmap_mode="r")
    n = min(n0, emb.shape[1], len(neu))
    sem = np.array(emb[LAYER, :n], dtype=np.float64)
    neu = 2.0 * np.sqrt(np.asarray(neu[:n], dtype=np.float64))
    return sem, neu, f"{pid}, {neu.shape[1]} neurons"


def real_row(fig, gs_row, X_full, sl, label, D, mag_note, pc8=False, raw=None, Yshow=None, raw_note="", logpos=None):
    sp = "8 PCs" if pc8 else f"full {D}-D"
    if pc8:                                                 # X_full = 8 centered PCs; Yshow = 20
        Y = Yshow[sl]
    else:
        Xc = X_full - X_full.mean(0)
        Y = PCA(n_components=8, svd_solver="randomized", random_state=0).fit_transform(Xc)[sl]
    Xs = X_full[sl]
    L, c = curv_disp(Xs)
    dsym = 0.5 * (L[:-1] + L[1:])                           # analysed displacement at word t
    pos = np.linalg.norm(Xs - X_full.mean(0), axis=1)       # |x_t - session centroid|
    mag = np.linalg.norm(Xs, axis=1)                        # |x_t|
    t = np.arange(1, len(Y) + 1)

    axP = fig.add_subplot(gs_row[:, 0]); style(axP)
    for i in range(len(Y) - 1):
        arrow(axP, Y[i, :2], Y[i + 1, :2], DISP, lw=1.4, ms=8)
    axP.scatter(Y[:, 0], Y[:, 1], s=30, color=INK, zorder=4)
    for i in range(len(Y)):
        axP.annotate(str(i + 1), Y[i, :2], xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
    axP.set_xlabel("PC1", fontsize=8, color=INK2); axP.set_ylabel("PC2", fontsize=8, color=INK2)
    axP.set_title(f"{label}\npath in PC1–PC2 (display only)",
                  loc="left", fontsize=9, color=INK)

    axX = fig.add_subplot(gs_row[0, 1:]); style(axX)
    nshow = Y.shape[1]
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("pcs", ["#0d366b", "#2a78d6", "#b7d3f6"])
    for j in range(nshow):
        axX.plot(t, Y[:, j], "-", lw=1.1, color=ramp(j / max(nshow - 1, 1)),
                 label=f"PC{j+1}" if j in (0, nshow - 1) else None)
    avg = Y.mean(1) if pc8 else Xs.mean(1)
    axX.plot(t, avg, "-o", ms=4, lw=2.4, color=INK, zorder=5,
             label=f"mean of the {nshow} PC coords" if pc8 else f"mean of all {D} coords")
    axX.set_ylabel("coords", fontsize=8, color=INK2)
    axX.legend(fontsize=6.5, frameon=False, ncol=5, loc="lower right", bbox_to_anchor=(1, 1.0))
    axX.set_title(f"x$_t$ in mean-subtracted PCs: first {nshow} shown (analysis uses PC1–8)" if pc8 else
                  f"x$_t$ = {D} numbers per word ({nshow} PCs shown)", loc="left", fontsize=9, color=INK)

    r = 1
    if pc8:
        Rs = raw[sl]; Dr = raw.shape[1]
        axR = fig.add_subplot(gs_row[r, 1:], sharex=axX); style(axR); r += 1
        axR.plot(t, Rs, "-", lw=0.4, color=INK2, alpha=0.15)
        axR.plot(t, Rs.mean(1), "-o", ms=4, lw=2.4, color=INK, zorder=5, label=f"mean of all {Dr} coords")
        axR.set_ylabel("raw coords", fontsize=8, color=INK2)
        axR.legend(fontsize=6.5, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0))
        axR.set_title(f"x$_t$ raw, no PCA: all {Dr} coords (gray)", loc="left", fontsize=9, color=INK)
    if not pc8:
        axS = fig.add_subplot(gs_row[r, 1:], sharex=axX); style(axS); r += 1
        axS.plot(t, pos, "-o", ms=3, lw=1.3, color=POS)
        axS.set_ylabel("‖x$_t$ − x̄‖", fontsize=8, color=INK2)
        axS.set_title(f"distance from session centroid ({sp})", loc="left", fontsize=9, color=INK)

    axM = fig.add_subplot(gs_row[r, 1:], sharex=axX); style(axM); r += 1
    axM.plot(t, mag, "-o", ms=3, lw=1.3, color=POS)
    axM.set_ylabel("‖y$_t$‖" if pc8 else "‖x$_t$‖", fontsize=8, color=INK2)
    axM.set_title("position p$_t$ = ‖y$_t$‖ = distance from session centroid (8 mean-subtracted PCs)" if pc8 else
                  f"position p$_t$ = magnitude = distance from origin ({sp}){mag_note}", loc="left", fontsize=9, color=INK)
    if logpos is not None:                                  # (session-wide position series, r with sqrt version)
        lp, rlog, logdesc, basedesc = logpos
        axL = fig.add_subplot(gs_row[r, 1:], sharex=axX); style(axL); r += 1
        axL.plot(t, lp[sl], "-o", ms=3, lw=1.3, color=POS, ls="--")
        axL.set_ylabel("‖y$_t$‖ log", fontsize=8, color=INK2)
        axL.set_title(f"position with {logdesc} (8 mean-subtracted PCs); "
                      f"r with {basedesc} position over all words = {rlog:.3f}", loc="left", fontsize=9, color=INK)
    if pc8:
        axMr = fig.add_subplot(gs_row[r, 1:], sharex=axX); style(axMr); r += 1
        axMr.plot(t, np.linalg.norm(Rs, axis=1), "-o", ms=3, lw=1.3, color=INK2)
        axMr.set_ylabel("‖x$_t$‖ raw", fontsize=8, color=INK2)
        axMr.set_title(f"not position: overall activity = distance from 0, raw, no PCA (full {Dr}-D){raw_note}", loc="left", fontsize=9, color=INK)

    axD = fig.add_subplot(gs_row[r, 1:], sharex=axX); style(axD)
    axD.bar(t[:-1] + 0.5, L, width=0.45, color=DISP, alpha=0.45, label="step ‖x$_{t+1}$ − x$_t$‖")
    axD.plot(t[1:-1], dsym, "o", ms=5, color=DISP, label="d$_t$ = mean of the two steps at word t")
    axD.set_ylabel("displacement", fontsize=8, color=INK2)
    axD.legend(fontsize=7, frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    axD.set_title(f"displacement ({sp})", loc="left", fontsize=9, color=INK)

    axC = fig.add_subplot(gs_row[r + 1, 1:], sharex=axX); style(axC)
    axC.plot(t[1:-1], c, "-o", ms=4, lw=1.3, color=CURV)
    axC.axhline(120, ls=":", color=INK2, lw=1)
    axC.text(t[-1] + 0.3, 120, "random\n≈120°", fontsize=7, color=INK2, va="center")
    axC.set_ylim(0, 180); axC.set_yticks([0, 60, 120, 180])
    axC.set_ylabel("c$_t$ (deg)", fontsize=8, color=INK2)
    axC.set_xlabel("word index t", fontsize=8, color=INK2)
    axC.set_title(f"curvature c$_t$ = turn angle at word t ({sp}); none at first/last word",
                  loc="left", fontsize=9, color=INK)
    axC.set_xticks(t)


def main():
    a = sys.argv[1:]
    pid = a[0] if a else "YEY"          # or "pseudopop" (word index = YEY-aligned common words)
    start = int(a[1]) if len(a) > 1 else 200
    N = int(a[2]) if len(a) > 2 else 12
    pc8 = len(a) > 3 and a[3] == "pc8"
    sem, neu, neu_desc = load(pid)
    sem_raw, neu_raw, sem20, neu20 = sem, neu, None, None
    if pc8:   # mean-subtracted, unscaled PCA fit on the whole session (|y_t| = distance from centroid)
        sem20 = PCA(n_components=20, svd_solver="full").fit_transform(sem)
        neu20 = PCA(n_components=20, svd_solver="full").fit_transform(neu)
        sem, neu = sem20[:, :8], neu20[:, :8]
        counts = (neu_raw / 2.0) ** 2                       # back from 2*sqrt(count)
        logpos = np.linalg.norm(PCA(n_components=8, svd_solver="full").fit_transform(np.log1p(counts)), axis=1)
        rlog = float(np.corrcoef(logpos, np.linalg.norm(neu, axis=1))[0, 1])
        slog = np.sign(sem_raw) * np.log1p(np.abs(sem_raw))   # signed log: activations can be negative
        slogpos = np.linalg.norm(PCA(n_components=8, svd_solver="full").fit_transform(slog), axis=1)
        rslog = float(np.corrcoef(slogpos, np.linalg.norm(sem, axis=1))[0, 1])
    nrow = 6 if pc8 else 5
    sl = slice(start, start + N)

    fig = plt.figure(figsize=(15, 29 if pc8 else 22), facecolor=BG)
    outer = fig.add_gridspec(3, 1, height_ratios=[0.75, 1.6 + 0.27 * pc8, 1.6 + 0.27 * pc8], hspace=0.2, top=0.95, bottom=0.03)
    top = outer[0].subgridspec(1, 3, wspace=0.15)
    toy_row(fig.add_subplot(top[0]), fig.add_subplot(top[1]), fig.add_subplot(top[2]))
    real_row(fig, outer[1].subgridspec(nrow + pc8, 3, hspace=0.75, wspace=0.3), sem, sl,
             f"Semantic: GPT-2 XL L{LAYER}, raw activations" + (" (1600-D → 8 PCs)" if pc8 else ""), 1600, "", pc8, sem_raw, sem20, "",
             (slogpos, rslog, "signed log sign(x)·log(1+|x|) instead of raw", "raw") if pc8 else None)
    real_row(fig, outer[2].subgridspec(nrow + pc8, 3, hspace=0.75, wspace=0.3), neu, sl,
             f"Neural: hippocampus {neu_desc}, 2√count" + (" (→ 8 PCs)" if pc8 else ""), neu_raw.shape[1], "" if pc8 else " = 2√(total count)", pc8, neu_raw, neu20, " = 2√(total count)", (logpos, rlog, "log(1+count) instead of 2√count", "2√count") if pc8 else None)
    fig.suptitle(f"Position, displacement, curvature — toy example (top), {pid} words {start}–{start+N-1} (bottom)" + (" — 8 mean-subtracted PCs" if pc8 else " — full space"),
                 fontsize=12, color=INK, x=0.01, ha="left", y=0.99)
    os.makedirs(FIG_DIR, exist_ok=True)
    base = os.path.join(FIG_DIR, f"trajectory_quantities_explainer_{pid}_w{start}_{N}" + ("_pc8" if pc8 else ""))
    fig.savefig(base + ".png", dpi=130, bbox_inches="tight"); fig.savefig(base + ".pdf", bbox_inches="tight")
    print("saved", base + ".png")


if __name__ == "__main__":
    main()
