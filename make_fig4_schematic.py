"""
make_fig4_schematic.py
======================
Schematic explaining the regression setup for Figure 4 panels A–D.
Each panel is self-contained with its own equation.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES  = os.path.join(BASE_DIR, "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "font.size": 11,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "svg.fonttype": "none",
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
})

C_CURV  = "#D95319"
C_SEM   = "#4472C4"
C_NEU   = "#2CA02C"
C_SURP  = "#7B4F9E"
C_CTRL  = "#888888"
C_BG    = "#FAFAFA"
C_EDGE  = "#BBBBBB"


def rounded_box(ax, cx, cy, w, h, label, sub=None,
                fc="white", ec="#999999", lw=1.2,
                fs=10, sub_fs=8.5, bold=False):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.025",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(rect)
    dy = 0.016 if sub else 0
    ax.text(cx, cy + dy, label,
            ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=5)
    if sub:
        ax.text(cx, cy - dy * 1.8, sub,
                ha="center", va="center", fontsize=sub_fs,
                color="#555555", style="italic", zorder=5)


def arr(ax, x0, y0, x1, y1, color="#444444", lw=1.5, label=None):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=11),
        zorder=4,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.005, my, label, ha="left", va="center",
                fontsize=8, color=color, zorder=5)


def panel_frame(ax, x, y, w, h, letter, title, title_color="#222222"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        facecolor=C_BG, edgecolor=C_EDGE, linewidth=1.3, zorder=1,
    )
    ax.add_patch(rect)
    ax.text(x + 0.018, y + h - 0.016, letter,
            ha="left", va="top", fontsize=15, fontweight="bold",
            color="#111111", zorder=6)
    ax.text(x + w / 2, y + h + 0.005, title,
            ha="center", va="bottom", fontsize=10,
            color=title_color, style="italic", zorder=6)


def equation_text(ax, cx, y, text, fs=8.0, color="#333333"):
    ax.text(cx, y, text, ha="center", va="top",
            fontsize=fs, color=color,
            bbox=dict(facecolor="#EFEFEF", edgecolor="none",
                      boxstyle="round,pad=0.25", alpha=0.85),
            zorder=6)


# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

fig.text(0.5, 0.97, "Figure 4 — Regression setup",
         ha="center", va="top", fontsize=14, fontweight="bold")

# ── source banner ─────────────────────────────────────────────────────────────
ax.text(0.5, 0.925, "Inputs derived word-by-word from each recording session:",
        ha="center", va="center", fontsize=9, color="#444444", style="italic")

for cx, label, sub, fc, ec in [
    (0.18, "$c_{sem}$, $d_{sem}$", "LLM trajectory (layer L)", "#EAF0FF", C_SEM),
    (0.50, "controls",
     "sent. pos · sent. start · turn boundary\nspeaker role · word length · log freq · dist-to-turn",
     "#F5F5F5", C_CTRL),
    (0.82, "$c_{neu}$, $d_{neu}$", "Neural trajectory (hippocampus PCA)", "#E8F5E9", C_NEU),
]:
    rounded_box(ax, cx, 0.875, 0.27, 0.067, label, sub,
                fc=fc, ec=ec, lw=1.4, fs=10.5, sub_fs=7.8)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A  (top-left)
# ─────────────────────────────────────────────────────────────────────────────
panel_frame(ax, 0.01, 0.50, 0.47, 0.32, "A",
            "Cross-space geometry alignment")

# semantic features
rounded_box(ax, 0.11, 0.735, 0.13, 0.065, "$c_{sem}$", "semantic\ncurvature",
            fc="#FFF3EE", ec=C_CURV, fs=11)
rounded_box(ax, 0.27, 0.735, 0.13, 0.065, "$d_{sem}$", "semantic\ndisplacement",
            fc="#EAF0FF", ec=C_SEM, fs=11)

# neural features
rounded_box(ax, 0.11, 0.595, 0.13, 0.065, "$c_{neu}$", "neural\ncurvature",
            fc="#FFF3EE", ec=C_CURV, fs=11)
rounded_box(ax, 0.27, 0.595, 0.13, 0.065, "$d_{neu}$", "neural\ndisplacement",
            fc="#E8F5E9", ec=C_NEU, fs=11)

# double-headed "correlation" arrows
for cx in [0.11, 0.27]:
    ax.annotate("", xy=(cx, 0.628), xytext=(cx, 0.702),
                arrowprops=dict(arrowstyle="<|-|>", color="#555555",
                                lw=1.4, mutation_scale=10), zorder=4)

ax.text(0.19, 0.665, "$r$ = ?", ha="center", va="center",
        fontsize=9, color="#444444")

equation_text(ax, 0.245, 0.543,
              r"$r(c_{sem},\, c_{neu})$   and   $r(d_{sem},\, d_{neu})$"
              "\n(Pearson correlation, per patient)",
              fs=8.2)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B  (top-right)
# ─────────────────────────────────────────────────────────────────────────────
panel_frame(ax, 0.52, 0.50, 0.47, 0.32, "B",
            "Controlled neural curvature model")

rounded_box(ax, 0.615, 0.735, 0.11, 0.062, "$c_{sem}$", fc="#FFF3EE", ec=C_CURV, fs=11)
rounded_box(ax, 0.735, 0.735, 0.11, 0.062, "$d_{sem}$", fc="#EAF0FF", ec=C_SEM, fs=11)
rounded_box(ax, 0.855, 0.735, 0.10, 0.062, "surprisal", fc="#F3EEF9", ec=C_SURP, fs=9.5)
rounded_box(ax, 0.675, 0.590, 0.14, 0.065, "$c_{neu}$", "outcome",
            fc="#FFF3EE", ec=C_CURV, lw=2.0, fs=11, bold=True)

arr(ax, 0.615, 0.704, 0.648, 0.622, color=C_CURV, label="$\\beta_{c_{sem}}$?")
arr(ax, 0.735, 0.704, 0.690, 0.622, color=C_SEM,  label="$\\beta_{d_{sem}}$?")
arr(ax, 0.855, 0.704, 0.710, 0.622, color=C_SURP, label="$\\beta_{surp}$")

ax.text(0.862, 0.640, "+ other\ncontrols", ha="center", va="center",
        fontsize=7.5, color=C_CTRL, style="italic")

equation_text(ax, 0.735, 0.543,
              r"$c_{neu} \sim \beta_1 c_{sem} + \beta_2 d_{sem} + \beta_3\,\mathrm{surprisal} + \mathbf{controls} + \varepsilon$",
              fs=8.0)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL C  (bottom-left)
# ─────────────────────────────────────────────────────────────────────────────
panel_frame(ax, 0.01, 0.10, 0.47, 0.32, "C",
            "Semantic geometry → surprisal")

rounded_box(ax, 0.11, 0.335, 0.13, 0.065, "$c_{sem}$", fc="#FFF3EE", ec=C_CURV, fs=11)
rounded_box(ax, 0.27, 0.335, 0.13, 0.065, "$d_{sem}$", fc="#EAF0FF", ec=C_SEM, fs=11)
rounded_box(ax, 0.19, 0.185, 0.20, 0.068,
            "surprisal$_{t+1}$", "outcome",
            fc="#F3EEF9", ec=C_SURP, lw=2.0, fs=10.5, bold=True)

arr(ax, 0.11, 0.302, 0.155, 0.219, color=C_CURV, label="$\\beta_{c_{sem}}$?")
arr(ax, 0.27, 0.302, 0.225, 0.219, color=C_SEM,  label="$\\beta_{d_{sem}}$?")

ax.text(0.390, 0.260, "+ controls\n(no surprisal)", ha="center", va="center",
        fontsize=7.5, color=C_CTRL, style="italic")

equation_text(ax, 0.245, 0.143,
              r"$\mathrm{surprisal}_{t+1} \sim \beta_1 c_{sem} + \beta_2 d_{sem} + \mathbf{controls} + \varepsilon$",
              fs=8.0)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL D  (bottom-right)
# ─────────────────────────────────────────────────────────────────────────────
panel_frame(ax, 0.52, 0.10, 0.47, 0.32, "D",
            "Neural geometry → surprisal")

rounded_box(ax, 0.615, 0.335, 0.13, 0.065, "$c_{neu}$", fc="#FFF3EE", ec=C_CURV, fs=11)
rounded_box(ax, 0.775, 0.335, 0.13, 0.065, "$d_{neu}$", fc="#E8F5E9", ec=C_NEU, fs=11)
rounded_box(ax, 0.695, 0.185, 0.20, 0.068,
            "surprisal$_{t+1}$", "outcome",
            fc="#F3EEF9", ec=C_SURP, lw=2.0, fs=10.5, bold=True)

arr(ax, 0.615, 0.302, 0.660, 0.219, color=C_CURV, label="$\\beta_{c_{neu}}$?")
arr(ax, 0.775, 0.302, 0.730, 0.219, color=C_NEU,  label="$\\beta_{d_{neu}}$?")

ax.text(0.893, 0.260, "+ controls\n(no surprisal)", ha="center", va="center",
        fontsize=7.5, color=C_CTRL, style="italic")

equation_text(ax, 0.745, 0.143,
              r"$\mathrm{surprisal}_{t+1} \sim \beta_1 c_{neu} + \beta_2 d_{neu} + \mathbf{controls} + \varepsilon$",
              fs=8.0)

# ── legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor="#FFF3EE", edgecolor=C_CURV,
                   label="Curvature  (direction change in trajectory)"),
    mpatches.Patch(facecolor="#EAF0FF", edgecolor=C_SEM,
                   label="Displacement  (step size, semantic space)"),
    mpatches.Patch(facecolor="#E8F5E9", edgecolor=C_NEU,
                   label="Displacement  (step size, neural space)"),
    mpatches.Patch(facecolor="#F3EEF9", edgecolor=C_SURP,
                   label="Surprisal  (next-word prediction error)"),
    mpatches.Patch(facecolor="#F5F5F5", edgecolor=C_CTRL,
                   label="Nuisance controls  (language / discourse features)"),
]
ax.legend(handles=legend_patches,
          loc="lower center", bbox_to_anchor=(0.5, -0.04),
          ncol=3, fontsize=8.2, frameon=True, framealpha=0.95,
          edgecolor=C_EDGE, columnspacing=1.2, handlelength=1.4)

os.makedirs(FIGURES, exist_ok=True)
for ext in ("pdf", "png", "svg"):
    path = os.path.join(FIGURES, f"fig4_schematic.{ext}")
    fig.savefig(path)
    if ext == "svg":
        txt = open(path).read()
        txt = txt.replace("TeX Gyre Termes", "Times New Roman")
        open(path, "w").write(txt)
    print(f"  -> {path}")
plt.close(fig)
