"""
make_freq_surp_tables_docx.py
================================
Renders the two frequency/surprisal control tables (gain/frequency
control on the podcast surprisal->neural-curvature correlation, and the
unique-R2 timescale grid) as clean booktabs-style ("LaTeX look") table
images, then embeds both into a single .docx that can be dropped into a
Google Doc/Sheet.

Saves:
  tables/freq_surp_tables.docx
  tables/podcast_gain_freq_table_render.png   (intermediate image)
  tables/freq_surp_curvature_table_render.png (intermediate image)

Run:
    conda run -n gpt2_embed python3 -u make_freq_surp_tables_docx.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def render_booktabs_table(rows, col_labels, col_widths, out_path, title=None,
                           fontsize=11, header_fontsize=11, row_h=1.0, title_fontsize=13):
    n_rows = len(rows)
    n_cols = len(col_labels)
    fig_w = sum(col_widths)

    x_edges = [0]
    for w in col_widths:
        x_edges.append(x_edges[-1] + w)
    total_w = x_edges[-1]
    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(n_cols)]

    # Layout in row units (each unit = one text line's worth of vertical space),
    # counting from the top down, then flipped so y increases upward for plotting.
    # Multi-line headers/titles need extra vertical room proportional to their
    # line count so rules and text never overlap.
    header_lines = max(lab.count("\n") + 1 for lab in col_labels)
    title_lines = title.count("\n") + 1 if title else 0

    header_row_y = n_rows + 1 + 0.35 * (header_lines - 1)
    data_row_ys = list(range(n_rows, 0, -1))
    header_half = 0.35 + 0.35 * (header_lines - 1)
    top_rule_y = header_row_y + header_half
    header_rule_y = header_row_y - header_half
    bottom_rule_y = data_row_ys[-1] - 0.6
    title_y = top_rule_y + 0.55 + 0.4 * (title_lines - 1) + 0.4

    fig_h = row_h * (n_rows + 3 + header_lines + 1.4 * title_lines) * 0.42
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    for xc, lab in zip(x_centers, col_labels):
        ax.text(xc, header_row_y, lab, ha="center", va="center", fontsize=header_fontsize, fontweight="bold")

    for y, row in zip(data_row_ys, rows):
        for xc, val in zip(x_centers, row):
            ax.text(xc, y, str(val), ha="center", va="center", fontsize=fontsize)

    ax.plot([0, total_w], [top_rule_y, top_rule_y], color="black", lw=1.4)
    ax.plot([0, total_w], [header_rule_y, header_rule_y], color="black", lw=0.8)
    ax.plot([0, total_w], [bottom_rule_y, bottom_rule_y], color="black", lw=1.4)

    if title:
        ax.text(total_w / 2, title_y, title, ha="center", va="center",
                fontsize=title_fontsize, fontweight="bold")
        ax.set_ylim(bottom_rule_y - 0.5, title_y + 0.4 * title_lines + 0.5)
    else:
        ax.set_ylim(bottom_rule_y - 0.5, top_rule_y + 0.5)
    ax.set_xlim(0, total_w)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}")


def build_gain_freq_table_image():
    df = pd.read_csv(os.path.join(BASE_DIR, "podcast_gain_freq_table.csv"))
    col_labels = ["Condition", "N", "mean r", "SEM", "t", "p", "N positive"]
    col_widths = [3.4, 0.6, 0.9, 0.8, 0.8, 1.1, 1.1]
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["condition"], int(r["n_patients"]), f"{r['mean_r']:+.4f}", f"{r['sem_r']:.4f}",
            f"{r['t']:.2f}", r["p"], f"{int(r['n_positive'])}/{int(r['n_patients'])}",
        ])
    out_path = os.path.join(BASE_DIR, "podcast_gain_freq_table_render.png")
    render_booktabs_table(
        rows, col_labels, col_widths, out_path,
        title="Surprisal -> neural curvature under gain / frequency controls\n(podcast, hippocampus, 13 patients, exit-aligned lag+1)",
    )
    return out_path


def build_unique_r2_table_image():
    df = pd.read_csv(os.path.join(BASE_DIR, "freq_surp_curvature_table.csv"))
    col_labels = ["Curvature", "Timescale", "POS\nctrl", "Unique R²\nfrequency",
                  "Unique R²\nsurprisal", "Ratio\n(freq/surp)", "Dominant"]
    col_widths = [1.3, 1.3, 0.8, 1.7, 1.7, 1.3, 1.3]
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["curvature"], r["timescale"], r["pos_control"],
            f"{r['unique_r2_freq']:.5f}{r['freq_sig']}",
            f"{r['unique_r2_surp']:.5f}{r['surp_sig']}",
            f"{r['ratio_freq_over_surp']:.2f}×", r["dominant"],
        ])
    out_path = os.path.join(BASE_DIR, "freq_surp_curvature_table_render.png")
    render_booktabs_table(
        rows, col_labels, col_widths, out_path,
        title="Frequency vs. surprisal: unique contribution to curvature by timescale\n(nested unique-$R^2$, patient-cluster bootstrap, 2000 resamples; *** p<0.001)",
        fontsize=10, header_fontsize=10, row_h=0.46,
    )
    return out_path


def build_docx(image_paths, captions, out_path):
    doc = Document()
    doc.add_heading("Frequency / Surprisal Curvature Control Tables", level=1)
    for img_path, caption in zip(image_paths, captions):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(img_path, width=Inches(6.5))
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)
        doc.add_paragraph()
    doc.save(out_path)
    print(f"\nSaved docx -> {out_path}")


def main():
    p1 = build_gain_freq_table_image()
    p2 = build_unique_r2_table_image()
    out_path = os.path.join(BASE_DIR, "freq_surp_tables.docx")
    build_docx(
        [p1, p2],
        [
            "Table 1. Additive/multiplicative gain control and a hard mid-decile SUBTLEX "
            "frequency restriction applied to the raw surprisal→neural-curvature correlation.",
            "Table 2. Nested unique-R² grid: frequency's and surprisal's independent, "
            "non-overlapping contribution to curvature across word-level vs. 15-word-chunk "
            "timescales, with and without function-word (POS) control.",
        ],
        out_path,
    )


if __name__ == "__main__":
    main()
