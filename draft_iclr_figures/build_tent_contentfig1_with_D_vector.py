"""
build_tent_contentfig1_with_D_vector.py
==========================================
Rebuilds tent_contentfig1_with_D.pdf as a TRUE VECTOR composite.

The existing tent_contentfig1_with_D.pdf (and tent_contentfig1.pdf) were
built with Pillow's Image.save(..., "PDF") on a rasterized PNG -- Pillow's
PDF driver just wraps a single JPEG in a minimal PDF wrapper (confirmed via
`pdffonts`/`pdfimages`: zero fonts, one full-page JPEG). Nothing in it is
selectable/editable in Illustrator because there was never any vector
content to begin with.

The four source panels, however, already exist as genuine vector PDFs
(matplotlib output with pdf.fonttype=42, embedded TrueType fonts, no
rasterization):
  - tentative_figures/fig_cross_space_paired_pc8_chunk3_11models.pdf   (left: curvature/displacement/position ablation bars)
  - tentative_figures/panelA_mean_curvature_pca20_nochunk.pdf          (top-right: Panel A)
  - tentative_figures/panelB_alignment_pca20_nochunk.pdf               (bottom-left: Panel B)
  - tentative_figures/panelC_curvature_distribution_pca20_nochunk_llama2_7b_L13.pdf  (bottom-right: Panel D)

This script places each source PDF's page as a scaled/translated Form
XObject on one new page (via pypdf's PageObject.add_transformation +
merge_page), which preserves the original vector paths and text objects
exactly -- Illustrator will see and let you select/edit each panel's
individual lines, points, and text, not a flattened image.

Run:
    conda run -n gpt2_embed python3 build_tent_contentfig1_with_D_vector.py
"""
import os
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject

BASE = os.path.dirname(os.path.abspath(__file__))
GP = os.path.dirname(BASE)
TENT = os.path.join(GP, "tentative_figures")

SOURCES = {
    "left": os.path.join(TENT, "fig_cross_space_paired_pc8_chunk3_11models.pdf"),
    "A": os.path.join(TENT, "panelA_mean_curvature_pca20_nochunk.pdf"),
    "B": os.path.join(GP, "figures", "fig_multimodel_layersweep_surprisal_curvature_pca20_podcast.pdf"),
    "D": os.path.join(TENT, "panelC_curvature_distribution_pca20_nochunk_llama2_7b_L13.pdf"),
}

OUT_PATH = os.path.join(BASE, "tent_contentfig1_with_D_vector.pdf")

# Target page size / layout (matches the visual layout of the original
# raster composite: big panel on the left spanning full height, 2x2-ish
# grid of A/B/D on the right -- but here laid out as left column + 2x2 right grid).
PAGE_W, PAGE_H = 1600, 830
LEFT_W = 620
RIGHT_W = PAGE_W - LEFT_W
RIGHT_COL_W = RIGHT_W / 2
ROW_H = PAGE_H / 2


def place(writer, base_page, src_path, x0, y0, box_w, box_h, pad=10):
    reader = PdfReader(src_path)
    page = reader.pages[0]
    src_w = float(page.mediabox.width)
    src_h = float(page.mediabox.height)
    avail_w, avail_h = box_w - 2 * pad, box_h - 2 * pad
    scale = min(avail_w / src_w, avail_h / src_h)
    new_w, new_h = src_w * scale, src_h * scale
    # center within the box
    tx = x0 + (box_w - new_w) / 2
    ty = y0 + (box_h - new_h) / 2
    transform = Transformation().scale(scale, scale).translate(tx, ty)
    base_page.merge_transformed_page(page, transform)


def main():
    writer = PdfWriter()
    base_page = writer.add_blank_page(width=PAGE_W, height=PAGE_H)

    # left panel: full height
    place(writer, base_page, SOURCES["left"], 0, 0, LEFT_W, PAGE_H)
    # top-right: Panel A
    place(writer, base_page, SOURCES["A"], LEFT_W, ROW_H, RIGHT_COL_W, ROW_H)
    # bottom-left-of-right-block: Panel B
    place(writer, base_page, SOURCES["B"], LEFT_W, 0, RIGHT_COL_W, ROW_H)
    # bottom-right: Panel D
    place(writer, base_page, SOURCES["D"], LEFT_W + RIGHT_COL_W, 0, RIGHT_COL_W, ROW_H)

    with open(OUT_PATH, "wb") as f:
        writer.write(f)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
