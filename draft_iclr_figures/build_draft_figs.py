"""
build_draft_figs.py
====================
Composites existing rendered figures into ICLR-draft panels, per section.
Does NOT recompute anything — pure image assembly for a first-pass layout.
Run: conda run -n gpt2_embed python3 build_draft_figs.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

GP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))

def load(p):
    return Image.open(os.path.join(GP, p)).convert("RGB")

def add_title(im, text, pad=60, font_size=42):
    w, h = im.size
    canvas = Image.new("RGB", (w, h + pad), "white")
    canvas.paste(im, (0, pad))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    d.text((20, 10), text, fill="black", font=font)
    return canvas

def hstack(images, gap=30, bg="white"):
    h = max(im.height for im in images)
    resized = []
    for im in images:
        scale = h / im.height
        resized.append(im.resize((int(im.width * scale), h)))
    total_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, h), bg)
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas

def vstack(images, gap=30, bg="white"):
    w = max(im.width for im in images)
    resized = []
    for im in images:
        scale = w / im.width
        resized.append(im.resize((w, int(im.height * scale))))
    total_h = sum(im.height for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (w, total_h), bg)
    y = 0
    for im in resized:
        canvas.paste(im, (0, y))
        y += im.height + gap
    return canvas

# --- Section 2: curvature dominance + flattening + histogram (podcast) ---
fig1 = load("figures/fig1_layer_profiles_pca20_podcast.png")
fig1 = add_title(fig1, "DRAFT Fig 1 — Podcast: curvature flattens across middle layers (n=9 patients, hippocampus)")
fig1.save(os.path.join(OUT, "draft_fig1_curvature_dominance_podcast.png"))

# --- Section 3: surprisal split by layer + max-separation scatter + PCA trajectory ---
fig2 = load("figures/fig2_surprisal_geometry_pca20_podcast.png")
fig2 = add_title(fig2, "DRAFT Fig 2 — Podcast: surprisal-split curvature coupling by layer + trajectory example")
fig2.save(os.path.join(OUT, "draft_fig2_surprisal_layers_podcast.png"))

# --- Section 5: word-frequency control, podcast-only, n=9 (nested-deviance) ---
freq = load("figures/appendix/fig_subtlex_freq_control_surprisal_split_hippocampus_nested.png")
freq = add_title(freq, "DRAFT Fig 3a — Podcast frequency control (nested-deviance), n=9")
freq.save(os.path.join(OUT, "draft_fig3a_freq_control_podcast.png"))

# --- Section 4+6 combined: lag+1 event-triggered curvature, convo vs podcast, semantic vs neural ---
# This is the real purpose-built comparison figure (podcast/figures/), not a composite.
lag_compare = load("podcast/figures/fig_convo_podcast_curvature_comparison.png")
lag_compare.save(os.path.join(OUT, "draft_fig4_lag1_convo_vs_podcast.png"))

# --- Section 4+5+6 combined: same comparison, with frequency overlay (mid layer) ---
lag_freq_compare = load("podcast/figures/fig_convo_podcast_curvature_comparison_freq_surp_overlay_midlayer.png")
lag_freq_compare.save(os.path.join(OUT, "draft_fig4b_lag1_freq_overlay_convo_vs_podcast.png"))

# --- Section 5: additive-gain control, podcast, n=9 ---
gain = load("figures/appendix/fig_gain_control_surprisal_split_hippocampus_10bins.png")
gain = add_title(gain, "DRAFT Fig 3b — Podcast additive-gain control, n=9")
gain.save(os.path.join(OUT, "draft_fig3b_gain_control_podcast.png"))

# NOTE: magnitude-vs-direction decomposition (fig_curvature_gain_direction_control.png)
# is CONVO ONLY (n=15) -- no podcast version exists yet. Not included here.

# --- Section 7: jabberwocky ---
jab = load("jabberwocky/figures/fig_cross_space_paired_jabberwocky.png")
jab = add_title(jab, "DRAFT Fig 5 — Jabberwocky: curvature vs displacement vs position (NOTE: curvature vs displacement n.s., p=0.645)")
jab.save(os.path.join(OUT, "draft_fig5_jabberwocky.png"))

print("Done. Wrote 5 draft figures to", OUT)
