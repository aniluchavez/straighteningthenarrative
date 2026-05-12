# Hippocampal Geometry Paper — Figure Code

Code for reproducing all figures and tables in:  
**"Trajectory curvature links hippocampal geometry to LLM semantic representations"**

---

## Repository structure

```
geometry_paper_code/
├── analysis/          # scripts that run core analyses → produce results CSVs/JSONs
├── figures/           # scripts that read results → produce paper figures
└── requirements.txt
```

---

## Analysis scripts

Run these first. They write results files that the figure scripts read.

| Script | What it does |
|--------|-------------|
| `analysis/run_geometry_analysis.py` | Core geometry pipeline: loads per-patient neural and LLM embeddings, computes PCA trajectories, curvature, displacement, and cross-space geometric coupling for all 9 models across all layers. Writes per-model JSON results. |
| `analysis/run_geometry_paper_curvature_alignment.py` | Word-level cross-space ridge-regression decoding (LLM → Neural and Neural → LLM), with curvature/displacement binning, partial correlations, and null shuffles. Writes per-lag pointwise CSVs. |
| `analysis/run_multimodel_linear_curvature_alignment.py` | Runs the decoding pipeline across all 9 LLM architectures and aggregates a model-level summary CSV (mean curvature, mean decoding quality). |
| `analysis/run_layer_sweep.sh` | Calls `run_geometry_paper_curvature_alignment.py` for each of the 33 LLaMA layers at both lags. Estimated runtime ~2–4 hours. |

---

## Figure scripts

| Script | Output | Figure |
|--------|--------|--------|
| `figures/make_geometry_figures.py` | `fig1_layer_profiles`, `fig2_surprisal_geometry`, `fig3_event_triggered` | **Main figures 1–3** — layer profiles, surprisal-geometry relationship, event-triggered curvature |
| `figures/make_cross_space_paired_fig.py` | `fig_cross_space_paired` | **Main figure** — paired cross-space decoding summary |
| `figures/make_elbow_justification_fig.py` | `fig_elbow_justification` | **Methods figure** — PCA dimensionality elbow criterion |
| `figures/make_ablation_table.py` | `ablation_table.tex` | **Table** — ablation results |
| `figures/make_ablation_table_fig.py` | `fig_ablation_*` | **Supplementary** — ablation visualisation |
| `figures/plot_cross_state_decoding.py` | `fig_cross_state_decoding` | **Main figure** — mean decoding r (A), curvature quintile vs decoding (B), per-patient partial r curvature vs displacement (C) |
| `figures/make_cross_space_layer_fig.py` | `fig_cross_space_layer_profiles_chunk` | **Appendix S1** — cross-space coupling across layers for all 9 models |
| `figures/plot_appendix_multimodel_geometry.py` | `fig_appendix_multimodel_geometry` | **Appendix S2** — heatmap of coupling per model × metric; scatter of LLM curvature vs hippocampal readout |
| `figures/plot_layer_sweep.py` | `fig_layer_sweep` | **Appendix S3** — partial r of curvature and displacement across all 33 LLaMA layers |
| `figures/make_event_triggered_displacement.py` | `fig_event_triggered_displacement` | **Appendix** — event-triggered displacement at surprising words |
| `figures/plot_summary_figure.py` | `fig_summary` | **Appendix** — flatness story summary (within-space + cross-space quintiles + curvature vs displacement) |

---

## Environment

Python 3.10+, dependencies in `requirements.txt`. LLM embedding extraction for models
with ≥7B parameters (LLaMA, Gemma, Mistral) requires a GPU with ≥24 GB VRAM.
All downstream geometry and decoding analyses run on CPU.

```bash
pip install -r requirements.txt
```

---

## Data note

Raw neural recordings and LLM embeddings are not included as they contain protected
health information (PHI). Precomputed results CSVs and JSONs needed to reproduce the
figures are available upon reasonable request.
