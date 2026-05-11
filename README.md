# Hippocampal Geometry Paper — Figure Code

Code for reproducing figures in:  
**"Trajectory curvature links hippocampal geometry to LLM semantic representations"**

---

## Repository structure

```
geometry_paper_code/
├── analysis/          # scripts that run the core analyses and produce results CSVs/JSONs
├── figures/           # scripts that read those results and produce the paper figures
└── requirements.txt
```

---

## Analysis scripts

These scripts must be run first; they write results to disk that the figure scripts read.

| Script | What it does |
|--------|-------------|
| `analysis/run_geometry_analysis.py` | Core geometry pipeline: loads per-patient neural and LLM embeddings, computes PCA trajectories, curvature, displacement, and cross-space geometric coupling for all 9 models across all layers. Writes per-model JSON results. |
| `analysis/run_geometry_paper_curvature_alignment.py` | Word-level cross-space ridge-regression decoding (LLM → Neural and Neural → LLM), with curvature/displacement binning, partial correlations, and null shuffles. Writes per-lag pointwise CSVs used by the main figure. |
| `analysis/run_multimodel_linear_curvature_alignment.py` | Runs the same decoding pipeline across all 9 LLM architectures and aggregates a model-level summary CSV (mean curvature, mean decoding quality) used by the multi-model appendix figure. |
| `analysis/run_layer_sweep.sh` | Shell script that calls `run_geometry_paper_curvature_alignment.py` sequentially for each of the 33 LLaMA layers (both lags), producing the per-layer pointwise CSVs used by the layer sweep figure. Estimated runtime ~2–4 hours. |

---

## Figure scripts

Run after the analysis scripts.

| Script | Output | Figure in paper |
|--------|--------|----------------|
| `figures/plot_cross_state_decoding.py` | `fig_cross_state_decoding.{pdf,png,svg}` | **Main figure** — mean decoding r by direction/lag (A), curvature quintile vs decoding (B), per-patient partial r curvature vs displacement (C) |
| `figures/make_cross_space_layer_fig.py` | `fig_cross_space_layer_profiles_chunk.{pdf,png,svg}` | **Appendix S1** — cross-space geometric coupling (displacement, curvature, position) across normalised layer position for all 9 models |
| `figures/plot_appendix_multimodel_geometry.py` | `fig_appendix_multimodel_geometry.{pdf,png,svg}` | **Appendix S2** — heatmap of coupling per model × metric, and scatter of LLM curvature vs hippocampal readout across architectures |
| `figures/plot_layer_sweep.py` | `fig_layer_sweep.{pdf,png,svg}` | **Appendix S3** — partial r of curvature and displacement on decoding quality across all 33 LLaMA layers, both directions and lags |

---

## Environment

Python 3.10+, dependencies in `requirements.txt`. Large model embedding extraction
(LLaMA, Gemma, Mistral) requires a GPU with ≥24 GB VRAM; all other scripts run on CPU.

```bash
pip install -r requirements.txt
```

---

## Data note

Neural recording data and raw LLM embeddings are not included in this repository as they
contain protected health information (PHI). The precomputed results CSVs and JSONs
required to reproduce the figures can be made available upon request.
