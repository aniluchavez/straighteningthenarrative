# Straightening the Narrative — Curvature + Geometry Paper

Neural–semantic trajectory curvature analyses and the ICLR geometry paper:
**"Trajectory curvature links hippocampal geometry to LLM semantic representations"**

Working tree lives at `/scratch/aniluchavez/straighteningthenarrative`
(formerly `neural_network_similarity/Experiments/p6/geometry_paper`; a symlink
remains at that old path).

## Layout

- Root: conversation-data geometry/curvature analyses (`run_geometry_analysis*.py`,
  `compute_*`), figure builders (`make_*`), paper source (`methods_section.tex`,
  `embeddings_methods.tex`, `references.bib`), SLURM scripts (`*.sbatch`)
- `podcast/`: podcast pipeline (`podcast_loader.py`)
- `jabberwocky/`: Jabberwocky pipeline (`jabberwocky_loader.py`)
- `tables/`, `results/`: LaTeX tables
- `draft_iclr_figures/`: ICLR draft figure builders
- `legacy/`: May 2026 figure-release snapshot (`analysis/`, `figures/`), superseded
  by the scripts above

## Environment

Python 3.10+, dependencies in `requirements.txt`. LLM embedding extraction for models
with ≥7B parameters (LLaMA, Gemma, Mistral) requires a GPU with ≥24 GB VRAM.
All downstream geometry and decoding analyses run on CPU.

## Data note

Only code and LaTeX are tracked (see `.gitignore`). Raw neural recordings, LLM
embeddings (`*/EmbedCache`), results, figures, and transcript-derived files stay on
`/scratch` and are never committed: they contain protected health information (PHI).
