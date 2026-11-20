# semantic-curvature

Neural–semantic trajectory curvature analyses and the ICLR geometry paper
(formerly `neural_network_similarity/Experiments/p6/geometry_paper`; a symlink
remains at that old path).

- Root: conversation-data geometry/curvature analyses, figure builders, paper `.tex`
- `podcast/`: podcast pipeline (`podcast_loader.py`)
- `jabberwocky/`: Jabberwocky pipeline (`jabberwocky_loader.py`)

Only code and LaTeX are tracked (see `.gitignore`). Data, embedding caches
(`*/EmbedCache`), results, figures, and transcript-derived files stay on
`/scratch` and are never committed (they contain PHI).
