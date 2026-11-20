"""
make_freq_surp_curvature_table.py
====================================
Formats results/freq_surprisal_full_grid_test.json (already computed by
compute_freq_surprisal_full_grid_test.py) into a paper-ready table: for
each {curvature type} x {timescale}, the raw correlation and the unique
(nested-deviance) R^2 for frequency and surprisal, with and without a
function-word/POS control, plus the dominance ratio.

Saves:
  tables/freq_surp_curvature_table.csv
  tables/freq_surp_curvature_table.tex

Run:
    cd .../geometry_paper/podcast
    conda run -n gpt2_embed python3 -u make_freq_surp_curvature_table.py
"""
import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
TABLES_DIR = os.path.join(PARENT_DIR, "tables")
IN_PATH = os.path.join(BASE_DIR, "results", "freq_surprisal_full_grid_test.json")

LABELS = {
    "semantic_chunk1": ("Semantic", "Word (1)"),
    "semantic_chunk15": ("Semantic", "Chunk (15)"),
    "neural_chunk1": ("Neural", "Word (1)"),
    "neural_chunk15": ("Neural", "Chunk (15)"),
}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    d = json.load(open(IN_PATH))
    rows = []
    for tag, (curv_label, scale_label) in LABELS.items():
        cell = d[tag]
        for pos_key, pos_label in [("no_pos_control", "No"), ("with_pos_control", "Yes")]:
            uf = cell[pos_key]["unique_freq"]
            us = cell[pos_key]["unique_surp"]
            ratio = cell[pos_key]["ratio_freq_over_surp"]
            rows.append(dict(
                curvature=curv_label, timescale=scale_label, pos_control=pos_label,
                n=cell["n_rows"],
                raw_r_freq=round(cell["raw_r_freq"], 3), raw_r_surp=round(cell["raw_r_surp"], 3),
                unique_r2_freq=round(uf["unique_r2"], 5), freq_ci=f"[{uf['ci_lo']:.5f}, {uf['ci_hi']:.5f}]",
                freq_sig=stars(uf["p"]),
                unique_r2_surp=round(us["unique_r2"], 5), surp_ci=f"[{us['ci_lo']:.5f}, {us['ci_hi']:.5f}]",
                surp_sig=stars(us["p"]),
                ratio_freq_over_surp=round(ratio, 2),
                dominant="Frequency" if ratio > 1 else "Surprisal",
            ))
    df = pd.DataFrame(rows)

    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, "freq_surp_curvature_table.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {csv_path}")

    tex_path = os.path.join(TABLES_DIR, "freq_surp_curvature_table.tex")
    with open(tex_path, "w") as f:
        f.write(r"\begin{tabular}{llcrrrl}" + "\n")
        f.write(r"  \toprule" + "\n")
        f.write(r"  Curvature & Timescale & POS ctrl & Unique $R^2$ freq & Unique $R^2$ surp & "
                r"Ratio (freq/surp) & Dominant \\" + "\n")
        f.write(r"  \midrule" + "\n")
        for _, row in df.iterrows():
            f.write(f"  {row['curvature']} & {row['timescale']} & {row['pos_control']} & "
                    f"{row['unique_r2_freq']:.5f}{row['freq_sig']} & "
                    f"{row['unique_r2_surp']:.5f}{row['surp_sig']} & "
                    f"{row['ratio_freq_over_surp']:.2f}$\\times$ & {row['dominant']} \\\\\n")
        f.write(r"  \bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
    print(f"Saved -> {tex_path}")


if __name__ == "__main__":
    main()
