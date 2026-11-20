"""
compute_podcast_surprisal.py
=============================
Computes GPT-2-large word-level surprisal for the 9 podcast_comparison
patients (no precomputed surprisal exists for this dataset, unlike the
convo pipeline's ConvoDATAS/Surprisal). Matches the embeddings model
already extracted for these patients (word_embeddings_gpt2.npy, 37 layers
x 1280-dim = gpt2-large).

Reuses the same get_llm_surprisal.py driver and !ARTICLE-file convention
as hippocampal-speaker-semantics/scripts/run_surprisal_new_patients.py.
"""
import os, sys, glob, re, subprocess
import numpy as np
import pandas as pd

PODCAST_ROOT = "/mnt/labworlds/Hayden/Hayden_Lab/speech_247/podcast_comparison"
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Surprisal")
TMP_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
SCRIPT       = "/scratch/aniluchavez/llm_surprisal/get_llm_surprisal.py"
MODEL        = "gpt2-large"
MODE         = "word"

PATIENTS = ["YEY", "YEZ", "YFA", "YFC", "YFD", "YFF", "YFG", "YFI", "YFK"]

_UNINTELLIGIBLE = re.compile(
    r'^[xX\*\?\_\#]{2,}$'
    r'|^\[.*\]$'
    r'|^\(.*\)$'
    r'|^<.*>$',
    re.IGNORECASE
)


def is_unintelligible(w):
    return bool(_UNINTELLIGIBLE.match(w.strip())) if w.strip() else True


def clean_word(w):
    """Match convo's CleanedWord convention: lowercase, strip all
    non-alphanumeric characters (verified against PTYEU_task147:
    CollapsedWord -> CleanedWord is exactly this transform, 0 mismatches)."""
    return re.sub(r"[^a-z0-9]", "", w.lower())


def build_article_file(words_raw, path):
    """words_raw: uncleaned text (is_unintelligible needs raw brackets/markers
    like '[inaudible]' or '**', which cleaning would destroy)."""
    cleaned = []
    for w in words_raw:
        if is_unintelligible(w):
            cleaned.append("the")
        else:
            cleaned.append(clean_word(w).replace(" ", "-") or "the")
    with open(path, "w") as f:
        f.write("!ARTICLE\n")
        f.write(" ".join(cleaned) + "\n")


def parse_surprisal_output(raw):
    lines = raw.strip().splitlines()
    assert lines[0].strip() == "word llmsurp", f"Unexpected header: {lines[0]}"
    return [float(line.split()[1]) for line in lines[1:] if len(line.split()) == 2]


def find_transcript(pid):
    cands = [f for f in glob.glob(os.path.join(PODCAST_ROOT, pid, "transcript", "*.xlsx"))
             if not os.path.basename(f).startswith(("._", "~$"))]
    assert len(cands) == 1, f"{pid}: expected exactly 1 transcript xlsx, found {cands}"
    return cands[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    for pid in PATIENTS:
        tx_path = find_transcript(pid)
        print(f"\n=== {pid}: {tx_path} ===")
        df = pd.read_excel(tx_path, sheet_name="Sheet1")
        words_raw = df["text"].fillna("").astype(str).tolist()
        print(f"  {len(words_raw)} words")

        article_path = os.path.join(TMP_DIR, f"{pid}_input.txt")
        build_article_file(words_raw, article_path)

        cmd = [sys.executable, SCRIPT, article_path, MODEL, MODE]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR:\n{result.stderr[-2000:]}")
            continue

        surprisals = parse_surprisal_output(result.stdout)
        if len(surprisals) != len(words_raw):
            print(f"  MISMATCH: {len(surprisals)} surprisal values for {len(words_raw)} words -- skipping")
            continue

        words_clean = [clean_word(w) for w in words_raw]
        surp_array = [np.nan if is_unintelligible(w) else s for w, s in zip(words_raw, surprisals)]
        out_df = pd.DataFrame({
            "Word": words_clean,
            "RowIndex": range(len(words_clean)),
            "surprisal": surp_array,
        })
        out_path = os.path.join(OUT_DIR, f"{pid}_surprisal.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}  (mean surprisal={out_df['surprisal'].mean():.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
