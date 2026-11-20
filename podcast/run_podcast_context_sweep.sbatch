#!/bin/bash
#SBATCH --job-name=podcast_ctxsweep
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/podcast/logs/podcast_ctxsweep.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/podcast/logs/podcast_ctxsweep.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
cd /scratch/aniluchavez/semantic-curvature/podcast
mkdir -p logs EmbedCache

for CTX in 100 50 10 3; do
  $PY -u extract_podcast_embeddings.py --models llama-3.1-8b --context_words $CTX --max_tokens 512
done

for CTX in 200 100 50 10 3; do
  $PY -u extract_podcast_embeddings.py --models gpt2-xl --context_words $CTX --max_tokens 512
done
