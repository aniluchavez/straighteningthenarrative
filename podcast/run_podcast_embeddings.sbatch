#!/bin/bash
#SBATCH --job-name=podcast_embed
#SBATCH --partition=guppy
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/podcast/logs/podcast_embed.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/podcast/logs/podcast_embed.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
cd /scratch/aniluchavez/semantic-curvature/podcast
mkdir -p logs EmbedCache

$PY -u extract_podcast_embeddings.py --context_words 200 --max_tokens 512
$PY -u extract_podcast_fasttext.py
