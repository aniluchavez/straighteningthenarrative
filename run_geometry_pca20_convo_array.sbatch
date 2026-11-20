#!/bin/bash
# PCA(20)-reduced-trajectory geometry pipeline, CONVO, all 9 models.
# One array task per model. PTYFU_task224 excluded (stale embeddings,
# regenerating separately via run_ptyfu_reconstructed_embeddings.sbatch).
#
# Submit with: sbatch run_geometry_pca20_convo_array.sbatch

#SBATCH --job-name=geom_pca20_convo
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/logs/geom_pca20_convo_%a.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/logs/geom_pca20_convo_%a.err

set -euo pipefail
cd /scratch/aniluchavez/semantic-curvature
mkdir -p logs

MODELS=(gpt2 gpt2-medium bert-base roberta-base opt-350m llama-2-7b llama-3.1-8b gemma-2-9b mistral-7b)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u run_geometry_analysis_pca20.py --models "$MODEL" --regions hippocampus
