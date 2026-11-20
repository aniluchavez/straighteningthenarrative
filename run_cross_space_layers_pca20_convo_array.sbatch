#!/bin/bash
# PCA(20)-reduced-trajectory cross-space-layers (fig1 Panel B), CONVO, all 9 models.
#
# Submit with: sbatch run_cross_space_layers_pca20_convo_array.sbatch

#SBATCH --job-name=xsl_pca20_convo
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/logs/xsl_pca20_convo_%a.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/logs/xsl_pca20_convo_%a.err

set -euo pipefail
cd /scratch/aniluchavez/semantic-curvature
mkdir -p logs

MODELS=(gpt2 gpt2-medium bert-base roberta-base opt-350m llama-2-7b llama-3.1-8b gemma-2-9b mistral-7b)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
$PY -u compute_cross_space_layers_pca20.py --model "$MODEL"
