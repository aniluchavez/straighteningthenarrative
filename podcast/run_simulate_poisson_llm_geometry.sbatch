#!/bin/bash
#SBATCH --job-name=sim_poisson_llm
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/podcast/logs/sim_poisson_llm.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/podcast/logs/sim_poisson_llm.err

set -euo pipefail
cd /scratch/aniluchavez/semantic-curvature/podcast
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3 -u simulate_poisson_llm_geometry.py
