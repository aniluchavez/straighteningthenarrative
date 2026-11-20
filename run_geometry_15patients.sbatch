#!/bin/bash
#SBATCH --job-name=geom_15pat
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/aniluchavez/semantic-curvature/logs/geom_15patients.out
#SBATCH --error=/scratch/aniluchavez/semantic-curvature/logs/geom_15patients.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
cd /scratch/aniluchavez/semantic-curvature
mkdir -p logs

$PY -u run_geometry_analysis.py --models llama-3.1-8b --regions hippocampus --overwrite
