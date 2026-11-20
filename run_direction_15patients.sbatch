#!/bin/bash
#SBATCH --job-name=geom_direction
#SBATCH --partition=guppy
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/aniluchavez/straighteningthenarrative/logs/geom_direction.out
#SBATCH --error=/scratch/aniluchavez/straighteningthenarrative/logs/geom_direction.err

PY=/scratch/aniluchavez/miniforge3/envs/gpt2_embed/bin/python3
cd /scratch/aniluchavez/straighteningthenarrative
mkdir -p logs

$PY -u run_event_triggered_direction.py
