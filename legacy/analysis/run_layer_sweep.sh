#!/usr/bin/env bash
# run_layer_sweep.sh
# Runs curvature alignment across all LLM layers (0–32) for both lags.
# Outputs go to results/geometry_paper_curvature_alignment/ with _layerN suffix.
# Usage: bash run_layer_sweep.sh [--lags "0 1"] [--max-layer 32]
#
# Estimated runtime: ~2-5 min/layer for 10 patients → ~1-2h total per lag.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LAGS="${LAGS:-0 1}"
MAX_LAYER="${MAX_LAYER:-32}"
REGIONS="hippocampus ACC"

echo "=== Layer sweep: layers 0-${MAX_LAYER}, lags: ${LAGS} ==="

for lag in $LAGS; do
    for layer in $(seq 0 "$MAX_LAYER"); do
        echo ""
        echo "--- layer=${layer}  lag=${lag} ---"
        python3 -u run_geometry_paper_curvature_alignment.py \
            --regions hippocampus ACC \
            --llm-pcs 64 \
            --shuffle-cv \
            --decoder ridge \
            --directions LLM_to_neural neural_to_LLM \
            --target-lag "$lag" \
            --match-pcs \
            --force-layer "$layer" \
            --n-bins 5 \
            --n-splits 5
    done
done

echo ""
echo "=== Layer sweep complete ==="
