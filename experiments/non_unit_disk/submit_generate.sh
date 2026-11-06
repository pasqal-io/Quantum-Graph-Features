#!/bin/bash
#SBATCH --job-name=nud_generate
#SBATCH --partition=internal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Exact MUTAG features (N <= 24, emu-sv on GPU): one array task per (coupling, pulse).
#   sbatch --array=0-3 experiments/non_unit_disk/submit_generate.sh results/non_unit_disk/data
set -euo pipefail
unset VIRTUAL_ENV
OUT=${1:-results/non_unit_disk/data}
COMBOS=("geo afm" "geo fm" "nn afm" "nn fm")
read -r COUPLING PULSE <<< "${COMBOS[${SLURM_ARRAY_TASK_ID:-0}]}"
mkdir -p results/non_unit_disk/logs
echo "host=$(hostname) start=$(date) coupling=${COUPLING} pulse=${PULSE}"
uv run --frozen --extra generate python -u experiments/non_unit_disk/generate.py \
    --coupling "${COUPLING}" --pulse "${PULSE}" --out "${OUT}"
echo "end=$(date)"
