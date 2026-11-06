#!/bin/bash
#SBATCH --job-name=non_unit_disk
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# Slurm array wrapper for the exact non-unit-disk MUTAG benchmark (CPU).
# Each array task runs PARALLEL single-threaded processes; finished trials are skipped.
#   NTASKS=4 sbatch --array=0-3 experiments/non_unit_disk/submit.sh
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset VIRTUAL_ENV
NTASKS=${NTASKS:-16}
PARALLEL=${PARALLEL:-32}
TASK=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$((NTASKS * PARALLEL))
LOGS=results/non_unit_disk/logs
mkdir -p "${LOGS}"
for ((p = 0; p < PARALLEL; p++)); do
    chunk=$((TASK * PARALLEL + p))
    uv run --frozen python -u experiments/non_unit_disk/run.py "$@" --chunk "${chunk}/${TOTAL}" \
        > "${LOGS}/nud_${SLURM_JOB_ID:-local}_${chunk}.log" 2>&1 &
done
wait
