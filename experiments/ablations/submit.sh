#!/bin/bash
#SBATCH --job-name=ablations
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00
# Slurm array wrapper for the ZINC_tiny ablation (CPU, as the original runs).
# Each array task runs PARALLEL single-threaded processes; finished trials are skipped.
#   sbatch --array=0-15 experiments/ablations/submit.sh
# ABL_ENV=paper: bit-identical trials, as the original sweeps (pinned env reproduce/paper_env,
# four threads per process, AMD EPYC 7742 nodes without GPU):
#   ABL_ENV=paper NTASKS=8 PARALLEL=16 sbatch -p internal -c 64 --array=0-7 experiments/ablations/submit.sh
set -euo pipefail
THREADS=${THREADS:-$([[ "${ABL_ENV:-}" == "paper" ]] && echo 4 || echo 1)}
export OMP_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS}
unset VIRTUAL_ENV
NTASKS=${NTASKS:-16}
PARALLEL=${PARALLEL:-32}
TASK=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$((NTASKS * PARALLEL))
LOGS=results/ablations/logs
mkdir -p "${LOGS}"
if [[ "${ABL_ENV:-}" == "paper" ]]; then
    export PYTHONPATH=src
    RUN=(uv run --frozen --no-sync --project reproduce/paper_env python -u)
else
    RUN=(uv run --frozen python -u)
fi
for ((p = 0; p < PARALLEL; p++)); do
    chunk=$((TASK * PARALLEL + p))
    "${RUN[@]}" experiments/ablations/run.py "$@" --chunk "${chunk}/${TOTAL}" \
        > "${LOGS}/abl_${SLURM_JOB_ID:-local}_${chunk}.log" 2>&1 &
done
wait
