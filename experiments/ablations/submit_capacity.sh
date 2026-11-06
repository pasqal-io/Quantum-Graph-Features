#!/bin/bash
#SBATCH --job-name=capacity
#SBATCH --partition=internal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
# Slurm array wrapper for the GT capacity study (Fig. 6; one GPU per array task, PARALLEL
# trials per GPU). Env: NTASKS (= array size, default 3), PARALLEL (default 5); remaining
# arguments go to `capacity.py run`. Finished trials are skipped, so re-submitting resumes.
#   sbatch --array=0-2 experiments/ablations/submit_capacity.sh
set -euo pipefail
export OMP_NUM_THREADS=2
unset VIRTUAL_ENV
NTASKS=${NTASKS:-3}
PARALLEL=${PARALLEL:-5}
TASK=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$((NTASKS * PARALLEL))
LOGS=results/ablations/logs
mkdir -p "${LOGS}"
echo "host=$(hostname) start=$(date) task=${TASK}/${NTASKS} parallel=${PARALLEL} args=$*"
for ((p = 0; p < PARALLEL; p++)); do
    chunk=$((TASK * PARALLEL + p))
    uv run --frozen python -u experiments/ablations/capacity.py run "$@" --chunk "${chunk}/${TOTAL}" \
        > "${LOGS}/capacity_${SLURM_JOB_ID:-local}_${chunk}.log" 2>&1 &
done
wait
echo "end=$(date)"
