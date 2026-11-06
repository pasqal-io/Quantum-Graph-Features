#!/bin/bash
#SBATCH --job-name=gnns
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# Slurm array wrapper for the downstream GNN grid (CPU, as the original runs).
# Each array task runs PARALLEL single-threaded trial processes; together the
# NTASKS x PARALLEL processes cover the grid (run.py --chunk), finished trials are skipped.
# Env: NTASKS (= array size, default 40), PARALLEL (default 32), THREADS per process
# (default 1); the remaining arguments go to run.py.
#   sbatch --array=0-39 experiments/downstream_gnns/submit.sh --dataset mutag
#   NTASKS=10 sbatch --array=0-9 experiments/downstream_gnns/submit.sh --dataset zinc
# Validation subset (README): NTASKS=1 sbatch --array=0 ... --dataset mutag --configs validation
# GNN_ENV=paper: the pinned environment of the original runs (reproduce/paper_env; run
# `uv sync --frozen --project reproduce/paper_env` once). The published trials are reproduced bit
# for bit on AMD EPYC 7742 nodes without GPU (here: `-p internal`, no --gres) with one thread per
# process for MUTAG and four for ZINC:
#   GNN_ENV=paper NTASKS=2 PARALLEL=64 sbatch -p internal -c 64 --array=0-1 \
#       experiments/downstream_gnns/submit.sh --dataset mutag
#   GNN_ENV=paper THREADS=4 NTASKS=15 PARALLEL=30 sbatch -p internal -c 120 --array=0-14 \
#       experiments/downstream_gnns/submit.sh --dataset zinc
set -euo pipefail
THREADS=${THREADS:-1}  # per trial process
export OMP_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS}
unset VIRTUAL_ENV
NTASKS=${NTASKS:-40}
PARALLEL=${PARALLEL:-32}
TASK=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$((NTASKS * PARALLEL))
LOGS=results/downstream_gnns/logs
mkdir -p "${LOGS}"
echo "host=$(hostname) start=$(date) task=${TASK}/${NTASKS} parallel=${PARALLEL} args=$*"
if [[ "${GNN_ENV:-}" == "paper" ]]; then
    export PYTHONPATH=src
    RUN=(uv run --frozen --no-sync --project reproduce/paper_env python -u)
else
    RUN=(uv run --frozen python -u)
fi
for ((p = 0; p < PARALLEL; p++)); do
    chunk=$((TASK * PARALLEL + p))
    "${RUN[@]}" experiments/downstream_gnns/run.py "$@" --chunk "${chunk}/${TOTAL}" \
        > "${LOGS}/gnns_${SLURM_JOB_ID:-local}_${chunk}.log" 2>&1 &
done
wait
echo "end=$(date)"
