#!/bin/bash
#SBATCH --job-name=esa
#SBATCH --partition=internal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Slurm array wrapper for the ESA grid (one GPU per array task, PARALLEL trials per GPU).
# Env: NTASKS (= array size, default 8), PARALLEL (default 4), LOGS (default
# results/esa/logs), ESA_ENV (default: the project environment, `uv run --extra esa`;
# `paper`: the pinned environment of the original runs, reproduce/paper_env, with
# the xformers attention kernel: bit-identical trials; run
# `uv sync --frozen --project reproduce/paper_env` once before submitting).
# The remaining arguments go to run.py. Finished trials are skipped, so re-submitting
# resumes.
#   sbatch --array=0-7 experiments/esa/submit.sh --configs paper
#   ESA_ENV=paper sbatch --array=0-7 experiments/esa/submit.sh --configs paper --out results/esa_exact
#   NTASKS=2 sbatch --array=0-1 experiments/esa/submit.sh --configs validation
set -euo pipefail
export OMP_NUM_THREADS=2
unset VIRTUAL_ENV
NTASKS=${NTASKS:-8}
PARALLEL=${PARALLEL:-4}
LOGS=${LOGS:-results/esa/logs}
TASK=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$((NTASKS * PARALLEL))
case "${ESA_ENV:-default}" in
    default) RUN=(uv run --frozen --extra esa python -u experiments/esa/run.py) ;;
    paper)
        export PYTHONPATH=src${PYTHONPATH:+:${PYTHONPATH}}
        RUN=(uv run --frozen --no-sync --project reproduce/paper_env
             python -u experiments/esa/run.py --attention xformers)
        ;;
    *) echo "ESA_ENV must be 'default' or 'paper', got '${ESA_ENV}'" >&2; exit 2 ;;
esac
mkdir -p "${LOGS}"
echo "host=$(hostname) start=$(date) task=${TASK}/${NTASKS} parallel=${PARALLEL}" \
    "env=${ESA_ENV:-default} args=$*"
for ((p = 0; p < PARALLEL; p++)); do
    chunk=$((TASK * PARALLEL + p))
    "${RUN[@]}" "$@" --chunk "${chunk}/${TOTAL}" \
        > "${LOGS}/esa_${SLURM_JOB_ID:-local}_${chunk}.log" 2>&1 &
done
wait
echo "end=$(date)"
