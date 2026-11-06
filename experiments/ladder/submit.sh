#!/bin/bash
#SBATCH --job-name=ladder
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
# Slurm array wrapper for the ladder emulation (emu-mps, CPU): one array task per shard.
# Run `generate.py pool` first. Env: PULSE (afm|fm), NSHARDS (default 40), POOL, OUT_DIR.
#   PULSE=afm sbatch --array=0-39 experiments/ladder/submit.sh
#   PULSE=fm  sbatch --array=0-39 experiments/ladder/submit.sh
# 474 graphs x ~100 s per graph and pulse: about 20 min per shard with 40 shards.
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset VIRTUAL_ENV
PULSE=${PULSE:?set PULSE=afm or PULSE=fm}
NSHARDS=${NSHARDS:-40}
POOL=${POOL:-results/ladder/gen/pool.pt}
OUT_DIR=${OUT_DIR:-results/ladder/gen}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
echo "host=$(hostname) start=$(date) pulse=${PULSE} shard=${SHARD}/${NSHARDS}"
uv run --frozen --extra generate python -u experiments/ladder/generate.py emulate \
    --pool "${POOL}" --pulse "${PULSE}" --shard "${SHARD}" --nshards "${NSHARDS}" --cpu \
    --out "${OUT_DIR}/${PULSE}_$(printf %03d "${SHARD}").pt"
echo "end=$(date)"
