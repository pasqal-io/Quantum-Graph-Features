#!/bin/bash
# Optional Slurm wrapper: run a grid of run.py on whole GPU nodes. Each job takes G GPUs
# and starts P worker processes per GPU (the GT models are small and Python-bound; one
# process does not saturate an A100). Workers split the grid with --chunk.
#
#   experiments/downstream_gt/submit.sh <n_jobs> <gpus_per_job> <procs_per_gpu> <run.py args...>
#   e.g. experiments/downstream_gt/submit.sh 1 8 3 --dataset mutag --configs paper
#
# Env: PYTHON (default "uv run python"), PARTITION (default internal), GRES (default
# gpu:<gpus_per_job>), TIME (default 2-00:00:00), MEM_PER_PROC in GB (default 6),
# LOGDIR (default results/downstream_gt/logs).
# Re-submitting only runs the trials whose results.json is missing.
set -euo pipefail
N=$1; G=$2; P=$3; shift 3
PYTHON=${PYTHON:-"uv run python"}
LOGDIR=${LOGDIR:-results/downstream_gt/logs}
W=$((N * G * P))
mkdir -p "$LOGDIR"
sbatch --parsable -p "${PARTITION:-internal}" --gres="${GRES:-gpu:$G}" --cpus-per-task=$((2 * G * P)) \
    --mem=$((${MEM_PER_PROC:-6} * G * P))G --time="${TIME:-2-00:00:00}" --array=0-$((N - 1)) \
    -J dsgt -o "$LOGDIR/%A_%a.log" --wrap "
IFS=, read -ra GPUS <<< \"\$CUDA_VISIBLE_DEVICES\"
for g in \$(seq 0 $((G - 1))); do
  for p in \$(seq 0 $((P - 1))); do
    w=\$(( (SLURM_ARRAY_TASK_ID * $G + g) * $P + p ))
    CUDA_VISIBLE_DEVICES=\${GPUS[\$g]} OMP_NUM_THREADS=2 $PYTHON experiments/downstream_gt/run.py $* \
        --chunk \$w/$W > $LOGDIR/\${SLURM_ARRAY_JOB_ID}_w\$w.log 2>&1 &
  done
done
wait"
