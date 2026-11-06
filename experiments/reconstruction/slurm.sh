#!/bin/bash
#SBATCH --job-name=recon
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=8:00:00
# Optional Slurm wrapper: runs any Python module of the experiment from the repository root.
# Memory: 160G covers the peak of the ZINC residual export (~160 GB, see README); MUTAG
# sweeps need a few GB (override with sbatch --mem=...).
# Env (read by the submit line, command-line options override the #SBATCH defaults):
#   PARTITION (default cpu), GRES (default none; CPU only), e.g.
#   sbatch -p "${PARTITION:-cpu}" ${GRES:+--gres=$GRES} experiments/reconstruction/slurm.sh ...
#   sbatch experiments/reconstruction/slurm.sh qgf.reconstruction.run sweep --dataset MUTAG ...
#   sbatch experiments/reconstruction/slurm.sh qgf.reconstruction.residual_dataset --dataset ZINC_small
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
unset VIRTUAL_ENV
module="$1"; shift
echo "host=$(hostname) start=$(date) module=${module} args=$*"
uv run --frozen python -u -m "${module}" "$@"
echo "end=$(date)"
