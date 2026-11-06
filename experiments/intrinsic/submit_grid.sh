#!/bin/bash
# Submit the full intrinsic-benchmark grid as one Slurm job array (one A100 per element by
# default; set GRES= and PARTITION=cpu for a CPU-only run).
#
#   experiments/intrinsic/submit_grid.sh [datasets...]   # default: MUTAG ZINC_small
#
# One array element = one (dataset, pe_type, task category), running 5 tasks in
# sequence; results go to results/intrinsic/grid/<dataset>/<pe>/<task>/result.json.
# Existing result.json files are skipped, so the script can be re-run to fill gaps.
# Prepare the caches first (see README): the first element would otherwise compute them.
# Env: PARTITION (default internal), GRES (default gpu:1; empty for CPU-only),
#      THROTTLE (max concurrent elements, default 10),
#      CPUS (per element, default 4), LOGDIR (default results/intrinsic/logs).
set -euo pipefail

DATASETS=${*:-"MUTAG ZINC_small"}
PARTITION=${PARTITION:-internal}
GRES=${GRES-gpu:1}
THROTTLE=${THROTTLE:-10}
CPUS=${CPUS:-4}
LOGDIR=${LOGDIR:-results/intrinsic/logs}
OUT=results/intrinsic/grid

mkdir -p "$LOGDIR" "$OUT"
LIST="$OUT/joblist_$(date +%Y%m%d_%H%M%S).txt"  # read by the array elements: keep it
uv run python - "$LIST" $DATASETS <<'EOF'
import sys
from qgf.intrinsic.metrics import BENCHMARK_REGISTRY
from qgf.intrinsic.run import available_pe_types

with open(sys.argv[1], "w") as f:
    for ds in sys.argv[2:]:
        for pe in available_pe_types(ds):
            for tasks in BENCHMARK_REGISTRY.values():
                f.write(f"{ds} {pe} {' '.join(tasks)}\n")
EOF
N=$(wc -l < "$LIST")
MEM=16G
GRES_OPT=""
[ -n "$GRES" ] && GRES_OPT="--gres=$GRES"
echo "$N array elements -> $OUT"
sbatch --parsable -p "$PARTITION" $GRES_OPT -c "$CPUS" --mem=$MEM -t 12:00:00 \
    -J intrinsic --array=0-$((N - 1))%$THROTTLE -o "$LOGDIR/%A_%a.log" \
    --wrap "set -e; L=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" $LIST); set -- \$L; \
OMP_NUM_THREADS=$CPUS uv run python -m qgf.intrinsic.run --dataset \$1 --pe_type \$2 \
--benchmark \${@:3} --output_dir $OUT --skip_existing"
