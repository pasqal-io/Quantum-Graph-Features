"""INTERNAL ONLY: export the per-trial results used by the App. G statistics to CSV.

Internal use only: requires Pasqal-internal packages/data; not needed to reproduce results
from the released data.

Reads the stored internal result trees (GT runs and the other-backbone runs) and writes
experiments/stats/inputs/{gt_trials.csv,other_backbones_mutag_trials.csv} with columns
arch,dataset,config,fold,seed,value (value = test macro-F1 for MUTAG, test MAE for ZINC;
config = canonical name used by qgf.stats; fold = -1 for ZINC).

    python scripts/internal/downstream_export_stats_inputs.py <graph-transformer repo>
"""

import csv
import json
import sys
from pathlib import Path

from qgf.stats.pe_benchmark import CANONICAL_CONFIGS, CANONICAL_TO_DATA, CANONICAL_TO_IGOR_ARCH

G = Path(sys.argv[1]) / "results/experiences"
GT_MUTAG = G / "PE_benchmark_MUTAG/20260312_160819"
GT_ZINC = G / "PE_benchmark_ZINC_small/20260307_090119"
IGOR = G / "igor_result"
IGOR_ROOT = {
    "rrwp_laplacian": IGOR / "20260422_185525_all_models_v3_laplacian",
    "laplacian": IGOR / "20260429_004920_all_models_v3_laplacian_only",
}
IGOR_MAIN = IGOR / "20260405_132642_multimodel_cpu_v2_large"
OUT = Path(__file__).resolve().parents[2] / "experiments/stats/inputs"


def rows_from(cfg_dir: Path, arch: str, dataset: str, config: str, key: str):
    for p in sorted(cfg_dir.glob("**/results.json")):
        parts = {q.split("_")[0]: int(q.split("_")[1]) for q in p.parent.relative_to(cfg_dir).parts}
        yield [
            arch,
            dataset,
            config,
            parts.get("fold", -1),
            parts["seed"],
            json.loads(p.read_text())[key],
        ]


def write(name: str, rows: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "dataset", "config", "fold", "seed", "value"])
        w.writerows(rows)
    print(OUT / name, len(rows))


gt, other = [], []
for cfg in CANONICAL_CONFIGS:
    d = CANONICAL_TO_DATA[cfg]
    gt += rows_from(GT_MUTAG / d, "GT", "MUTAG", cfg, "test_f1_macro")
    gt += rows_from(GT_ZINC / d, "GT", "ZINC_small", cfg, "test_mae")
    for arch, igor_arch in CANONICAL_TO_IGOR_ARCH.items():
        root = IGOR_ROOT.get(cfg, IGOR_MAIN)
        other += rows_from(root / igor_arch / d, arch, "MUTAG", cfg, "test_f1_macro")
write("gt_trials.csv", gt)
write("other_backbones_mutag_trials.csv", other)
