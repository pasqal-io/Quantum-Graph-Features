"""Convert a MUTAG result tree of the other backbones into a stats trials CSV.

    uv run python experiments/stats/gnn_trials.py results/downstream_gnns/mutag \
        results/stats/other_backbones_mutag_trials_rerun.csv

The CSV has the columns of ``other_backbones_mutag_trials.csv`` (canonical config and
architecture names, ``dataset = MUTAG``) and can replace it in the ``--trials`` list.
"""

import argparse
import csv
from pathlib import Path

from qgf.stats.pe_benchmark import read_gnn_results_tree


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("root", type=Path, help="<results>/downstream_gnns/mutag")
    p.add_argument("out", type=Path)
    args = p.parse_args(argv)
    table = read_gnn_results_tree(args.root)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "dataset", "config", "fold", "seed", "value"])
        for (arch, ds, cfg), trials in table.items():
            for (fold, seed), value in sorted(trials.items()):
                w.writerow([arch, ds, cfg, fold, seed, value])
    print(f"{sum(len(t) for t in table.values())} trials, {len(table)} cells -> {args.out}")


if __name__ == "__main__":
    main()
