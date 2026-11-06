"""Summarise the exact non-unit-disk benchmark (original runs or a re-run) and write the table.

Cells: mean ± std (ddof 1) of the test macro F1 over 10 folds x 5 seeds per (backbone,
config). Paired differences over the (backbone, fold, seed) trials: mean and two-sided
Wilcoxon signed-rank p-value. With ``--results`` the re-run is compared with
``expected/trials.csv`` (|delta| / std_ref per cell) and the per-trial table is written to
``rerun_trials.csv``.

    uv run python experiments/non_unit_disk/analyze.py --out experiments/non_unit_disk/expected
    uv run python experiments/non_unit_disk/analyze.py --results results/non_unit_disk/mutag \\
        --out results/non_unit_disk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

EXPECTED = Path(__file__).parent / "expected"
MODELS = ["GCN", "GAT", "GATv2", "GINE", "GEN", "Transformer", "General"]
COLUMNS = {  # config -> (column header, short name)
    "rrwp_lap_ring_geom": ("Classical\\\\ \\scriptsize $35/33$", "classical"),
    "rrwp_lap_ring_geom_af_fm": ("${+}$AFM${+}$FM\\\\ \\scriptsize emulated", "emulated"),
    "rrwp_lap_ring_geom_exact": ("${+}$AFM${+}$FM\\\\ \\scriptsize exact, $C_6/r^6$", "exact"),
    "rrwp_lap_ring_geom_exact_nn": (
        "${+}$AFM${+}$FM\\\\ \\scriptsize exact, $J_{NN}A$",
        "exact_nn",
    ),
}
PAIRS = [
    ("rrwp_lap_ring_geom_af_fm", "rrwp_lap_ring_geom"),
    ("rrwp_lap_ring_geom_exact", "rrwp_lap_ring_geom"),
    ("rrwp_lap_ring_geom_exact_nn", "rrwp_lap_ring_geom"),
    ("rrwp_lap_ring_geom_exact_nn", "rrwp_lap_ring_geom_exact"),
    ("rrwp_lap_ring_geom_exact", "rrwp_lap_ring_geom_af_fm"),
]
KEY = ["model", "config", "fold", "seed"]
EXTRA = ["test_accuracy", "best_epoch", "train_size", "val_size", "test_size", "abs_pe_dim",
         "rel_pe_dim"]  # fmt: skip


def read_tree(root: Path) -> pd.DataFrame:
    """Per-trial table of a ``<model>/<config>/fold_<k>/seed_<s>/results.json`` tree."""
    rows = []
    for p in sorted(root.glob("*/*/fold_*/seed_*/results.json")):
        r = json.loads(p.read_text())
        rows.append((r["model_type"], r["config"], r["fold"], r["seed"], r["test_f1_macro"],
                     *(r.get(k) for k in EXTRA)))  # fmt: skip
    return pd.DataFrame(rows, columns=[*KEY, "value", *EXTRA]).sort_values(KEY, ignore_index=True)


def summary(trials: pd.DataFrame) -> pd.DataFrame:
    g = trials.groupby(["model", "config"])["value"]
    return g.agg(mean="mean", std="std", n_trials="size").reset_index()


def paired(trials: pd.DataFrame) -> pd.DataFrame:
    wide = trials.pivot_table(index=["model", "fold", "seed"], columns="config", values="value")
    rows = []
    for a, b in PAIRS:
        if a not in wide or b not in wide:
            continue
        d = (wide[a] - wide[b]).dropna()
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        rows.append({"a": COLUMNS[a][1], "b": COLUMNS[b][1], "mean_diff": d.mean(),
                     "n": len(d), "wilcoxon_p": p})  # fmt: skip
    return pd.DataFrame(rows)


def table(summ: pd.DataFrame, pairs: pd.DataFrame, what: str) -> str:
    cells = summ.set_index(["model", "config"])
    configs = [c for c in COLUMNS if c in set(summ.config)]
    pair_text = "; ".join(
        f"{r.a} ${{-}}$ {r.b} $= {r.mean_diff:+.4f}$ ($p={r.wilcoxon_p:.2g}$)"
        for r in pairs.itertuples()
    )
    lines = [
        "\\begin{table}[t]",
        "\\caption{%",
        "  MUTAG, the 173 graphs with $N \\le 24$: strong classical baseline (RRWP${+}$Laplacian"
        "${+}$ring${+}$geometry) versus its extension by AFM${+}$FM features from the 1000-shot",
        "  emulation, from exact state-vector dynamics with the register couplings $C_6/r^6$, and",
        "  from exact dynamics of the graph Hamiltonian $J_{NN} A$ (not realisable by a unit-disk",
        "  register). Mean macro F$_1$ ($\\uparrow$) $\\pm$ standard deviation over 10 stratified",
        f"  CV folds $\\times$ 5 seeds ({what}). Bold: best configuration per backbone.",
        f"  Paired over (backbone, fold, seed): {pair_text}.",
        "}",
        "\\label{tab:mutag-exact-non-unit-disk}",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{l {' '.join('c' * len(configs))}}}",
        "\\toprule",
        " & " + " & ".join(f"\\shortstack{{{COLUMNS[c][0]}}}" for c in configs) + " \\\\",
        "\\midrule",
    ]
    for m in MODELS:
        if not any((m, c) in cells.index for c in configs):
            continue
        vals = {c: cells.loc[(m, c)] for c in configs if (m, c) in cells.index}
        best = max(vals, key=lambda c: vals[c]["mean"])
        parts = []
        for c in configs:
            if c not in vals:
                parts.append("---")
                continue
            mean = f"{vals[c]['mean']:.3f}"
            mean = f"\\mathbf{{{mean}}}" if c == best else mean
            parts.append(f"${mean}{{_{{\\pm {vals[c]['std']:.3f}}}}}$")
        lines.append(f"  {m:<12} & " + " & ".join(parts) + " \\\\")
    means = summ.groupby("config")["mean"].mean()
    lines += [
        "\\midrule",
        "  Mean & " + " & ".join(f"${means[c]:.3f}$" for c in configs) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=None, help="re-run tree (default: original)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    ref = pd.read_csv(EXPECTED / "trials.csv")
    trials = ref if args.results is None else read_tree(args.results)
    what = "original runs" if args.results is None else "re-run with this repository"
    summ, pairs = summary(trials), paired(trials)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = "" if args.results is None else "rerun_"
    summ.to_csv(args.out / f"{stem}summary.csv", index=False, float_format="%.6f")
    pairs.to_csv(args.out / f"{stem}paired.csv", index=False, float_format="%.6g")
    tables = args.out / "tables" / ("paper" if args.results is None else "rerun")
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "table_mutag_exact173.tex").write_text(table(summ, pairs, what))
    print(summ.pivot(index="model", columns="config", values="mean").round(4).to_string())
    print(pairs.to_string(index=False))
    if args.results is not None:
        trials.to_csv(args.out / "rerun_trials.csv", index=False)
        cmp = summ.merge(summary(ref), on=["model", "config"], suffixes=("", "_ref"))
        delta = cmp["mean"] - cmp["mean_ref"]
        cmp["delta_over_std"] = delta.abs() / cmp["std_ref"]
        se = (cmp["std"] ** 2 / cmp["n_trials"] + cmp["std_ref"] ** 2 / cmp["n_trials_ref"]) ** 0.5
        cmp["z"] = delta / se  # Welch statistic of the two cell means
        cmp.to_csv(args.out / "rerun_vs_original.csv", index=False, float_format="%.6f")
        same = trials.merge(ref, on=KEY, suffixes=("", "_ref"))
        n_same = int(((same["value"] - same["value_ref"]).abs() < 1e-9).sum())
        print(f"{len(cmp)} cells: max |delta|/std {cmp.delta_over_std.max():.3f}, "
              f"mean {cmp.delta_over_std.mean():.3f}, max |z| {cmp.z.abs().max():.2f}; "
              f"{n_same} / {len(same)} trials identical to the original")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
