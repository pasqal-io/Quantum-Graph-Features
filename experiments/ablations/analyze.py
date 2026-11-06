"""Tables 8-10 of the ZINC_tiny ablation from the original trials or from a re-run.

Each paper cell pools the trials listed in ``expected/trials.csv`` (several training
settings and repeated seeds, as in the original analysis); a re-run fills every listed trial
with the value of its re-run (one run per distinct trial). Cells: mean ± std (ddof 1) of the
pooled MAE; the baseline row pools the ferro and antiferro no-PE cells by count (as the
original table scripts). Table 8: all model sizes pooled; Tables 9 (GCN) and 10 (GAT): one
column per size. "All feat." = ``zinc_mode = zinc``.

With ``--results`` it also writes, to ``--csv-out`` (default ``--out``), the per-trial re-run
(``rerun_trials.csv``, one row per distinct trial), the summaries of the pooled re-run
(``rerun_summary_agg.csv``, ``rerun_summary_per_size.csv``, format of ``expected/summary_*``)
and the cell-by-cell comparison with the paper (``rerun_comparison.csv``: delta and
``z = |delta| / std`` of the paper cell, ranks), and prints the comparison and the claims
(Table 8: quantum rows vs the baseline; Tables 9-10: best size per row).

    uv run python experiments/ablations/analyze.py --out experiments/ablations/expected/tables/paper
    uv run python experiments/ablations/analyze.py --results results/ablations \
        --out results/ablations/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED = Path(__file__).parent / "expected"
KEY = ["model_type", "dataset", "config", "size", "zinc_mode", "setting", "seed"]
TRIAL_COLS = ["test_mae", "test_rmse", "best_val_mae", "best_epoch", "epochs_run", "num_params"]
GROUP = ["model_type", "dataset", "config", "zinc_mode"]
SIZES = ["tiny", "small", "medium", "large", "huge", "very_huge", "massive", "giant"]
SIZE_LABELS = ["Tiny", "Small", "Medium", "Large", "Huge", "V.~Huge", "Massive", "Giant"]
ROWS = [  # (label in Table 8, label in Tables 9-10, dataset or None = both, config, section)
    ("No PE", "ZINC (Baseline)", None, "no_pe", "Baseline"),
    ("FM$_{1\\text{b}}$", "FM$_{1\\text{b}}$", "ferro", "onebody_only", "Quantum (FM)"),
    ("FM$_{2\\text{b}}$", "FM$_{2\\text{b}}$", "ferro", "twobody_only", None),
    ("FM", "FM", "ferro", "onebody_twobody", None),
    ("AFM$_{1\\text{b}}$", "AFM$_{1\\text{b}}$", "antiferro", "onebody_only", "Quantum (AFM)"),
    ("AFM$_{2\\text{b}}$", "AFM$_{2\\text{b}}$", "antiferro", "twobody_only", None),
    ("AFM", "AFM", "antiferro", "onebody_twobody", None),
]
ROW_NAMES = ["baseline", "FM_1b", "FM_2b", "FM", "AFM_1b", "AFM_2b", "AFM"]


def read_rerun(root: Path) -> pd.DataFrame:
    """One row per re-run trial (``results.json`` files under ``root``)."""
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("**/results.json"))]
    if not rows:
        raise FileNotFoundError(f"no results.json under {root}")
    return pd.DataFrame(rows)[[*KEY, *TRIAL_COLS]].sort_values(KEY).reset_index(drop=True)


def pooled_trials(rerun: pd.DataFrame | None, zinc_mode: str = "zinc") -> pd.DataFrame:
    """The paper's trial list with original values, or with re-run values (``rerun``)."""
    t = pd.read_csv(EXPECTED / "trials.csv")
    t = t[t.zinc_mode == zinc_mode]
    if rerun is None:
        return t
    t = t.drop(columns="test_mae").merge(rerun[[*KEY, "test_mae"]], on=KEY, how="left")
    return t.dropna(subset=["test_mae"])


def summaries(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(agg, per_size)``: MAE mean / std (ddof 1) / count per cell, as ``expected/summary_*``."""

    def summary(by):
        g = trials.groupby(by).test_mae
        return g.agg(MAE_mean="mean", MAE_std="std", count="size").reset_index()

    return summary(GROUP), summary([*GROUP, "size"])


def cell(trials: pd.DataFrame, model: str, dataset: str | None, config: str, size=None):
    """``(mean, std, n)``; the baseline (``dataset=None``) pools ferro and antiferro."""
    sub = trials[(trials.model_type == model) & (trials.config == config)]
    if size is not None:
        sub = sub[sub["size"] == size]
    parts = [
        sub[sub.dataset == d].test_mae
        for d in (["ferro", "antiferro"] if dataset is None else [dataset])
    ]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    n = np.array([len(p) for p in parts])
    means = np.array([p.mean() for p in parts])
    var = np.array([p.var(ddof=1) if len(p) > 1 else 0.0 for p in parts])
    return (n * means).sum() / n.sum(), np.sqrt((n * var).sum() / n.sum()), int(n.sum())


def fmt(v, rank) -> str:
    if v is None:
        return "---"
    m, s = f"{v[0]:.3f}", f"{v[1]:.3f}"
    if rank == 0:
        return f"$\\best{{{m}{{_{{\\pm {s}}}}}}}$"
    if rank == 1:
        return f"$\\second{{{m}}}{{_{{\\pm {s}}}}}$"
    return f"${m}{{_{{\\pm {s}}}}}$"


def ranks(values: dict) -> dict:
    """``{(row, column): rank}`` of the cells, by increasing mean within each column."""
    out = {}
    for c in dict.fromkeys(cc for _, cc in values):
        col = sorted((v[0], r) for (r, cc), v in values.items() if cc == c and v is not None)
        out.update({(r, c): i for i, (_, r) in enumerate(col)})
    return out


def table(values: dict, columns: list[str], col_labels: list[str], per_size: bool, header: str,
          caption: str, label: str) -> str:  # fmt: skip
    rank = {k: r for k, r in ranks(values).items() if r < 2}
    ncol = len(columns)
    lines = [
        "\\begin{table*}" if per_size else "\\begin{table}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\centering",
        f"\\begin{{tabular}}{{l {' '.join('c' * ncol)}}}",
        "\\toprule",
        f"\\multirow{{2}}{{*}}{{Method}} & \\multicolumn{{{ncol}}}{{c}}{{{header}}} \\\\",
        f"\\cmidrule(lr){{2-{ncol + 1}}}",
        " & " + " & ".join(col_labels) + " \\\\",
    ]
    for r, (lab8, lab9, _, _, section) in enumerate(ROWS):
        if section:
            lines += [
                "\\midrule",
                f"\\multicolumn{{{ncol + 1}}}{{l}}{{\\textit{{{section}}}}} \\\\",
            ]
        label_r = lab9 if per_size else lab8
        cells = [fmt(values.get((r, c)), rank.get((r, c))) for c in columns]
        lines.append(f"  {label_r} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}" if per_size else "\\end{table}", ""]
    return "\n".join(lines)


def cell_values(trials: pd.DataFrame) -> dict[str, dict]:
    """``{table: {(row index, column): (mean, std, n) or None}}`` for Tables 8, 9, 10."""
    out = {
        "table8": {
            (r, m): cell(trials, m, d, c)
            for r, (_, _, d, c, _) in enumerate(ROWS)
            for m in ("GCN", "GAT")
        }
    }
    for model, num in (("GCN", 9), ("GAT", 10)):
        out[f"table{num}"] = {
            (r, s): cell(trials, model, d, c, s)
            for r, (_, _, d, c, _) in enumerate(ROWS)
            for s in SIZES
        }
    return out


def tables(trials: pd.DataFrame, what: str) -> dict[str, str]:
    out = {}
    values = cell_values(trials)
    out["table8_ablation.tex"] = table(
        values["table8"],
        ["GCN", "GAT"],
        ["GCN", "GAT"],
        False,
        "ZINC$^{*}_{\\text{test}}$ (MAE $\\downarrow$)",
        "Quantum feature ablation on ZINC$_{\\text{tiny}}$ ($\\sim$1.5k graphs, emulation). All "
        "node and edge quantum features are used unless stated otherwise. Mean MAE "
        f"($\\downarrow$) $\\pm$ standard deviation over the pooled runs ({what}).",
        "tab:ablation-zinc-tiny",
    )
    for model, num in (("GCN", 9), ("GAT", 10)):
        out[f"table{num}_{model.lower()}_per_size.tex"] = table(
            values[f"table{num}"],
            SIZES,
            SIZE_LABELS,
            True,
            f"{model} on ZINC$_{{\\text{{test}}}}^{{*}}$ (MAE $\\downarrow$)",
            f"Model-size ablation for {model} on ZINC$_{{\\text{{tiny}}}}$ ($\\sim$1.5k graphs, "
            f"emulation). Mean MAE ($\\downarrow$) $\\pm$ standard deviation over the pooled "
            f"runs ({what}).",
            f"tab:{model.lower()}-zinc-per-size-all",
        )
    return out


def cells_frame(trials: pd.DataFrame) -> pd.DataFrame:
    """Long table of the cells of Tables 8-10: mean, std, n and rank within the column."""
    rows = []
    for name, values in cell_values(trials).items():
        rank = ranks(values)
        for (r, c), v in values.items():
            if v is not None:
                rows.append({"table": name, "row": ROW_NAMES[r], "column": c, "mean": v[0],
                             "std": v[1], "n": v[2], "rank": rank[(r, c)]})  # fmt: skip
    return pd.DataFrame(rows)


def compare(paper: pd.DataFrame, rerun: pd.DataFrame) -> pd.DataFrame:
    """Cell-by-cell comparison of two :func:`cells_frame` tables (``z = |delta| / std_paper``)."""
    m = paper.merge(rerun, on=["table", "row", "column"], suffixes=("_paper", "_rerun"))
    m["delta"] = m.mean_rerun - m.mean_paper
    m["z"] = m.delta.abs() / m.std_paper
    return m


def report(cmp: pd.DataFrame) -> str:
    """Agreement statistics per table: ``|delta| / std`` and best / runner-up per column."""
    lines = []
    for name, t in cmp.groupby("table", sort=False):
        worst = t.loc[t.z.idxmax()]
        cols = list(dict.fromkeys(t.column))
        top = {k: {c: set(t[(t.column == c) & (t[f"rank_{k}"] == 0)].row) for c in cols}
               for k in ("paper", "rerun")}  # fmt: skip
        top2 = {k: {c: set(t[(t.column == c) & (t[f"rank_{k}"] < 2)].row) for c in cols}
                for k in ("paper", "rerun")}  # fmt: skip
        lines.append(
            f"{name}: {len(t)} cells, |delta|/std < 1: {(t.z < 1).sum()}, < 2: {(t.z < 2).sum()}, "
            f"median {t.z.median():.2f}, max {worst.z:.2f} ({worst.row} / {worst.column}); "
            f"mean delta {t.delta.mean():+.4f}, max |delta| {t.delta.abs().max():.4f}; "
            f"same best row in {sum(top['paper'][c] == top['rerun'][c] for c in cols)}/"
            f"{len(cols)} columns, same best-two set in "
            f"{sum(top2['paper'][c] == top2['rerun'][c] for c in cols)}/{len(cols)}"
        )
    return "\n".join(lines)


def trial_report(rerun: pd.DataFrame) -> str:
    """Seed-level agreement: each re-run trial vs the original runs of the same trial."""
    t = pd.read_csv(EXPECTED / "trials.csv")
    ref = t.groupby(KEY).agg(orig=("test_mae", "mean"), orig_params=("num_params", "first"))
    m = rerun.merge(ref.reset_index(), on=KEY)
    d = (m.test_mae - m.orig).abs()
    return (
        f"{len(m)} trials matched: num_params equal in {(m.num_params == m.orig_params).sum()}; "
        f"|delta MAE| < 1e-4 in {(d < 1e-4).sum()}, median {d.median():.4f}, "
        f"90th percentile {d.quantile(0.9):.4f}, mean signed {(m.test_mae - m.orig).mean():+.4f}"
    )


def claims(cells: pd.DataFrame) -> str:
    """The paper's statements: Table 8 quantum rows vs the baseline; best size (Tables 9-10)."""
    lines = []
    t8 = cells[cells.table == "table8"]
    for model in ("GCN", "GAT"):
        t = t8[t8.column == model].set_index("row")
        if "baseline" not in t.index or len(t) < 3:  # partial re-run
            continue
        base = t.loc["baseline", "mean"]
        q = t.drop("baseline")
        below = list(q[q["mean"] < base].index)
        order = t.sort_values("mean").index
        lines.append(
            f"table8 {model}: baseline {base:.3f}; quantum rows below it: {len(below)}/{len(q)} "
            f"({', '.join(below)}); best {order[0]} {t.loc[order[0], 'mean']:.3f}, "
            f"runner-up {order[1]} {t.loc[order[1], 'mean']:.3f}"
        )
    for name in ("table9", "table10"):
        t = cells[cells.table == name]
        if t.empty or "large" not in set(t.column):
            continue
        best = t.loc[t.groupby("row", sort=False)["mean"].idxmin()]
        by_size = t.groupby("column")["mean"].mean().reindex(SIZES)
        glob = t.loc[t["mean"].idxmin()]
        lines.append(
            f"{name}: best size per row "
            + ", ".join(f"{r.row}={r.column}" for r in best.itertuples())
            + f"; 'large' best in {(best.column == 'large').sum()}/{len(best)} rows; "
            f"lowest row-averaged MAE at {by_size.idxmin()} ({by_size.min():.3f}; large "
            f"{by_size['large']:.3f}); lowest cell {glob.row}/{glob.column} {glob['mean']:.3f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=None, help="re-run tree (default: original)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--csv-out", type=Path, default=None, help="re-run CSVs (default: --out)")
    args = p.parse_args(argv)
    rerun = None if args.results is None else read_rerun(args.results)
    trials = pooled_trials(rerun)
    what = "original runs" if rerun is None else "re-run with this repository"
    args.out.mkdir(parents=True, exist_ok=True)
    for name, tex in tables(trials, what).items():
        (args.out / name).write_text(tex)
    print(f"tables -> {args.out}")
    paper = cells_frame(pooled_trials(None))
    if rerun is None:
        print(claims(paper))
        return 0
    ref = pooled_trials(None)
    n_distinct = len(ref.drop_duplicates(KEY))
    print(f"{len(rerun)} re-run trials ({n_distinct} distinct in the paper's list); "
          f"{len(trials)} of {len(ref)} pooled trials filled")  # fmt: skip
    csv = args.csv_out or args.out
    csv.mkdir(parents=True, exist_ok=True)
    rerun.to_csv(csv / "rerun_trials.csv", index=False, float_format="%.6f")
    agg, per_size = summaries(trials)
    agg.to_csv(csv / "rerun_summary_agg.csv", index=False, float_format="%.6f")
    per_size.to_csv(csv / "rerun_summary_per_size.csv", index=False, float_format="%.6f")
    cmp = compare(paper, cells_frame(trials))
    cmp.to_csv(csv / "rerun_comparison.csv", index=False, float_format="%.6f")
    print(f"csv -> {csv}")
    print("== per trial:", trial_report(rerun))
    print("== re-run vs paper (z = |delta| / std of the paper cell)")
    print(report(cmp))
    print("== claims, paper")
    print(claims(paper))
    print("== claims, re-run")
    print(claims(cells_frame(trials)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
