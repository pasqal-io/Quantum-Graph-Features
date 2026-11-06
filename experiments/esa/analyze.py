"""Aggregate ESA trials into Table 4 (binary F1, mean ± std ddof 0 over 10 folds x 5 seeds).

    uv run python experiments/esa/analyze.py --results results/esa \
        [--expected experiments/esa/expected] [--tables results/esa/tables] \
        [--summary-out S.csv] [--trials-out T.csv]
    uv run python experiments/esa/analyze.py --from-expected \
        --tables experiments/esa/expected/tables/paper

Writes ``<results>/mutag_summary.csv`` and ``<results>/mutag_trials.csv`` (or the paths of
``--summary-out`` / ``--trials-out``); with ``--expected`` prints the difference to the
reference runs per cell (in units of the reference standard deviation) and counts the trials
identical to the reference ones (F1, accuracy, ROC-AUC, MCC and gate at the 6 decimals of
the reference CSV); ``--tables`` writes Table 4 (``table4_esa.tex``) from the re-run, or from
the original runs (``--from-expected``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED = Path(__file__).parent / "expected"
_R = r"\text{RWSE}^{*}"
EMU, HW = "Hybrid (classical ${+}$ quantum emulation)", "Hybrid (classical ${+}$ quantum hardware)"
ROWS = [  # (config, math label, section)
    ("no_pe", _R, "Baseline"),
    ("af_stat", _R + r"{+}\text{AFM}_{\text{stat}}", EMU),
    ("fm_stat", _R + r"{+}\text{FM}_{\text{stat}}", None),
    ("af", _R + r"{+}\text{AFM}", None),
    ("fm", _R + r"{+}\text{FM}", None),
    ("af_fm", _R + r"{+}\text{AFM}{+}\text{FM}", None),
    ("hw_afm_3t", _R + r"{+}\text{HW-AFM}", HW),
    ("hw_afm_3t_stat", _R + r"{+}\text{HW-AFM}_{\text{stat}}", None),
]
GATE_NAMES = {"af_stat": "AFM$_{\\mathrm{stat}}$", "fm_stat": "FM$_{\\mathrm{stat}}$", "af": "AFM",
              "fm": "FM", "af_fm": "AFM+FM", "hw_afm_3t": "HW-AFM",
              "hw_afm_3t_stat": "HW-AFM$_{\\mathrm{stat}}$"}  # fmt: skip


TRIAL_KEY = ["fusion", "config", "fold", "seed"]
METRICS = ["f1", "accuracy", "auc", "mcc", "gate_mean"]


def read_tree(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("*/*/fold_*/seed_*/results.json")):
        r = json.loads(p.read_text())
        rows.append(
            {
                **{k: r[k] for k in TRIAL_KEY},
                "f1": r["test_f1"],
                "accuracy": r["test_accuracy"],
                "auc": r["test_auc"],
                "mcc": r["test_mcc"],
                "gate_mean": r["gate_mean"],
                "abs_pe_dim": r["abs_pe_dim"],
            }
        )
    return pd.DataFrame(rows, columns=TRIAL_KEY + METRICS + ["abs_pe_dim"])


def compare_trials(trials: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Per-trial comparison with the reference CSV (values rounded to its 6 decimals)."""
    m = trials.merge(ref, on=TRIAL_KEY, suffixes=("", "_ref"), how="left", indicator=True)
    same = m["_merge"] == "both"
    for k in METRICS:
        a, b = m[k].astype(float).round(6), m[f"{k}_ref"].astype(float).round(6)
        same &= (a == b) | (a.isna() & b.isna())
    return m.assign(identical=same).drop(columns="_merge")


def aggregate(trials: pd.DataFrame) -> pd.DataFrame:
    g = trials.groupby(["fusion", "config"], sort=False)
    out = g["f1"].agg(mean="mean", n_trials="size")
    out["std"] = g["f1"].std(ddof=1)
    out["std_ddof0"] = g["f1"].std(ddof=0)
    out["gate_mean"] = g["gate_mean"].mean()
    return out.reset_index()


def latex(summary: pd.DataFrame, what: str) -> str:
    """Table 4 layout; best / runner-up over all cells by the displayed (rounded) mean."""
    cells = {(r.config, r.fusion): (r.mean, r.std_ddof0) for r in summary.itertuples()}
    order = sorted({round(m, 3) for m, _ in cells.values()}, reverse=True)
    rank = {k: order.index(round(m, 3)) for k, (m, _) in cells.items()}
    gates = summary[summary.fusion == "gated"].set_index("config")["gate_mean"]
    gate_text = ", ".join(
        f"${gates[c]:.3f}$ for {name}" for c, name in GATE_NAMES.items() if c in gates
    )

    def fmt(key) -> str:
        if key not in cells:
            return "---"
        m, sd = f"{cells[key][0]:.3f}", f"{cells[key][1]:.3f}"
        if rank[key] == 0:
            return f"$\\best{{{m}{{_{{\\pm {sd}}}}}}}$"
        if rank[key] == 1:
            return f"$\\second{{{m}}}{{_{{\\pm {sd}}}}}$"
        return f"${m}{{_{{\\pm {sd}}}}}$"

    lines = [
        "\\begin{table}[t]",
        "\\caption{%",
        "ESA model performance on MUTAG.",
        "RWSE$^{*}$ denotes RWSE augmented with ring-membership features.",
        "Mean F$_1$ of the positive class ($\\uparrow$) $\\pm$ standard deviation (population) "
        f"over 10 stratified cross-validation folds $\\times$ 5 seeds ({what}).",
        f"For gated fusion, the initial gate value is $0.88$; learned gate values are {gate_text}.",
        "\\mbox{$\\best{\\text{Best}}$} and \\mbox{$\\second{\\text{2nd best}}$} indicate the best "
        "and runner-up methods overall.",
        "}",
        "\\label{tab:mutag-esa}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l c c}",
        "\\toprule",
        "\\multirow{2}{*}{Method} & \\multicolumn{2}{c}{MUTAG (F$_1$ $\\uparrow$)} \\\\",
        "\\cmidrule(lr){2-3}",
        " & Concat. & Gated \\\\",
    ]
    for config, label, section in ROWS:
        if section:
            lines += ["\\midrule", f"\\multicolumn{{3}}{{l}}{{\\textit{{{section}}}}} \\\\"]
        row = f"  ${label}$ & {fmt((config, 'concat'))} & {fmt((config, 'gated'))}"
        lines.append(row + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=Path("results/esa"))
    p.add_argument("--expected", type=Path, default=None)
    p.add_argument("--tables", type=Path, default=None, help="write table4_esa.tex here")
    p.add_argument("--from-expected", action="store_true", help="tables of the original runs")
    p.add_argument("--summary-out", type=Path, default=None, help="default <results>/...")
    p.add_argument("--trials-out", type=Path, default=None, help="default <results>/...")
    args = p.parse_args(argv)
    if args.from_expected:
        if args.tables is None:
            p.error("--from-expected needs --tables")
        args.tables.mkdir(parents=True, exist_ok=True)
        ref = pd.read_csv(EXPECTED / "mutag_summary.csv")
        (args.tables / "table4_esa.tex").write_text(latex(ref, "original runs"))
        print(f"wrote {args.tables / 'table4_esa.tex'}")
        return 0
    trials = read_tree(args.results)
    if trials.empty:
        print(f"no results under {args.results}")
        return 1
    summary = aggregate(trials)
    summary_out = args.summary_out or args.results / "mutag_summary.csv"
    trials_out = args.trials_out or args.results / "mutag_trials.csv"
    summary.to_csv(summary_out, index=False, float_format="%.6f")
    trials.to_csv(trials_out, index=False, float_format="%.6f")
    table = summary.assign(
        cell=summary["mean"].map("{:.3f}".format)
        + " ± "
        + summary["std_ddof0"].map("{:.3f}".format)
    ).pivot(index="config", columns="fusion", values="cell")
    print(table.to_string())
    if args.tables is not None:
        args.tables.mkdir(parents=True, exist_ok=True)
        tex = latex(summary, "re-run with this repository")
        (args.tables / "table4_esa.tex").write_text(tex)
    if args.expected is not None:
        ref = pd.read_csv(args.expected / "mutag_summary.csv")
        m = summary.merge(ref, on=["fusion", "config"], suffixes=("", "_ref"))
        m["abs_delta"] = (m["mean"] - m["mean_ref"]).abs()
        m["delta_over_std"] = m["abs_delta"] / m["std_ref"]
        cols = [
            "fusion",
            "config",
            "mean",
            "mean_ref",
            "abs_delta",
            "delta_over_std",
            "gate_mean",
            "gate_mean_ref",
            "n_trials",
        ]
        print(m[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"max |delta|/std = {m['delta_over_std'].max():.2f} over {len(m)} cells")
        cmp = compare_trials(trials, pd.read_csv(args.expected / "mutag_trials.csv"))
        print(f"identical trials: {int(cmp['identical'].sum())}/{len(cmp)}")
        diff = cmp[~cmp["identical"]]
        if len(diff):
            cols = TRIAL_KEY + [c for k in METRICS for c in (k, f"{k}_ref")]
            print(diff[cols].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
