"""LaTeX tables of the downstream benchmark in the paper's layout (Tables 2-3, geometry table,
hardware comparison), from the published runs or from a re-run with this repository.

    # published runs (expected/), GT column as published
    uv run python experiments/downstream_gnns/tables.py --source paper \
        --out experiments/downstream_gnns/expected/tables/paper
    # re-run: GNN summaries of analyze.py, GT column from experiments/downstream_gt
    uv run python experiments/downstream_gnns/tables.py --source rerun \
        --gnn-dir results/downstream_gnns --gt-dir experiments/downstream_gt/expected \
        --out experiments/downstream_gnns/expected/tables/rerun

Cells: ``mean ± std``; GNN std = ddof 1 over the 50 MUTAG trials / 5 ZINC seeds, except the
MUTAG hardware rows (std of the 10 fold means, as published); GT std = std (ddof 0) of the
fold means (MUTAG) or of the seeds (ZINC), as in the GT port. ``\\best`` / ``\\second`` mark
the best and runner-up row of each column (macros of the paper).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
MODELS = ["GT", "GCN", "GAT", "GINE", "Transformer", "General"]
ROWS_MUTAG = [
    ("Baseline", [("No Data", "no_pe")]),
    (
        "Classical",
        [
            ("Laplacian PE", "laplacian"),
            ("RRWP", "rrwp"),
            ("RRWP + Laplacian (full)", "rrwp_laplacian"),
        ],
    ),
    (
        "Quantum",
        [
            ("AFM", "af"),
            ("AFM$_{\\text{stat}}$", "af_stat"),
            ("FM", "fm"),
            ("FM$_{\\text{stat}}$", "fm_stat"),
            ("AFM + FM", "af_fm"),
        ],
    ),
    (
        "Hybrid",
        [
            ("RRWP + AFM", "rrwp_af"),
            ("RRWP + FM", "rrwp_fm"),
            ("RRWP + AFM$_{\\text{stat}}$", "rrwp_af_stat"),
            ("RRWP + FM$_{\\text{stat}}$", "rrwp_fm_stat"),
            ("RRWP + AFM + FM", "rrwp_af_fm"),
            ("RRWP + AFM$_{\\text{stat}}$ + FM$_{\\text{stat}}$", "rrwp_af_stat_fm_stat"),
        ],
    ),
    (
        "Quantum hardware",
        [
            ("HW-AFM", "hw_afm"),
            ("HW-AFM$_{\\text{stat}}$", "hw_afm_stat"),
            ("RRWP + HW-AFM", "rrwp_hw_afm"),
            ("RRWP + HW-AFM$_{\\text{stat}}$", "rrwp_hw_afm_stat"),
        ],
    ),
]
ROWS_ZINC = ROWS_MUTAG[:-1]
HW = {"hw_afm", "hw_afm_stat", "rrwp_hw_afm", "rrwp_hw_afm_stat"}

# GT column as published (Table 2, Table 3, geometry reply); the GT port's own runs are in
# experiments/downstream_gt.
GT_PAPER = {
    "mutag": {
        "no_pe": (0.681, 0.106),
        "laplacian": (0.798, 0.051),
        "rrwp": (0.857, 0.062),
        "rrwp_laplacian": (0.852, 0.066),
        "af": (0.828, 0.056),
        "af_stat": (0.810, 0.051),
        "fm": (0.849, 0.065),
        "fm_stat": (0.836, 0.065),
        "af_fm": (0.835, 0.078),
        "rrwp_af": (0.855, 0.078),
        "rrwp_fm": (0.866, 0.060),
        "rrwp_af_stat": (0.853, 0.075),
        "rrwp_fm_stat": (0.847, 0.073),
        "rrwp_af_fm": (0.879, 0.078),
        "rrwp_af_stat_fm_stat": (0.864, 0.056),
        "hw_afm": (0.791, 0.100),
        "hw_afm_stat": (0.800, 0.083),
        "rrwp_hw_afm": (0.851, 0.074),
        "rrwp_hw_afm_stat": (0.860, 0.082),
        "rrwp_lap_ring_geom": (0.8725, 0.0520),
        "rrwp_lap_ring_geom_af_fm": (0.8832, 0.0543),
    },
    "zinc": {
        "no_pe": (0.325, 0.015),
        "laplacian": (0.236, 0.013),
        "rrwp": (0.114, 0.007),
        "rrwp_laplacian": (0.112, 0.004),
        "af": (0.163, 0.007),
        "af_stat": (0.165, 0.006),
        "fm": (0.180, 0.009),
        "fm_stat": (0.183, 0.006),
        "af_fm": (0.154, 0.009),
        "rrwp_af": (0.111, 0.004),
        "rrwp_fm": (0.118, 0.004),
        "rrwp_af_stat": (0.106, 0.005),
        "rrwp_fm_stat": (0.111, 0.007),
        "rrwp_af_fm": (0.114, 0.005),
        "rrwp_af_stat_fm_stat": (0.107, 0.005),
    },
}
# GT geometry configs use the GT port's encoders (rrwp_lap_ring_geo*), the GNNs their own
GT_NAMES = {
    "rrwp_lap_ring_geom": "rrwp_lap_ring_geo",
    "rrwp_lap_ring_geom_af_fm": "rrwp_lap_ring_geo_af_fm",
}


def gnn_cells(path: Path, paper_laplacian: Path | None = None) -> dict:
    """``{(model, config): (mean, std)}`` from a GNN summary CSV (HW rows: ``std_fold``)."""
    s = pd.read_csv(path)
    if paper_laplacian is not None:  # the published Laplacian row (1-dim Laplacian run)
        s = pd.concat([s[s.config != "laplacian"], pd.read_csv(paper_laplacian)])
    cells = {}
    for r in s.itertuples():
        std = r.std_fold if (r.config in HW and "std_fold" in s and pd.notna(r.std_fold)) else r.std
        cells[(r.model, r.config)] = (r.mean, std)
    return cells


def gt_cells(dataset: str, source: str, gt_dir: Path | None) -> dict:
    if source == "paper":
        return {("GT", c): v for c, v in GT_PAPER[dataset].items()}
    s = pd.read_csv(gt_dir / f"{dataset}_summary.csv").set_index("config")
    out = {("GT", c): (r["mean"], r["std"]) for c, r in s.iterrows()}
    for gnn_name, gt_name in GT_NAMES.items():
        if gt_name in s.index:
            out[("GT", gnn_name)] = (s.loc[gt_name, "mean"], s.loc[gt_name, "std_ddof1"])
    return out


def fmt(mean: float, std: float, rank: int | None, digits: int = 3) -> str:
    m, sd = f"{mean:.{digits}f}", f"{std:.{digits}f}"
    if rank == 0:
        return f"$\\best{{{m}{{_{{\\pm {sd}}}}}}}$"
    if rank == 1:
        return f"$\\second{{{m}}}{{_{{\\pm {sd}}}}}$"
    return f"${m}{{_{{\\pm {sd}}}}}$"


def ranks(cells: dict, model: str, configs: list[str], higher_better: bool) -> dict[str, int]:
    vals = {c: cells[(model, c)][0] for c in configs if (model, c) in cells}
    order = sorted(vals, key=vals.get, reverse=higher_better)
    return {c: i for i, c in enumerate(order[:2])}


def main_table(cells: dict, dataset: str, caption: str, label: str) -> str:
    rows = ROWS_MUTAG if dataset == "mutag" else ROWS_ZINC
    configs = [c for _, block in rows for _, c in block]
    higher = dataset == "mutag"
    rk = {m: ranks(cells, m, configs, higher) for m in MODELS}
    metric = (
        "MUTAG (F$_1$ macro $\\uparrow$)"
        if higher
        else "ZINC$^{*}_{\\text{subset}}$ (MAE $\\downarrow$)"
    )
    lines = [
        "\\begin{table*}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l c c c c c c}",
        "\\toprule",
        f"\\multirow{{2}}{{*}}{{Method}} & \\multicolumn{{6}}{{c}}{{{metric}}} \\\\",
        "\\cmidrule(lr){2-7}",
        " & " + " & ".join(MODELS) + " \\\\",
    ]
    for block, entries in rows:
        lines += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\textit{{{block}}}}} \\\\"]
        for name, c in entries:
            cells_row = [
                fmt(*cells[(m, c)], rk[m].get(c)) if (m, c) in cells else "---" for m in MODELS
            ]
            lines.append(f"  {name} & " + " & ".join(cells_row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""]
    return "\n".join(lines)


def geometry_table(cells: dict) -> str:
    """Geometry-derived controls (MUTAG): RRWP + L + Rings + Geometry vs + AFM + FM."""
    lines = [
        "\\begin{tabular}{l c c}",
        "\\toprule",
        "Backbone & RRWP + L + Rings + Geometry & + AFM + FM \\\\",
        "\\midrule",
    ]
    for m in MODELS:
        d = 4 if m == "GT" else 3
        a, b = cells.get((m, "rrwp_lap_ring_geom")), cells.get((m, "rrwp_lap_ring_geom_af_fm"))
        if a is None or b is None:
            continue
        lines.append(
            f"{m} & ${a[0]:.{d}f} \\pm {a[1]:.{d}f}$ & ${b[0]:.{d}f} \\pm {b[1]:.{d}f}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def hardware_table(cells: dict) -> str:
    """RRWP + AFM vs RRWP + HW-AFM, full and compressed (MUTAG, means)."""
    lines = [
        "\\begin{tabular}{l c c c c c c}",
        "\\toprule",
        "Backbone & RRWP+AFM (full) & RRWP+HW-AFM & $\\Delta$ (full) & "
        "RRWP+AFM$_{\\text{stat}}$ & RRWP+HW-AFM$_{\\text{stat}}$ & $\\Delta$ (stat) \\\\",
        "\\midrule",
    ]
    for m in MODELS:
        keys = ["rrwp_af", "rrwp_hw_afm", "rrwp_af_stat", "rrwp_hw_afm_stat"]
        if any((m, k) not in cells for k in keys):
            continue
        a, b, c, d = (round(cells[(m, k)][0], 3) for k in keys)  # deltas of shown values
        lines.append(
            f"{m} & {a:.3f} & {b:.3f} & {b - a:+.3f} & {c:.3f} & {d:.3f} & {d - c:+.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=["paper", "rerun"], required=True)
    p.add_argument(
        "--gnn-dir",
        type=Path,
        default=HERE / "expected",
        help="directory with <dataset>_summary.csv (rerun: analyze.py output)",
    )
    p.add_argument("--gt-dir", type=Path, default=None, help="GT summaries (rerun only)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    what = "published runs" if args.source == "paper" else "re-run with this repository"
    for ds in ("mutag", "zinc"):
        summary = args.gnn_dir / f"{ds}_summary.csv"
        if args.source == "rerun":  # expected/ holds the re-run as rerun_<dataset>_summary.csv
            summary = next(
                (f for f in (args.gnn_dir / f"rerun_{ds}_summary.csv", summary) if f.exists()),
                summary,
            )
            if (
                summary.parent.resolve() == (HERE / "expected").resolve()
                and summary.name == f"{ds}_summary.csv"
            ):
                summary = args.gnn_dir / "missing"  # never mistake the paper runs for a re-run
        if not summary.exists():
            print(f"[{ds}] no GNN summary in {args.gnn_dir}, skipped")
            continue
        lap = HERE / "expected" / f"{ds}_paper_laplacian_v2.csv" if args.source == "paper" else None
        cells = gnn_cells(summary, lap) | gt_cells(ds, args.source, args.gt_dir)
        if ds == "mutag":
            caption = (
                "Comparison of GNN architectures and feature sets on MUTAG. Mean macro F$_1$ "
                "($\\uparrow$) $\\pm$ standard deviation over 10 stratified cross-validation folds "
                f"$\\times$ 5 seeds ({what})."
            )
            (args.out / "table2_mutag.tex").write_text(
                main_table(cells, ds, caption, "tab:pe-comparison-mutag")
            )
            (args.out / "table_geometry_mutag.tex").write_text(geometry_table(cells))
            (args.out / "table_hardware_mutag.tex").write_text(hardware_table(cells))
        else:
            caption = (
                "Comparison of GNN architectures and feature sets on ZINC$^{*}_{\\text{subset}}$. "
                "Mean MAE ($\\downarrow$) $\\pm$ standard deviation over seeds (GT 10, others 5; "
                f"{what})."
            )
            (args.out / "table3_zinc.tex").write_text(
                main_table(cells, ds, caption, "tab:pe-comparison-zinc")
            )
        print(f"[{ds}] tables -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
