"""Paired statistical tests for the downstream feature-set sweep (paper App. G).

Port of the internal ``stats.py``. Inputs are per-trial results:

- GT (this repo's downstream benchmark): either result trees written by
  ``experiments/downstream_gt/run.py`` (``--gt-mutag-root`` / ``--gt-zinc-root``) or a
  trials CSV;
- the five other backbones on MUTAG: a trials CSV (``--trials``), whose cells can be
  replaced by those of a result tree of ``experiments/downstream_gnns/run.py``
  (``--gnn-mutag-root``);
- the five other backbones on ZINC*_subset: only (mean, std) summaries are available,
  hard-coded below (Table 3).

Trials CSV columns: ``arch,dataset,config,fold,seed,value`` with ``dataset`` in
{MUTAG, ZINC_small}, ``config`` a canonical name (see ``CANONICAL_CONFIGS``),
``fold = -1`` for ZINC and ``value`` = test macro-F1 (MUTAG) or test MAE (ZINC).

Outputs (in ``--output-dir``): cell_aggregates.csv, cell_trials.pkl, stats_report.md,
bayes_*.pdf/png, and baycomp's last-sample.pickle (Stan posterior cache).

The hierarchical test needs pystan (extra ``stats``); httpstan's compiled models need a
libstdc++ from GCC >= 11 at runtime (on RHEL 8 e.g. ``module load compilers/gcc/13.4.0``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Constants and mappings
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_CONFIGS: list[str] = [
    "no_pe",
    "random",
    "laplacian",
    "rrwp",
    "rrwp_laplacian",
    "afm",
    "afm_stat",
    "fm",
    "fm_stat",
    "afm_fm",
    "rrwp_afm",
    "rrwp_fm",
    "rrwp_afm_stat",
    "rrwp_fm_stat",
    "rrwp_afm_fm",
    "rrwp_afm_stat_fm_stat",
]

CLASSICAL_BLOCK: frozenset[str] = frozenset(
    {"no_pe", "random", "laplacian", "rrwp", "rrwp_laplacian"}
)
HYBRID_BLOCK: frozenset[str] = frozenset(
    {
        "afm",
        "afm_stat",
        "fm",
        "fm_stat",
        "afm_fm",
        "rrwp_afm",
        "rrwp_fm",
        "rrwp_afm_stat",
        "rrwp_fm_stat",
        "rrwp_afm_fm",
        "rrwp_afm_stat_fm_stat",
    }
)

HARDCODED_CLASSICAL_CHAMPION = "rrwp"
HARDCODED_QUANTUM_CHAMPION = "rrwp_afm_stat_fm_stat"

ARCHITECTURES: list[str] = ["GT", "GCN", "GAT", "GINE", "Transformer", "GeneralConv"]
OTHER_ARCHITECTURES: list[str] = ["GCN", "GAT", "GINE", "Transformer", "GeneralConv"]
DATASETS: list[str] = ["MUTAG", "ZINC_small"]

# Canonical config -> folder name of the GT result trees
CANONICAL_TO_DATA: dict[str, str] = {
    "no_pe": "no_pe",
    "random": "random",
    "laplacian": "laplacian",
    "rrwp": "rrwp",
    "rrwp_laplacian": "rrwp_laplacian",
    "afm": "af",
    "afm_stat": "af_stat",
    "fm": "fm",
    "fm_stat": "fm_stat",
    "afm_fm": "af_fm",
    "rrwp_afm": "rrwp_af",
    "rrwp_fm": "rrwp_fm",
    "rrwp_afm_stat": "rrwp_af_stat",
    "rrwp_fm_stat": "rrwp_fm_stat",
    "rrwp_afm_fm": "rrwp_af_fm",
    "rrwp_afm_stat_fm_stat": "rrwp_af_stat_fm_stat",
}

# Canonical arch -> folder name in the other-backbone result trees
CANONICAL_TO_IGOR_ARCH: dict[str, str] = {
    "GAT": "GAT",
    "GCN": "GCN",
    "GINE": "GINE",
    "Transformer": "Transformer",
    "GeneralConv": "General",
}

GT_MISSING_CONFIGS: frozenset[str] = frozenset()

# ZINC*_subset summaries for the non-GT architectures (Table 3):
# {canonical_config: {arch: (mean_mae, std_mae)}}
ZINC_SMALL_SUMMARY: dict[str, dict[str, tuple[float, float]]] = {
    "no_pe": {
        "GCN": (0.591, 0.066),
        "GAT": (0.434, 0.008),
        "GINE": (0.236, 0.041),
        "Transformer": (0.339, 0.009),
        "GeneralConv": (0.331, 0.006),
    },
    "random": {
        "GCN": (0.584, 0.004),
        "GAT": (0.492, 0.010),
        "GINE": (0.382, 0.006),
        "Transformer": (0.438, 0.010),
        "GeneralConv": (0.407, 0.008),
    },
    "laplacian": {
        "GCN": (0.520, 0.057),
        "GAT": (0.430, 0.006),
        "GINE": (0.180, 0.009),
        "Transformer": (0.275, 0.023),
        "GeneralConv": (0.290, 0.005),
    },
    "rrwp": {
        "GCN": (0.252, 0.006),
        "GAT": (0.223, 0.002),
        "GINE": (0.117, 0.007),
        "Transformer": (0.126, 0.006),
        "GeneralConv": (0.148, 0.004),
    },
    "rrwp_laplacian": {
        "GCN": (0.278, 0.017),
        "GAT": (0.229, 0.007),
        "GINE": (0.117, 0.007),
        "Transformer": (0.123, 0.008),
        "GeneralConv": (0.145, 0.005),
    },
    "afm": {
        "GCN": (0.339, 0.040),
        "GAT": (0.288, 0.010),
        "GINE": (0.188, 0.002),
        "Transformer": (0.191, 0.007),
        "GeneralConv": (0.214, 0.006),
    },
    "afm_stat": {
        "GCN": (0.419, 0.071),
        "GAT": (0.360, 0.024),
        "GINE": (0.192, 0.007),
        "Transformer": (0.195, 0.006),
        "GeneralConv": (0.218, 0.009),
    },
    "fm": {
        "GCN": (0.479, 0.017),
        "GAT": (0.430, 0.005),
        "GINE": (0.248, 0.040),
        "Transformer": (0.283, 0.036),
        "GeneralConv": (0.325, 0.005),
    },
    "fm_stat": {
        "GCN": (0.465, 0.016),
        "GAT": (0.413, 0.004),
        "GINE": (0.208, 0.012),
        "Transformer": (0.292, 0.013),
        "GeneralConv": (0.309, 0.005),
    },
    "afm_fm": {
        "GCN": (0.314, 0.020),
        "GAT": (0.304, 0.009),
        "GINE": (0.189, 0.003),
        "Transformer": (0.195, 0.012),
        "GeneralConv": (0.215, 0.012),
    },
    "rrwp_afm": {
        "GCN": (0.255, 0.011),
        "GAT": (0.215, 0.008),
        "GINE": (0.127, 0.008),
        "Transformer": (0.128, 0.006),
        "GeneralConv": (0.157, 0.007),
    },
    "rrwp_fm": {
        "GCN": (0.285, 0.016),
        "GAT": (0.242, 0.008),
        "GINE": (0.130, 0.004),
        "Transformer": (0.146, 0.006),
        "GeneralConv": (0.166, 0.005),
    },
    "rrwp_afm_stat": {
        "GCN": (0.244, 0.004),
        "GAT": (0.210, 0.011),
        "GINE": (0.124, 0.009),
        "Transformer": (0.125, 0.006),
        "GeneralConv": (0.150, 0.006),
    },
    "rrwp_fm_stat": {
        "GCN": (0.257, 0.010),
        "GAT": (0.228, 0.004),
        "GINE": (0.121, 0.009),
        "Transformer": (0.120, 0.010),
        "GeneralConv": (0.145, 0.006),
    },
    "rrwp_afm_fm": {
        "GCN": (0.262, 0.011),
        "GAT": (0.235, 0.006),
        "GINE": (0.135, 0.011),
        "Transformer": (0.147, 0.004),
        "GeneralConv": (0.165, 0.010),
    },
    "rrwp_afm_stat_fm_stat": {
        "GCN": (0.247, 0.008),
        "GAT": (0.209, 0.004),
        "GINE": (0.116, 0.009),
        "Transformer": (0.128, 0.002),
        "GeneralConv": (0.149, 0.008),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loading
# ─────────────────────────────────────────────────────────────────────────────

# Pairing key: (fold, seed) for MUTAG; (-1, seed) for ZINC_small (no folds)
TrialKey = tuple[int, int]
CellKey = tuple[str, str, str]  # (arch, dataset, canonical_config)
TrialTable = dict[CellKey, dict[TrialKey, float]]


def read_trials_csv(path: Path) -> TrialTable:
    """Trials CSV -> {(arch, dataset, config): {(fold, seed): value}} (file order kept)."""
    table: TrialTable = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["arch"], r["dataset"], r["config"])
            table.setdefault(key, {})[(int(r["fold"]), int(r["seed"]))] = float(r["value"])
    return table


def read_gt_results_tree(root: Path, dataset: str) -> TrialTable:
    """GT result tree (``<cfg>/[fold_k/]seed_s/results.json``) -> trials table.

    Same traversal order as the original loaders (sorted fold dirs, sorted seed dirs).
    """
    metric = "test_f1_macro" if dataset == "MUTAG" else "test_mae"
    table: TrialTable = {}
    for cfg in CANONICAL_CONFIGS:
        cfg_dir = root / CANONICAL_TO_DATA[cfg]
        if not cfg_dir.exists():
            continue
        result: dict[TrialKey, float] = {}
        fold_dirs = (
            [d for d in sorted(cfg_dir.iterdir()) if d.is_dir() and d.name.startswith("fold_")]
            if dataset == "MUTAG"
            else [cfg_dir]
        )
        for fold_dir in fold_dirs:
            fold = int(fold_dir.name.split("_")[1]) if dataset == "MUTAG" else -1
            for seed_dir in sorted(fold_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                p = seed_dir / "results.json"
                if p.exists():
                    seed = int(seed_dir.name.split("_")[1])
                    result[(fold, seed)] = json.loads(p.read_text())[metric]
        if result:
            table[("GT", dataset, cfg)] = result
    return table


def read_gnn_results_tree(root: Path) -> TrialTable:
    """MUTAG result tree of the other backbones (``<arch>/<cfg>/fold_k/seed_s/results.json``,
    ``experiments/downstream_gnns/run.py``) -> trials table, for the cells present."""
    table: TrialTable = {}
    for arch, arch_dir in CANONICAL_TO_IGOR_ARCH.items():
        for cfg in CANONICAL_CONFIGS:
            result: dict[TrialKey, float] = {}
            for p in sorted(
                (root / arch_dir / CANONICAL_TO_DATA[cfg]).glob("fold_*/seed_*/results.json")
            ):
                r = json.loads(p.read_text())
                result[(int(r["fold"]), int(r["seed"]))] = r["test_f1_macro"]
            if result:
                table[(arch, "MUTAG", cfg)] = result
    return table


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build cell_trials and cell_aggregates
# ─────────────────────────────────────────────────────────────────────────────

CellTrials = dict[CellKey, np.ndarray | None]  # None = summary-only


def build_all_trials(trials: TrialTable) -> CellTrials:
    cell: CellTrials = {}
    for arch in ARCHITECTURES:
        for ds in ("MUTAG",) if arch != "GT" else ("MUTAG", "ZINC_small"):
            for cfg in CANONICAL_CONFIGS:
                if arch == "GT" and cfg in GT_MISSING_CONFIGS:
                    continue
                t = trials.get((arch, ds, cfg))
                if t:
                    cell[(arch, ds, cfg)] = np.array(list(t.values()))
                elif arch != "GT":
                    print(f"  WARNING: missing trials for {(arch, ds, cfg)}")
    for cfg, arch_data in ZINC_SMALL_SUMMARY.items():
        for arch in arch_data:
            cell[(arch, "ZINC_small", cfg)] = None  # summary only
    return cell


def _cell_stats(key: CellKey, cell: CellTrials) -> tuple[float, float, float, int, bool]:
    """Return (mean, std_ddof0, std_ddof1, n, is_summary)."""
    arch, _dataset, config = key
    trials = cell.get(key)
    if trials is not None:
        n = len(trials)
        return (
            float(np.mean(trials)),
            float(np.std(trials, ddof=0)),
            float(np.std(trials, ddof=1)) if n > 1 else float("nan"),
            n,
            False,
        )
    mean, std = ZINC_SMALL_SUMMARY[config][arch]
    return mean, std, float("nan"), 0, True


def build_cell_aggregates(cell: CellTrials) -> pd.DataFrame:
    rows = []
    for key in cell:
        arch, dataset, config = key
        mean, std0, std1, n, is_summary = _cell_stats(key, cell)
        rows.append(
            {
                "arch": arch,
                "dataset": dataset,
                "config": config,
                "n_trials": n,
                "mean": mean,
                "std_ddof0": std0,
                "std_ddof1": std1,
                "is_summary": is_summary,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "arch", "config"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Helpers: goodness, signed diff, cell lookup
# ─────────────────────────────────────────────────────────────────────────────


def goodness(mean: float, dataset: str) -> float:
    """Higher is better in both datasets."""
    return mean if dataset == "MUTAG" else -mean


def signed_diff(quantum_mean: float, classical_mean: float, dataset: str) -> float:
    """Positive = quantum better (sign-corrected)."""
    if dataset == "MUTAG":
        return quantum_mean - classical_mean
    return classical_mean - quantum_mean  # lower MAE = better -> flip


def get_cell(df: pd.DataFrame, arch: str, dataset: str, config: str) -> tuple[float, float] | None:
    """Return (mean, std_ddof0) or None if not found."""
    row = df[(df.arch == arch) & (df.dataset == dataset) & (df.config == config)]
    if row.empty:
        return None
    r = row.iloc[0]
    return float(r["mean"]), float(r["std_ddof0"])


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Champion identification (artefact 5a)
# ─────────────────────────────────────────────────────────────────────────────


def find_champion(df: pd.DataFrame, block: frozenset[str]) -> str:
    """Config in block with highest average goodness over all available cells."""
    sub = df[df["config"].isin(block)].copy()
    sub = sub[~((sub["arch"] == "GT") & (sub["config"].isin(GT_MISSING_CONFIGS)))]
    sub["_g"] = sub.apply(lambda r: goodness(r["mean"], r["dataset"]), axis=1)
    return str(sub.groupby("config")["_g"].mean().idxmax())


def assert_champions(df: pd.DataFrame) -> tuple[str, str]:
    classical = find_champion(df, CLASSICAL_BLOCK)
    quantum = find_champion(df, HYBRID_BLOCK)
    if classical != HARDCODED_CLASSICAL_CHAMPION:
        raise AssertionError(
            f"Classical champion mismatch: recomputed={classical!r}, "
            f"hardcoded={HARDCODED_CLASSICAL_CHAMPION!r}"
        )
    if quantum != HARDCODED_QUANTUM_CHAMPION:
        raise AssertionError(
            f"Quantum champion mismatch: recomputed={quantum!r}, "
            f"hardcoded={HARDCODED_QUANTUM_CHAMPION!r}"
        )
    print(f"  Classical champion: {classical}")
    print(f"  Quantum champion:   {quantum}")
    return classical, quantum


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Frequentist sign tests (artefact 5b and 5c)
# ─────────────────────────────────────────────────────────────────────────────


def sign_test_5b(df: pd.DataFrame, classical: str, quantum: str) -> dict[str, Any]:
    """5(b): champion vs champion, one-sided sign test over 12 cells."""
    per_cell: list[dict[str, Any]] = []
    diffs: list[float] = []
    for arch in ARCHITECTURES:
        for ds in DATASETS:
            cl = get_cell(df, arch, ds, classical)
            qu = get_cell(df, arch, ds, quantum)
            if cl is None or qu is None:
                print(f"    [5b] SKIP ({arch}, {ds}): missing data")
                continue
            cm, cstd = cl
            qm, qstd = qu
            diff = signed_diff(qm, cm, ds)
            pooled = np.sqrt(cstd**2 + qstd**2)
            diffs.append(diff)
            per_cell.append(
                {
                    "arch": arch,
                    "dataset": ds,
                    "classical_mean": cm,
                    "quantum_mean": qm,
                    "diff": diff,
                    "pooled_noise": float(pooled),
                    "within_noise": bool(abs(diff) < 0.5 * pooled),
                }
            )
    k = int(sum(d > 0 for d in diffs))
    n = len(diffs)
    binom = scipy_stats.binomtest(k, n, p=0.5, alternative="greater")
    return {
        "k_wins": k,
        "n_cells": n,
        "p_value": float(binom.pvalue),
        "per_cell": per_cell,
        "diffs": diffs,
    }


def _best_in_block(df: pd.DataFrame, arch: str, dataset: str, block: frozenset[str]) -> str | None:
    """Best config in block for (arch, dataset): best mean, tie -> lower std."""
    cands = df[(df.arch == arch) & (df.dataset == dataset) & (df.config.isin(block))]
    if cands.empty:
        return None
    best_mean = cands["mean"].max() if dataset == "MUTAG" else cands["mean"].min()
    tied = cands[np.isclose(cands["mean"], best_mean)]
    return str(tied.loc[tied["std_ddof0"].idxmin(), "config"])


def sign_test_5c(df: pd.DataFrame) -> dict[str, Any]:
    """5(c): best-of-class per cell sign test (exploratory sensitivity check)."""
    per_cell: list[dict[str, Any]] = []
    diffs_nontied: list[float] = []
    for arch in ARCHITECTURES:
        for ds in DATASETS:
            best_cl = _best_in_block(df, arch, ds, CLASSICAL_BLOCK)
            best_hy = _best_in_block(df, arch, ds, HYBRID_BLOCK)
            if best_cl is None or best_hy is None:
                print(f"    [5c] SKIP ({arch}, {ds}): no best found")
                continue
            cl = get_cell(df, arch, ds, best_cl)
            qu = get_cell(df, arch, ds, best_hy)
            if cl is None or qu is None:
                continue
            diff = signed_diff(qu[0], cl[0], ds)
            literal_tie = abs(diff) < 1e-4
            per_cell.append(
                {
                    "arch": arch,
                    "dataset": ds,
                    "best_classical": best_cl,
                    "best_hybrid": best_hy,
                    "classical_mean": cl[0],
                    "hybrid_mean": qu[0],
                    "diff": diff,
                    "literal_tie": literal_tie,
                }
            )
            if not literal_tie:
                diffs_nontied.append(diff)
    k = int(sum(d > 0 for d in diffs_nontied))
    n = len(diffs_nontied)
    binom = scipy_stats.binomtest(k, n, p=0.5, alternative="greater") if n > 0 else None
    return {
        "k_wins": k,
        "n_cells": n,
        "p_value": float(binom.pvalue) if binom else float("nan"),
        "per_cell": per_cell,
        "diffs": diffs_nontied,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Paired MUTAG trials (needed for 6b/6c)
# ─────────────────────────────────────────────────────────────────────────────


def load_paired_mutag(
    trials: TrialTable, classical_cfg: str, quantum_cfg: str
) -> dict[str, dict[TrialKey, tuple[float, float]]]:
    """(classical_f1, quantum_f1) pairs keyed by (fold, seed), per architecture."""
    paired: dict[str, dict[TrialKey, tuple[float, float]]] = {}
    for arch in ARCHITECTURES:
        cl = trials.get((arch, "MUTAG", classical_cfg), {})
        qu = trials.get((arch, "MUTAG", quantum_cfg), {})
        common = set(cl) & set(qu)
        paired[arch] = {k: (cl[k], qu[k]) for k in sorted(common)}
    return paired


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Bayesian tests (artefact 6)
# ─────────────────────────────────────────────────────────────────────────────


def _fix_simplex_labels(fig: Any) -> None:
    """Push the two bottom-corner labels of a baycomp simplex outward to avoid overlap."""
    ax = fig.get_axes()[0]
    for txt in ax.texts:
        xp, yp = txt.get_position()
        if yp < 0:
            if xp < 0.3:
                txt.set_position((xp - 0.1, yp - 0.07))
                txt.set_ha("right")
            elif xp > 0.7:
                txt.set_position((xp + 0.1, yp - 0.07))
                txt.set_ha("left")


def _ci95(arr: Any, col: int) -> tuple[float, float]:
    """95% interval (2.5th-97.5th percentile) from posterior samples."""
    lo, hi = np.percentile(arr[:, col], [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_probs_ci(
    test_cls: Any,
    x: np.ndarray,
    y: np.ndarray,
    rope: float,
    n_boots: int,
    nsamples: int,
    seed: int = 42,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Bootstrap 95% CI on (p_left, p_rope, p_right) by resampling paired observations."""
    rng = np.random.default_rng(seed)
    n = len(x)
    pl_arr: list[float] = []
    pe_arr: list[float] = []
    pr_arr: list[float] = []
    for _ in range(n_boots):
        idx = rng.integers(0, n, size=n)
        pl, pe, pr = test_cls.probs(x[idx], y[idx], rope=rope, nsamples=nsamples)
        pl_arr.append(float(pl))
        pe_arr.append(float(pe))
        pr_arr.append(float(pr))

    def ci(a: list[float]) -> tuple[float, float]:
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return ci(pl_arr), ci(pe_arr), ci(pr_arr)


def _plot_probs_bar(
    p_quantum: float,
    p_rope: float,
    p_classical: float,
    ci_quantum: tuple[float, float],
    ci_rope: tuple[float, float],
    ci_classical: tuple[float, float],
    rope: float,
    stem: str,
) -> bool:
    """Horizontal bar chart for a 3-outcome Bayesian test (fallback for the 6c plot)."""
    cats = ["P(quantum)", "P(rope / equiv)", "P(classical)"]
    vals = [p_quantum, p_rope, p_classical]
    cis = [ci_quantum, ci_rope, ci_classical]
    try:
        fig, ax = plt.subplots(figsize=(5.5, 2.6))
        bars = ax.barh(
            cats, vals, color=["#00C887", "#AAAAAA", "#FF986E"], edgecolor="#333333", linewidth=0.5
        )
        for bar, v, (lo, hi) in zip(bars, vals, cis):
            cy = bar.get_y() + bar.get_height() / 2
            ax.errorbar(
                v,
                cy,
                xerr=[[v - lo], [hi - v]],
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=3,
                capthick=0.8,
            )
            ax.text(
                max(v, hi) + 0.015, cy, f"{v:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=8
            )
        ax.set_xlim(0, 1.35)
        ax.set_xlabel("Posterior probability")
        ax.set_title(f"6(c) Hierarchical Bayesian test — MUTAG (ROPE ±{rope} F1)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(f"{stem}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  [6c] Plot error: {e}")
        return False


def bayes_sign_test_6a(
    df: pd.DataFrame, classical: str, quantum: str, rope: float = 0.5
) -> dict[str, Any]:
    """6(a): Bayesian sign test on the 12 cell-level normalised diffs."""
    import baycomp

    diffs_norm: list[float] = []
    for arch in ARCHITECTURES:
        for ds in DATASETS:
            cl = get_cell(df, arch, ds, classical)
            qu = get_cell(df, arch, ds, quantum)
            if cl is None or qu is None:
                continue
            cm, cstd = cl
            qm, qstd = qu
            diff = signed_diff(qm, cm, ds)
            pooled = float(np.sqrt(cstd**2 + qstd**2))
            diffs_norm.append(diff / pooled if pooled > 0 else 0.0)

    x = np.array(diffs_norm)
    y = np.zeros(len(x))
    # x encodes quantum advantage: baycomp P(left) = P(quantum clearly better).
    test = baycomp.SignTest(x, y, rope, nsamples=50000)
    p_left, p_rope, p_right = test.probs()
    print("    Computing bootstrap CI for 6a (2000 boots)…")
    ci_left, ci_rope, ci_right = _bootstrap_probs_ci(
        baycomp.SignTest, x, y, rope, n_boots=2000, nsamples=50000
    )
    plot_saved = False
    try:
        fig = test.plot(names=("Quantum", "Classical"))
        _fix_simplex_labels(fig)
        fig.savefig("bayes_sign_test_cell.pdf", bbox_inches="tight")
        fig.savefig("bayes_sign_test_cell.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        plot_saved = True
    except Exception as e:
        print(f"  [6a] Plot error: {e}")
    return {
        "p_left": float(p_left),
        "p_rope": float(p_rope),
        "p_right": float(p_right),
        "ci_left": ci_left,
        "ci_rope": ci_rope,
        "ci_right": ci_right,
        "n_cells": len(x),
        "rope": rope,
        "diffs_norm": diffs_norm,
        "plot_saved": plot_saved,
    }


def bayes_signed_rank_6b(
    trials: TrialTable, classical: str, quantum: str, rope: float = 0.005
) -> dict[str, Any]:
    """6(b): Bayesian signed-rank test on the pooled MUTAG paired trials."""
    import baycomp

    paired = load_paired_mutag(trials, classical, quantum)
    cl_all: list[float] = []
    qu_all: list[float] = []
    per_arch_n: dict[str, int] = {}
    for arch in ARCHITECTURES:
        pairs = paired.get(arch, {})
        per_arch_n[arch] = len(pairs)
        for cl_f1, qu_f1 in pairs.values():
            cl_all.append(cl_f1)
            qu_all.append(qu_f1)
    x = np.array(qu_all)
    y = np.array(cl_all)

    # Median per-cell std across MUTAG cells (all archs x both configs), for the report
    cell_stds = [
        float(np.std(list(t.values()), ddof=1))
        for arch in ARCHITECTURES
        for cfg in (classical, quantum)
        if (t := trials.get((arch, "MUTAG", cfg)))
    ]
    median_cell_std = float(np.median(cell_stds)) if cell_stds else float("nan")

    test = baycomp.SignedRankTest(x, y, rope, nsamples=50000)
    p_left, p_rope, p_right = test.probs()
    print("    Computing bootstrap CI for 6b (500 boots)…")
    ci_left, ci_rope, ci_right = _bootstrap_probs_ci(
        baycomp.SignedRankTest, x, y, rope, n_boots=500, nsamples=1000
    )
    plot_saved = False
    try:
        fig = test.plot(names=("Quantum", "Classical"))
        _fix_simplex_labels(fig)
        fig.savefig("bayes_signedrank_mutag.pdf", bbox_inches="tight")
        fig.savefig("bayes_signedrank_mutag.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        plot_saved = True
    except Exception as e:
        print(f"  [6b] Plot error: {e}")
    return {
        "p_left": float(p_left),
        "p_rope": float(p_rope),
        "p_right": float(p_right),
        "ci_left": ci_left,
        "ci_rope": ci_rope,
        "ci_right": ci_right,
        "n_pairs": len(x),
        "rope": rope,
        "median_cell_std": median_cell_std,
        "per_arch_n": per_arch_n,
        "plot_saved": plot_saved,
        "x_arr": x.tolist(),
        "y_arr": y.tolist(),
    }


def bayes_hierarchical_6c(
    trials: TrialTable, classical: str, quantum: str, rope: float = 0.005
) -> dict[str, Any]:
    """6(c): hierarchical Bayesian test (pystan); skipped if Stan is unavailable."""
    try:
        import baycomp
    except ImportError:
        return {"skipped": True, "reason": "baycomp not installed"}

    paired = load_paired_mutag(trials, classical, quantum)
    min_n = min(len(v) for v in paired.values() if v)
    x_mats, y_mats = [], []
    for arch in ARCHITECTURES:
        items = sorted(paired.get(arch, {}).items())[:min_n]
        y_mats.append([v[0] for _, v in items])  # classical
        x_mats.append([v[1] for _, v in items])  # quantum
    x = np.array(x_mats)  # (6, min_n)
    y = np.array(y_mats)

    try:
        p_left, p_rope, p_right = baycomp.HierarchicalTest.probs(x, y, rope=rope, nsamples=50000)
        # Re-call sample(): reads baycomp's pickle cache, no extra Stan run.
        # Columns: [0=p_right, 1=p_rope, 2=p_left].
        samples_6c = baycomp.HierarchicalTest.sample(x, y, rope=rope, nsamples=50000)
        ci_left, ci_rope, ci_right = (
            _ci95(samples_6c, 2),
            _ci95(samples_6c, 1),
            _ci95(samples_6c, 0),
        )
        plot_saved = False
        try:
            fig = baycomp.HierarchicalTest.plot(
                x, y, rope=rope, nsamples=50000, names=("Quantum", "Classical")
            )
            _fix_simplex_labels(fig)
            fig.savefig("bayes_hierarchical_mutag.pdf", bbox_inches="tight")
            fig.savefig("bayes_hierarchical_mutag.png", bbox_inches="tight", dpi=300)
            plt.close(fig)
            plot_saved = True
        except Exception as pe:
            print(f"  [6c] baycomp plot failed ({pe}), using bar chart fallback")
            plot_saved = _plot_probs_bar(
                float(p_left),
                float(p_rope),
                float(p_right),
                ci_left,
                ci_rope,
                ci_right,
                rope,
                "bayes_hierarchical_mutag",
            )
        return {
            "p_left": float(p_left),
            "p_rope": float(p_rope),
            "p_right": float(p_right),
            "ci_left": ci_left,
            "ci_rope": ci_rope,
            "ci_right": ci_right,
            "n_arches": len(ARCHITECTURES),
            "n_per_arch": min_n,
            "rope": rope,
            "plot_saved": plot_saved,
            "x_arches": x.tolist(),
            "y_arches": y.tolist(),
        }
    except ImportError as e:
        return {"skipped": True, "reason": f"Stan not available ({e}); install extra 'stats'."}
    except Exception as e:
        # Typically httpstan's compiled model failing to load against an old libstdc++.
        return {"skipped": True, "reason": f"Stan run failed (libstdc++ >= GCC 11 needed?): {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Report generation
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def _pval_str(p: float) -> str:
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def _ci_str(ci: tuple[float, float] | None) -> str:
    return "N/A" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def write_report(
    classical: str,
    quantum: str,
    r5b: dict[str, Any],
    r5c: dict[str, Any],
    r6a: dict[str, Any],
    r6b: dict[str, Any],
    r6c: dict[str, Any],
    out: Path = Path("stats_report.md"),
) -> None:
    median_std = r6b.get("median_cell_std", float("nan"))
    rows5b = ["| arch | dataset | Δ (sign-corrected) | within noise? |", "|---|---|---|---|"]
    for e in r5b["per_cell"]:
        rows5b.append(
            f"| {e['arch']} | {e['dataset']} | {_fmt(e['diff'])} | {'✓' if e['within_noise'] else ''} |"
        )
    rows5c = [
        "| arch | dataset | best classical | best hybrid | Δ | tie? |",
        "|---|---|---|---|---|---|",
    ]
    for e in r5c["per_cell"]:
        rows5c.append(
            f"| {e['arch']} | {e['dataset']} | {e['best_classical']} "
            f"| {e['best_hybrid']} | {_fmt(e['diff'])} | {'✓' if e['literal_tie'] else ''} |"
        )
    per_arch_n = ", ".join(f"{a}: {n}" for a, n in r6b.get("per_arch_n", {}).items())
    if r6c.get("skipped"):
        sec_6c = f"### 6(c) Hierarchical test (Corani-Benavoli 2017)\n\n**Skipped.** {r6c.get('reason', '')}"
    else:
        sec_6c = "\n".join(
            [
                "### 6(c) Hierarchical test (Corani-Benavoli 2017)",
                "",
                "Convention: P(left) = P(quantum better), P(right) = P(classical better).",
                "",
                "| | value | 95% CI |",
                "|---|---|---|",
                f"| P(quantum better)   | {_fmt(r6c['p_left'])} | {_ci_str(r6c.get('ci_left'))} |",
                f"| P(rope / equiv)     | {_fmt(r6c['p_rope'])} | {_ci_str(r6c.get('ci_rope'))} |",
                f"| P(classical better) | {_fmt(r6c['p_right'])} | {_ci_str(r6c.get('ci_right'))} |",
                f"| architectures       | {r6c['n_arches']} | |",
                f"| pairs per arch      | {r6c['n_per_arch']} | |",
                f"| rope                | ±{r6c['rope']} F1 | |",
                "",
                f"Plot: `bayes_hierarchical_mutag.pdf` + `.png` (saved: {r6c.get('plot_saved', False)})",
            ]
        )

    sections = [
        "# Statistical analysis — downstream feature-set sweep",
        "",
        "## Champions (artefact 5a)",
        "",
        f"Classical champion (best avg across 12 cells): **{classical}**",
        f"Quantum champion (best avg across 12 cells):   **{quantum}**",
        "",
        "---",
        "",
        "## ROPE choice",
        "",
        "**Cell-level test (6a):** rope = ±0.5 in normalised units",
        r"(each paired mean diff is divided by $\sqrt{\sigma_{\text{cl}}^2 + \sigma_{\text{qu}}^2}$).",
        "",
        "**Trial-level test (6b):** rope = ±0.01 F1 points.",
        f"Median within-cell std across MUTAG cells for the two champion configs is **{_fmt(median_std)} F1**.",
        f"The rope (0.01) is ~{0.01 / median_std:.2f}× the typical cell noise.",
        "",
        "---",
        "",
        "## Artefact 5b — Champion vs champion frequentist sign test",
        "",
        f"H₁: quantum (`{quantum}`) better than classical (`{classical}`)",
        "Sign convention: positive Δ = quantum better (MUTAG: ΔF1; ZINC: ΔMAE flipped).",
        "",
        f"**k = {r5b['k_wins']} wins out of n = {r5b['n_cells']} cells → p = {_pval_str(r5b['p_value'])}**",
        "",
        "\n".join(rows5b),
        "",
        "---",
        "",
        "## Artefact 5c — Best-of-class per cell sign test",
        "",
        "For each (arch, dataset) cell the best classical and best hybrid config",
        "are selected independently (best mean; ties → lower std).",
        "",
        f"**k = {r5c['k_wins']} wins out of n = {r5c['n_cells']} non-tied cells → p = {_pval_str(r5c['p_value'])}**",
        "",
        "\n".join(rows5c),
        "",
        "---",
        "",
        "## Artefact 6a — Bayesian sign test (cell-level)",
        "",
        f"Input: {r6a.get('n_cells', 'N/A')} normalised cell-level diffs; rope = ±{r6a.get('rope', 0.5)}.",
        "Convention: baycomp P(left) = P(first arg better); first arg = quantum (diffs\\_norm > 0).",
        "",
        "| | value | 95% CI |",
        "|---|---|---|",
        f"| P(quantum clearly better)   | {_fmt(r6a['p_left'])} | {_ci_str(r6a.get('ci_left'))} |",
        f"| P(rope / equiv)             | {_fmt(r6a['p_rope'])} | {_ci_str(r6a.get('ci_rope'))} |",
        f"| P(classical clearly better) | {_fmt(r6a['p_right'])} | {_ci_str(r6a.get('ci_right'))} |",
        "",
        f"Plot: `bayes_sign_test_cell.pdf` + `.png` (saved: {r6a.get('plot_saved', False)})",
        "",
        "---",
        "",
        "## Artefact 6b — Bayesian signed-rank test (MUTAG trial-level)",
        "",
        f"Input: {r6b.get('n_pairs', 'N/A')} paired F1 observations ({per_arch_n}).",
        f"Rope = ±{r6b.get('rope', 0.005)} F1 pts. Median within-cell std = {_fmt(median_std)}.",
        "Convention: baycomp P(left) = P(first arg better); first arg = quantum F1.",
        "",
        "| | value | 95% CI |",
        "|---|---|---|",
        f"| P(quantum better)   | {_fmt(r6b['p_left'])} | {_ci_str(r6b.get('ci_left'))} |",
        f"| P(rope / equiv)     | {_fmt(r6b['p_rope'])} | {_ci_str(r6b.get('ci_rope'))} |",
        f"| P(classical better) | {_fmt(r6b['p_right'])} | {_ci_str(r6b.get('ci_right'))} |",
        "",
        f"Plot: `bayes_signedrank_mutag.pdf` + `.png` (saved: {r6b.get('plot_saved', False)})",
        "",
        "---",
        "",
        sec_6c,
    ]
    out.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Extended sensitivity analysis + HDI
# ─────────────────────────────────────────────────────────────────────────────

ROPES_CELL: list[float] = [0.2, 0.5, 0.8]  # Cohen's d units — Analysis 1
ROPES_TRIAL: list[float] = [0.005, 0.01, 0.02]  # F1 units — Analyses 2 & 3


def _hdi(samples: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    """95% interval. The original tried arviz.hdi first; arviz was not installed in the
    environment that produced the paper numbers, so the equal-tailed quantile interval
    below is what was reported."""
    lo = (1.0 - prob) / 2.0
    return float(np.quantile(samples, lo)), float(np.quantile(samples, 1.0 - lo))


def _t_posterior(d: np.ndarray, seed: int, nsamples: int = 100_000) -> dict[str, float]:
    """Flat-prior Normal model -> t(n-1) posterior on mean(d)."""
    n = len(d)
    rng = np.random.default_rng(seed)
    t_samp = np.asarray(
        scipy_stats.t.rvs(
            df=n - 1,
            loc=float(d.mean()),
            scale=float(d.std(ddof=1) / np.sqrt(n)),
            size=nsamples,
            random_state=int(rng.integers(0, 2**31)),
        )
    )
    lo, hi = _hdi(t_samp)
    return {"median": float(np.median(t_samp)), "hdi_lo": lo, "hdi_hi": hi}


def sensitivity(test_name: str, x: np.ndarray, y: np.ndarray, ropes: list[float]) -> list[dict]:
    """Run a baycomp test at each ROPE value."""
    import baycomp

    test = getattr(baycomp, test_name)
    out = []
    for rope in ropes:
        p_left, p_rope, p_right = test.probs(x, y, rope=rope, nsamples=50000)
        out.append(
            {
                "rope": rope,
                "p_hybrid": float(p_left),
                "p_rope": float(p_rope),
                "p_classical": float(p_right),
            }
        )
    return out


def hdi_hierarchical_params(
    x_arches: np.ndarray, y_arches: np.ndarray, pickle_path: str = "last-sample.pickle"
) -> dict[str, Any]:
    """Unscale the cached Stan posterior (mu, stdh) to F1 units and compute intervals.

    Stan works in units of std_diff = mean(std(y - x, axis=1)); mu_f1 > 0 means quantum
    better. NOTE (as in the original): the cache holds the *last* hierarchical run, i.e.
    the widest ROPE of the sensitivity sweep.
    """
    with open(pickle_path, "rb") as fh:
        _key, (mu_s, stdh_s, _nu_s) = pickle.load(fh)
    diff_mat = np.array(y_arches) - np.array(x_arches)  # classical - quantum
    std_diff = float(np.mean(np.std(diff_mat, axis=1, ddof=1)))
    mu_f1_s = -mu_s * std_diff
    tau_f1_s = stdh_s * std_diff
    mu_hdi, tau_hdi = _hdi(mu_f1_s), _hdi(tau_f1_s)
    per_arch = {
        arch: _t_posterior(-diff_mat[i], seed=i + 42) for i, arch in enumerate(ARCHITECTURES)
    }
    return {
        "mu_median": float(np.median(mu_f1_s)),
        "mu_hdi_lo": mu_hdi[0],
        "mu_hdi_hi": mu_hdi[1],
        "tau_median": float(np.median(tau_f1_s)),
        "tau_hdi_lo": tau_hdi[0],
        "tau_hdi_hi": tau_hdi[1],
        "per_arch": per_arch,
        "std_diff": std_diff,
    }


def plot_forest_arch(
    per_arch: dict[str, dict[str, float]],
    mu_median: float,
    mu_hdi_lo: float,
    mu_hdi_hi: float,
    stem: str = "bayes_hierarchical_forest",
) -> bool:
    """Forest plot: per-arch intervals + global mu band."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, arch in enumerate(ARCHITECTURES):
        info = per_arch[arch]
        ax.errorbar(
            info["median"],
            i,
            xerr=[[info["median"] - info["hdi_lo"]], [info["hdi_hi"] - info["median"]]],
            fmt="o",
            color="#333333",
            ecolor="#333333",
            elinewidth=1.2,
            capsize=4,
            capthick=1.0,
            zorder=3,
        )
    ax.axvspan(mu_hdi_lo, mu_hdi_hi, alpha=0.15, color="#4477AA", zorder=1)
    ax.axvline(
        mu_median,
        color="#4477AA",
        linewidth=1.5,
        linestyle="--",
        zorder=2,
        label=f"Global μ = {mu_median:.4f}\n95% HDI [{mu_hdi_lo:.4f}, {mu_hdi_hi:.4f}]",
    )
    ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_yticks(np.arange(len(ARCHITECTURES)))
    ax.set_yticklabels(ARCHITECTURES)
    ax.set_xlabel("F1 difference (quantum − classical)")
    ax.set_title("Per-architecture effect (hierarchical posterior, MUTAG)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fancybox=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return True


def _rope_rows(results: list[dict], marked: float) -> str:
    rows = ["| ROPE | P(hybrid) | P(rope) | P(classical) |", "|---|---|---|---|"]
    for r in results:
        mark = " ←" if r["rope"] == marked else ""
        rows.append(
            f"| ±{r['rope']} | {r['p_hybrid']:.4f} | {r['p_rope']:.4f} | {r['p_classical']:.4f} |{mark}"
        )
    return "\n".join(rows)


def write_extended_report(
    diffs_norm: list[float],
    x_trial: np.ndarray,
    y_trial: np.ndarray,
    x_arches: np.ndarray,
    y_arches: np.ndarray,
    out: Path = Path("stats_report.md"),
) -> dict[str, Any]:
    """Sensitivity over ROPE + intervals, appended to the report. Returns the numbers."""
    res: dict[str, Any] = {}
    sections = ["", "---", "", "## Extended sensitivity analysis"]

    print("  Analysis 1: sign test sensitivity…")
    x = np.array(diffs_norm)
    res["sens1"] = sensitivity("SignTest", x, np.zeros(len(x)), ROPES_CELL)
    res["hdi1"] = _t_posterior(x, seed=42)
    sections += [
        "",
        "### Analysis 1 — Cell-level sign test (sensitivity over ROPE, Cohen's d)",
        "",
        _rope_rows(res["sens1"], 0.5),
        "",
        (
            f"**HDI 95% on mean Cohen's d:** median = {res['hdi1']['median']:.4f},"
            f" [{res['hdi1']['hdi_lo']:.4f}, {res['hdi1']['hdi_hi']:.4f}]"
        ),
    ]

    if x_trial.size > 0:
        print("  Analysis 2: signed-rank sensitivity…")
        res["sens2"] = sensitivity("SignedRankTest", x_trial, y_trial, ROPES_TRIAL)
        res["hdi2"] = _t_posterior(x_trial - y_trial, seed=42)
        sections += [
            "",
            "### Analysis 2 — Trial-level signed-rank (sensitivity over ROPE, F1)",
            "",
            _rope_rows(res["sens2"], 0.01),
            "",
            (
                f"**HDI 95% on mean diff (quantum − classical):** median = {res['hdi2']['median']:.4f} F1,"
                f" [{res['hdi2']['hdi_lo']:.4f}, {res['hdi2']['hdi_hi']:.4f}] F1"
            ),
        ]

    if x_arches.size > 0:
        print("  Analysis 3: hierarchical sensitivity…")
        res["sens3"] = sensitivity("HierarchicalTest", x_arches, y_arches, ROPES_TRIAL)
        h = res["hdi3"] = hdi_hierarchical_params(x_arches, y_arches)
        forest = plot_forest_arch(h["per_arch"], h["mu_median"], h["mu_hdi_lo"], h["mu_hdi_hi"])
        sections += [
            "",
            "### Analysis 3 — Hierarchical test (sensitivity over ROPE, F1)",
            "",
            _rope_rows(res["sens3"], 0.01),
            "",
            (
                f"**HDI 95% on global μ (quantum − classical):** median = {h['mu_median']:.4f} F1,"
                f" [{h['mu_hdi_lo']:.4f}, {h['mu_hdi_hi']:.4f}] F1"
            ),
            (
                f"**HDI 95% on τ (between-arch std):** median = {h['tau_median']:.4f} F1,"
                f" [{h['tau_hdi_lo']:.4f}, {h['tau_hdi_hi']:.4f}] F1"
            ),
            "",
            "**Per-architecture 95% interval (independent t-posterior):**",
            "",
            "| Architecture | median F1 diff | lo | hi |",
            "|---|---|---|---|",
            *(
                f"| {a} | {i['median']:.4f} | {i['hdi_lo']:.4f} | {i['hdi_hi']:.4f} |"
                for a, i in h["per_arch"].items()
            ),
            "",
            f"Forest plot: `bayes_hierarchical_forest.pdf` + `.png` (saved: {forest})",
        ]

    with open(out, "a" if out.exists() else "w", encoding="utf-8") as fh:
        fh.write("\n".join(sections) + "\n")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 11.  CLI and main
# ─────────────────────────────────────────────────────────────────────────────


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {
            str(k): _jsonable(v) for k, v in o.items() if not str(k).endswith(("_arr", "_arches"))
        }
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run(
    trials: TrialTable,
    output_dir: Path,
    classical: str | None = None,
    quantum: str | None = None,
    hierarchical: bool = True,
) -> dict[str, Any]:
    """Full analysis; writes outputs to ``output_dir`` and returns a summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)  # baycomp writes its Stan cache (last-sample.pickle) in the cwd

    cell = build_all_trials(trials)
    df = build_cell_aggregates(cell)
    df.to_csv("cell_aggregates.csv", index=False)
    with open("cell_trials.pkl", "wb") as f:
        pickle.dump(cell, f)

    if classical or quantum:
        classical = classical or HARDCODED_CLASSICAL_CHAMPION
        quantum = quantum or HARDCODED_QUANTUM_CHAMPION
    else:
        classical, quantum = assert_champions(df)

    r5b = sign_test_5b(df, classical, quantum)
    r5c = sign_test_5c(df)
    print(f"  5b: k={r5b['k_wins']}/{r5b['n_cells']} p={_pval_str(r5b['p_value'])}")
    r6a = bayes_sign_test_6a(df, classical, quantum, rope=0.5)
    r6b = bayes_signed_rank_6b(trials, classical, quantum, rope=0.01)
    r6c = (
        bayes_hierarchical_6c(trials, classical, quantum, rope=0.01)
        if hierarchical
        else {"skipped": True, "reason": "disabled (--no-hierarchical)"}
    )
    write_report(classical, quantum, r5b, r5c, r6a, r6b, r6c)
    ext = write_extended_report(
        r6a["diffs_norm"],
        np.array(r6b["x_arr"]),
        np.array(r6b["y_arr"]),
        np.array(r6c.get("x_arches", [])),
        np.array(r6c.get("y_arches", [])),
    )
    summary = _jsonable(
        {
            "classical": classical,
            "quantum": quantum,
            "5b": r5b,
            "5c": r5c,
            "6a": r6a,
            "6b": r6b,
            "6c": r6c,
            "extended": ext,
        }
    )
    Path("stats_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired statistical tests (paper App. G)")
    p.add_argument(
        "--trials",
        type=Path,
        nargs="*",
        default=[],
        help="trials CSV files (arch,dataset,config,fold,seed,value)",
    )
    p.add_argument(
        "--gt-mutag-root",
        type=Path,
        default=None,
        help="GT MUTAG result tree (overrides GT MUTAG rows of --trials)",
    )
    p.add_argument(
        "--gt-zinc-root",
        type=Path,
        default=None,
        help="GT ZINC result tree (overrides GT ZINC rows of --trials)",
    )
    p.add_argument(
        "--gnn-mutag-root",
        type=Path,
        default=None,
        help="MUTAG result tree of the other backbones (replaces the cells it contains)",
    )
    p.add_argument("--classical", default=None, help="override classical champion")
    p.add_argument("--quantum", default=None, help="override quantum champion")
    p.add_argument("--no-hierarchical", action="store_true", help="skip the Stan-based test")
    p.add_argument("--output-dir", type=Path, default=Path("results/stats"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trials: TrialTable = {}
    for path in args.trials:
        trials.update(read_trials_csv(path.resolve()))
    for root, ds in ((args.gt_mutag_root, "MUTAG"), (args.gt_zinc_root, "ZINC_small")):
        if root is not None:
            trials = {k: v for k, v in trials.items() if not (k[0] == "GT" and k[1] == ds)}
            trials.update(read_gt_results_tree(root.resolve(), ds))
    if args.gnn_mutag_root is not None:
        rerun = read_gnn_results_tree(args.gnn_mutag_root.resolve())
        print(f"cells replaced by the re-run tree: {sorted(k[0] + ':' + k[2] for k in rerun)}")
        trials.update(rerun)
    run(
        trials,
        args.output_dir.resolve(),
        args.classical,
        args.quantum,
        hierarchical=not args.no_hierarchical,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
