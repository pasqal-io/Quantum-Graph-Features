"""Aggregate the placement-sensitivity cases: Spearman similarity to the reference layout.

For each case (noise level or alternative layout) the upper-triangle off-diagonal
correlations of all 20 times are pooled and compared with those of the reference
layout of the same graph by Spearman rank correlation; reported as mean +- std over
the 4 graphs x 10 runs. Also the per-timestep table of the original study
(early/late thirds) and a node-relabelling floor.

    uv run python experiments/layout_sensitivity/analyze.py data/raw/layout_sensitivity
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HARDWARE_JITTER_UM = (0.10, 0.20)  # trap placement jitter; sigma/d_min ~ 0.02-0.04
MODES = {"reembed": "re-embedding", "post_selected_embedding": "re-embedding (post-selected)"}


def load_cases(run_dir: Path, pulse: str = "afm") -> list[dict]:
    cases = []
    for path in sorted((run_dir / pulse).glob("*.npz")):
        with np.load(path) as z:
            cases.append({k: z[k] if z[k].ndim else z[k].item() for k in z.files})
    if not cases:
        raise FileNotFoundError(f"no .npz in {run_dir / pulse}")
    return cases


def pooled_spearman(ref: np.ndarray, other: np.ndarray) -> float:
    iu = np.triu_indices(ref.shape[1], k=1)
    return float(spearmanr(ref[:, iu[0], iu[1]].ravel(), other[:, iu[0], iu[1]].ravel()).statistic)


def per_timestep_spearman(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(ref.shape[1], k=1)
    out = np.full(ref.shape[0], np.nan)
    for t in range(ref.shape[0]):
        a, b = ref[t][iu], other[t][iu]
        if a.std() > 0 and b.std() > 0:
            out[t] = spearmanr(a, b).statistic
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/layout_sensitivity/summary.json"))
    args = parser.parse_args(argv)

    cases = load_cases(args.run_dir)
    refs = {
        c["graph_idx"]: c["corr"] for c in cases if c["mode"] == "noise" and c["sigma_rel"] == 0
    }
    pooled, curves = defaultdict(list), defaultdict(list)
    for c in cases:
        if c["mode"] == "noise" and c["sigma_rel"] == 0:
            continue
        key = (
            f"sigma={c['sigma_rel']:g}" if c["mode"] == "noise" else MODES.get(c["mode"], c["mode"])
        )
        pooled[key].append(pooled_spearman(refs[c["graph_idx"]], c["corr"]))
        curves[key].append(per_timestep_spearman(refs[c["graph_idx"]], c["corr"]))
    rng = np.random.default_rng(0)
    floor = []
    for corr in refs.values():
        perm = rng.permutation(corr.shape[1])
        floor.append(pooled_spearman(corr, corr[:, perm][:, :, perm]))

    summary = {"graphs": sorted(int(g) for g in refs), "pooled_spearman": {}, "per_timestep": {}}
    print(f"{'case':32s} pooled Spearman   early third   late third   (n)")
    for key in sorted(pooled, key=lambda k: (not k.startswith("sigma"), k)):
        v, cur = np.array(pooled[key]), np.vstack(curves[key])
        third = cur.shape[1] // 3
        early, late = np.nanmean(cur[:, :third], 1), np.nanmean(cur[:, -third:], 1)
        summary["pooled_spearman"][key] = {"mean": v.mean(), "std": v.std(), "n": len(v)}
        summary["per_timestep"][key] = np.nanmean(cur, 0).tolist()
        print(
            f"{key:32s} {v.mean():.3f} +- {v.std():.3f}   {early.mean():.3f}+-{early.std():.3f}"
            f"   {late.mean():.3f}+-{late.std():.3f}   ({len(v)})"
        )
    summary["permutation_floor"] = float(np.mean(floor))
    lo, hi = HARDWARE_JITTER_UM
    summary["hardware_sigma_over_dmin"] = [lo / 5.0, hi / 5.0]
    print(
        f"node-permutation floor {np.mean(floor):.3f}; hardware noise ~ sigma/d_min "
        f"{lo / 5:.2f}-{hi / 5:.2f}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
