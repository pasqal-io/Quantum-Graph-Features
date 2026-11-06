"""Emulator-QPU concordance at t4 on MUTAG AFM (App. D.4, Fig. 5).

QPU run: rescaled, time-corrected, padded AFM quench (t_emu = 715 ns, t_QPU = 940 ns),
per-graph CLT shot budget. Emulation: 1000 shots at 715 ns. Both are estimated with the
readout-corrected jackknife (eps, eps') = (0.025, 0.1). For each graph, Spearman rank
correlation between QPU and emulation entries: diagonal (one-body) and upper triangle
(two-body); then pooled over all (graph, entry) pairs.

    uv run python experiments/qpu_vs_emulation/run.py --raw-dir data/raw

The diagonal uses the single-atom readout correction (see ``qgf.generate.correlations``).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from qgf.generate.correlations import jackknife_correlation_from_arrays
from qgf.generate.qpu import qpu_pulse_duration
from qgf.generate.raw import QPU_RUNS, load_raw, raw_path

T_EMU_NS = 715
QPU_RUN = QPU_RUNS[2]  # T4
EXAMPLE_GRAPHS = [0, 50, 100, 150]


def concordance(emu, qpu) -> dict:
    t_emu = int(np.flatnonzero(emu.times_ns == T_EMU_NS)[0])
    sp_ob, sp_tb, pairs, mats = [], [], {"ob": ([], []), "tb": ([], [])}, {}
    for g, gid in enumerate(qpu.initial_id):
        e = emu.index_of(int(gid))
        ec, _ = jackknife_correlation_from_arrays(*emu.bit_matrix(e, t_emu))
        qc, _ = jackknife_correlation_from_arrays(*qpu.bit_matrix(g, 0))
        mats[int(gid)] = (ec, qc)
    max_abs = max(np.abs(m).max() for pair in mats.values() for m in pair)
    for ec, qc in mats.values():
        # Rounding restores exact ties (atoms with identical statistics) that the vectorised
        # estimator can break at the 1e-16 level; ranks would otherwise flip arbitrarily.
        ec, qc = np.round(ec / max_abs, 12), np.round(qc / max_abs, 12)
        n = ec.shape[0]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairs["ob"][0].append(np.diag(ec))
        pairs["ob"][1].append(np.diag(qc))
        pairs["tb"][0].append(ec[mask])
        pairs["tb"][1].append(qc[mask])
        sp_ob.append(spearmanr(np.diag(ec), np.diag(qc)).statistic if n > 2 else np.nan)
        sp_tb.append(spearmanr(ec[mask], qc[mask]).statistic if mask.sum() > 2 else np.nan)
    out = {"max_abs": float(max_abs)}
    for key, short, sp in (("onebody", "ob", sp_ob), ("twobody", "tb", sp_tb)):
        sp = np.array(sp)
        e, q = (np.concatenate(v) for v in pairs[short])
        out[key] = {
            "median": float(np.nanmedian(sp)),
            "mean": float(np.nanmean(sp)),
            "pooled": float(spearmanr(e, q).statistic),
            "per_graph": sp.tolist(),
        }
    out["_mats"], out["_pairs"] = mats, pairs
    return out


def check_time_correction(emu_pos_min: np.ndarray, qpu) -> dict:
    """Scale = r_min(QPU register) / r_min(emulation register); t_QPU = floor4(715 scale^6)."""
    regs = qpu.meta["registers"]
    scales = []
    for r, rmin in zip(regs, emu_pos_min, strict=True):
        p = np.array(r)
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        scales.append(d[d > 0].min() / rmin)
    raw_t = [T_EMU_NS * s**6 for s in scales]
    # 715 s^6 sits at ~943.99: float32 positions put a few graphs on either side of 944.
    return {
        "scale_min": min(scales),
        "scale_max": max(scales),
        "t_emu_scale6_range": [float(min(raw_t)), float(max(raw_t))],
        "t_qpu_ns": sorted({qpu_pulse_duration(T_EMU_NS, s) for s in scales}),
        "t_qpu_stored_ns": int(qpu.times_ns[0]),
    }


def plot(res: dict, out: Path) -> None:
    mats, pairs = res["_mats"], res["_pairs"]
    m = res["max_abs"]
    fig = plt.figure(figsize=(7.0, 7.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 0.8, 0.8], hspace=0.35)
    top = gs[0].subgridspec(2, len(EXAMPLE_GRAPHS), hspace=0.05, wspace=0.05)
    vmax = max(np.abs(mats[i][k] / m).max() for i in EXAMPLE_GRAPHS for k in (0, 1))
    for c, gid in enumerate(EXAMPLE_GRAPHS):
        for r, label in enumerate(("Emulation", "QPU")):
            ax = fig.add_subplot(top[r, c])
            ax.imshow(mats[gid][r] / m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"#{gid}  N={mats[gid][0].shape[0]}", fontsize=7)
            if c == 0:
                ax.set_ylabel(label, fontsize=7)
    mid = gs[1].subgridspec(1, 2, wspace=0.25)
    bot = gs[2].subgridspec(1, 2, wspace=0.25)
    for j, key in enumerate(("onebody", "twobody")):
        sp = np.array(res[key]["per_graph"])
        ax = fig.add_subplot(mid[j])
        ax.hist(sp[~np.isnan(sp)], bins=30, color="C0" if j == 0 else "C3")
        ax.axvline(
            res[key]["median"], color="k", ls="--", label=f"median = {res[key]['median']:.3f}"
        )
        ax.axvline(res[key]["mean"], color="0.5", ls=":", label=f"mean = {res[key]['mean']:.3f}")
        ax.set_xlabel(r"Spearman $\rho_s$", fontsize=7)
        ax.legend(fontsize=6)
        e, q = (np.concatenate(v) for v in pairs["ob" if j == 0 else "tb"])
        ax = fig.add_subplot(bot[j])
        ax.scatter(e, q, s=1, alpha=0.35, color="C0" if j == 0 else "C3", rasterized=True)
        lim = [min(e.min(), q.min()), max(e.max(), q.max())]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_title(f"{key} ($\\rho_s$ = {res[key]['pooled']:.3f})", fontsize=7)
        ax.set_xlabel("Emulation $C$", fontsize=7)
        ax.set_ylabel("QPU $C$", fontsize=7)
    fig.suptitle(r"MUTAG AFM, QPU vs emulation ($t_{emu}$=715 ns, $t_{QPU}$=940 ns)", fontsize=8)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results/qpu_vs_emulation"))
    args = parser.parse_args(argv)

    from qgf.data import load

    emu = load_raw(raw_path(args.raw_dir, "mutag_afm"))
    qpu = load_raw(raw_path(args.raw_dir, QPU_RUN))
    res = concordance(emu, qpu)
    graphs = {int(g.initial_id.flatten()[0]): g for g in load("mutag_afm")[0]}
    rmin = []
    for gid in qpu.initial_id:
        p = graphs[int(gid)].pos.double().numpy()
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        rmin.append(d[d > 0].min())
    summary = {
        "shots_qpu": [int(qpu.shot_counts[qpu._slice(g, 0)].sum()) for g in range(len(qpu))],
        "time_correction": check_time_correction(np.array(rmin), qpu),
        "onebody": res["onebody"],
        "twobody": res["twobody"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fig5.json").write_text(json.dumps(summary, indent=1))
    plot(res, args.out / "fig5.png")
    for k in ("onebody", "twobody"):
        print(
            f"{k}: median {res[k]['median']:.3f}  mean {res[k]['mean']:.3f}  "
            f"pooled {res[k]['pooled']:.3f}"
        )
    tc = summary["time_correction"]
    print(
        f"scale {tc['scale_min']:.5f}-{tc['scale_max']:.5f}, 715 s^6 in "
        f"{tc['t_emu_scale6_range']}, stored t_QPU {tc['t_qpu_stored_ns']} ns"
    )
    print(f"wrote {args.out}/fig5.json/.png")


if __name__ == "__main__":
    main()
