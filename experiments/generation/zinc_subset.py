"""Re-derive the ZINC*_subset (11 368 graphs) from the 11 494 embedded ZINC graphs.

The subset drops the embeddings of poorer quality, clusters C3 and C4 of the
embedding-quality classification on the edge-length uniformity
``ELU = std(d_connected) / mean(d_connected)`` (unbiased std over both directions of
every edge) and the gap ratio ``GR = quantile_1%(d_non_connected) / max(d_connected)``:
C4: ELU > 0.03; C3: ELU > 0.015 and GR < 1.3; C2: GR < 1.5; C1: rest.
C1 and C2 are kept.

    uv run python experiments/generation/zinc_subset.py --raw-dir data/raw

Checks the selected ids against the released ``zinc_afm`` splits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qgf.generate.raw import raw_path

EXCLUDED = {"C3", "C4"}


def elu_gr(pos: np.ndarray, edge_index: np.ndarray) -> tuple[float, float]:
    """Port of the original metrics (torch, float32 positions)."""
    pos = torch.as_tensor(pos)
    n = pos.shape[0]
    dist = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(0), dim=-1)
    adj = torch.zeros(n, n, dtype=torch.bool)
    ei = torch.as_tensor(edge_index)
    adj[ei[0], ei[1]] = True
    adj[ei[1], ei[0]] = True
    non = ~adj & ~torch.eye(n, dtype=torch.bool)
    d_conn, d_non = dist[adj], dist[non]
    elu = d_conn.std().item() / d_conn.mean().item()
    gr = torch.quantile(d_non.float(), 0.01).item() / d_conn.max().item()
    return elu, gr


def cluster(elu: float, gr: float) -> str:
    if elu > 0.03:
        return "C4"
    if elu > 0.015 and gr < 1.3:
        return "C3"
    if gr < 1.5:
        return "C2"
    return "C1"


def load_embeddings(path: Path):
    z = np.load(path)
    pos = np.split(z["pos"], np.cumsum(z["num_nodes"])[:-1])
    edges = np.split(z["edge_index"], np.cumsum(z["num_edges"])[:-1], axis=1)
    return z["initial_id"], pos, edges


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results/generation/zinc_subset.json"))
    parser.add_argument("--no-check", action="store_true", help="skip the released-data check")
    args = parser.parse_args(argv)

    report = {}
    for split in ("train", "val", "test"):
        ids, pos, edges = load_embeddings(raw_path(args.raw_dir, f"zinc_embeddings_{split}"))
        labels = [cluster(*elu_gr(p, e)) for p, e in zip(pos, edges, strict=True)]
        keep = sorted(int(i) for i, c in zip(ids, labels, strict=True) if c not in EXCLUDED)
        rep = {
            "embedded": len(ids),
            "clusters": {c: labels.count(c) for c in ("C1", "C2", "C3", "C4")},
            "kept": len(keep),
            "excluded_ids": sorted(
                int(i) for i, c in zip(ids, labels, strict=True) if c in EXCLUDED
            ),
        }
        if not args.no_check:
            from qgf.data import load

            released = sorted(int(g.initial_id.flatten()[0]) for g in load("zinc_afm", split)[0])
            rep["matches_released"] = keep == released
            rep["symmetric_difference"] = sorted(set(keep) ^ set(released))
        report[split] = rep
        print(split, {k: v for k, v in rep.items() if k != "excluded_ids"}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    total = sum(r["kept"] for r in report.values())
    print(f"kept {total} / {sum(r['embedded'] for r in report.values())}; wrote {args.out}")


if __name__ == "__main__":
    main()
