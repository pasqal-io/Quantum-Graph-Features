"""INTERNAL ONLY: reference numbers of the original temporal-PCA script (App. D.2).

Internal use only: requires Pasqal-internal packages/data; not needed to reproduce results
from the released data.

Runs ``run_pulse`` of graph-transformer's ``analyze_pca_{mutag,zinc}.py`` (read-only,
from its own venv) and dumps variance ratios to JSON; no figure is written.

    <graph-transformer>/.venv/bin/python scripts/internal/generation_reference_pca.py \
        --gt <graph-transformer> --dataset zinc --out results/temporal_pca/reference_zinc.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--gt", type=Path, required=True)
parser.add_argument("--dataset", choices=["mutag", "zinc"], required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

path = args.gt / f"results/analysis_scripts/analyze_pca_{args.dataset}.py"
spec = importlib.util.spec_from_file_location("ref_pca", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if args.dataset == "mutag":
    from graph_transformer.data.mutag_loader import load_mutag as loader
else:
    from graph_transformer.data.zinc_small_loader import load_zinc_small_all as loader
out = {}
for pulse in ("AFM", "FM"):
    ob, tb = mod.run_pulse(pulse, loader(pulse))
    out[pulse.lower()] = {
        "onebody": np.asarray(ob.var_ratio).tolist(),
        "twobody": np.asarray(tb.var_ratio).tolist(),
    }
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(out, indent=1))
print(f"wrote {args.out}")
