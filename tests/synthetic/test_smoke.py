"""Smoke tests of the ladder entry points, and the paper table from the released data."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qgf.data import DEFAULT_ROOT
from qgf.data.io import save_dataset
from qgf.synthetic.dataset import ladder_data, make_meta

ROOT = Path(__file__).parents[2]
EXPECTED = ROOT / "experiments" / "ladder" / "expected" / "summary.csv"


def _script(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_datasets(root: Path) -> None:
    rng = np.random.default_rng(0)
    specs = [(4, (c,)) for c in (0, 1, 2)] + [(5, (c,)) for c in (0, 1, 2, 3)]
    specs = specs * 2
    for pulse in ("afm", "fm"):
        graphs = []
        for i, (m, cr) in enumerate(specs):
            n = 2 * m
            occ = rng.uniform(0.05, 0.6, size=(20, n))
            conn = rng.normal(scale=0.05, size=(20, n, n))
            conn = conn + conn.transpose(0, 2, 1)
            graphs.append(
                ladder_data(
                    m,
                    cr,
                    y=i % 2,
                    degeneracy=1 + i % 2,
                    initial_id=i,
                    blind_group=i // 2,
                    cv_fold=[i % 5, (i + 1) % 5, (i + 2) % 5],
                    occupation=occ,
                    connected=conn,
                )
            )
        save_dataset(root / f"ladder_{pulse}.pt", graphs, make_meta(pulse, "test"))


def test_run_smoke(tmp_path):
    _tiny_datasets(tmp_path)
    out = tmp_path / "out"
    _script("experiments/ladder/run.py").main(["--data-root", str(tmp_path), "--out", str(out)])
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 7 and (out / "table.md").read_text().startswith("| Feature set")
    assert len(pd.read_csv(out / "per_fold.csv")) == 7 * 15


def test_generate_folds_are_grouped_and_stratified():
    gen = _script("experiments/ladder/generate.py")
    y = np.tile([0, 1], 20)
    groups = np.repeat(np.arange(20), 2)
    folds = gen.assign_folds(y, groups)
    assert folds.shape == (40, 3) and folds.min() == 0 and folds.max() == 4
    for rep in range(3):
        for g in range(20):  # a blind group never straddles two folds
            assert len(set(folds[groups == g, rep])) == 1


@pytest.mark.slow
@pytest.mark.skipif(
    not (DEFAULT_ROOT / "ladder_afm.pt").exists(), reason="ladder data not downloaded"
)
def test_paper_table_from_released_data(tmp_path):
    """Paper rows from data/processed equal experiments/ladder/expected (~20 s)."""
    _script("experiments/ladder/run.py").main(["--out", str(tmp_path)])
    got = pd.read_csv(tmp_path / "summary.csv").set_index("family")
    ref = pd.read_csv(EXPECTED).set_index("family").loc[got.index]
    cols = [c for c in got.columns if c.endswith(("_mean", "_std"))]
    np.testing.assert_allclose(got[cols].to_numpy(), ref[cols].to_numpy(), atol=1e-12)
