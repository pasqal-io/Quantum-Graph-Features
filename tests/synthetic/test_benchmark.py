import numpy as np
import pandas as pd

from qgf.synthetic.benchmark import (
    MAIN_TABLE,
    METRICS,
    cross_validate,
    evaluate,
    fold_splits,
    markdown_table,
)


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    X = rng.normal(size=(n, 3))
    X[:, 0] += 8 * y  # separable feature
    folds = np.stack([rng.permutation(np.arange(n) % 5) for _ in range(3)], 1)
    return {"signal": X, "noise": rng.normal(size=(n, 3))}, y, folds


def test_fold_splits_partition_the_graphs():
    _, _, folds = _toy()
    for rep in range(3):
        splits = fold_splits(folds, rep)
        assert len(splits) == 5
        tests = np.sort(np.concatenate([te for _, te in splits]))
        assert np.array_equal(tests, np.arange(len(folds)))
        for tr, te in splits:
            assert not set(tr) & set(te)


def test_evaluate_signal_vs_noise():
    F, y, folds = _toy()
    trials = cross_validate(F["signal"], y, folds)
    assert len(trials) == 15 and trials["roc_auc"].min() > 0.95
    summary, per_fold = evaluate(F, y, folds, ["signal", "noise"], n_perm=9)
    s = summary.set_index("family")
    assert {f"{m}_{x}" for m in METRICS for x in ("mean", "std")} <= set(s.columns)
    assert s.loc["signal", "roc_auc_mean"] > 0.95 > s.loc["noise", "roc_auc_mean"]
    assert s.loc["signal", "perm_p"] == 0.1  # floor 1 / (9 + 1)
    assert len(per_fold) == 30


def test_markdown_table_layout():
    summary = pd.DataFrame(
        [
            {
                "family": "afm",
                **{f"{m}_mean": 0.9 for m in METRICS},
                **{f"{m}_std": 0.01 for m in METRICS},
            }
        ]
    )
    lines = markdown_table(summary, MAIN_TABLE).splitlines()
    assert lines[0] == "| Feature set | Accuracy | F1 | ROC-AUC |"
    assert lines[2] == "| AFM | 0.900 ± 0.010 | 0.900 ± 0.010 | 0.900 ± 0.010 |"
    assert len(lines) == 3  # families absent from the summary are skipped
