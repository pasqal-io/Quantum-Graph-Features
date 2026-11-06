"""Pinned vs frustrated ladders: logistic regression on pooled features, stored CV folds.

Protocol (as in the paper's ladder table): standardisation + logistic regression
(``max_iter=5000``), fitted inside each training fold; 5 folds grouped by blind group
(``StratifiedGroupKFold(5, shuffle=True, random_state=r)``, r = 0, 1, 2) = 15 test folds.
Accuracy, F1 (pinned = positive class) and ROC-AUC: mean and standard deviation (ddof 0)
over the 15 folds. Optional permutation test: AUC of repeat 0 vs ``n_perm`` label shuffles.

The folds are read from the dataset (``cv_fold``) rather than regenerated: scikit-learn
>= 1.8 shuffles ``StratifiedGroupKFold`` differently from the 1.7 release behind the
reference table, which moves AUCs by up to 0.02.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

METRICS = ("accuracy", "f1", "roc_auc")

# (family, label) of the paper table; RRWP = RRWP-20, the paper encoding
MAIN_TABLE = [
    ("rrwp20", "RRWP"),
    ("geom", "Geometry"),
    ("rrwp20+lap+ring+geom", "RRWP+Lap+Ring+Geom"),
    ("fm", "FM"),
    ("afm", "AFM"),
    ("afm+fm", "AFM+FM"),
    ("rrwp20+lap+ring+geom+afm+fm", "RRWP+Lap+Ring+Geom+AFM+FM"),
]
FULL_TABLE = [
    ("n", "Node count"),
    ("rrwp2", "RRWP-2 (blind horizon)"),
    ("rrwp3", "RRWP-3"),
    ("rrwp20", "RRWP-20"),
    ("geom", "Geometry"),
    ("lap", "Laplacian"),
    ("ring", "Ring"),
    ("rrwp2+lap+ring+geom", "RRWP-2+Lap+Ring+Geom"),
    ("rrwp3+lap+ring+geom", "RRWP-3+Lap+Ring+Geom"),
    ("rrwp20+lap+ring+geom", "RRWP-20+Lap+Ring+Geom"),
    ("fm", "FM"),
    ("afm", "AFM"),
    ("afm+fm", "AFM+FM"),
    ("rrwp2+lap+ring+geom+afm+fm", "RRWP-2+Lap+Ring+Geom+AFM+FM"),
    ("rrwp3+lap+ring+geom+afm+fm", "RRWP-3+Lap+Ring+Geom+AFM+FM"),
    ("rrwp20+lap+ring+geom+afm+fm", "RRWP-20+Lap+Ring+Geom+AFM+FM"),
    ("gs", "GS (ideal ground state)"),
]


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def fold_splits(folds: np.ndarray, repeat: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(train, test)]`` of one repeat, in fold order, from ``folds [G, R]``."""
    col = folds[:, repeat]
    return [(np.nonzero(col != k)[0], np.nonzero(col == k)[0]) for k in np.unique(col)]


def cross_validate(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    """One row per (repeat, fold) with the three test metrics."""
    rows = []
    for rep in range(folds.shape[1]):
        for k, (tr, te) in enumerate(fold_splits(folds, rep)):
            model = logreg().fit(X[tr], y[tr])
            pred = model.predict(X[te])
            rows.append(
                {
                    "repeat": rep,
                    "fold": k,
                    "accuracy": accuracy_score(y[te], pred),
                    "f1": f1_score(y[te], pred, zero_division=0.0),
                    "roc_auc": roc_auc_score(y[te], model.decision_function(X[te])),
                }
            )
    return pd.DataFrame(rows)


def permutation_p(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray, n_perm: int, n_jobs: int = 1
) -> float:
    """Permutation p-value of the repeat-0 ROC-AUC (labels shuffled across all graphs)."""
    _, _, p = permutation_test_score(
        logreg(),
        X,
        y,
        cv=fold_splits(folds, 0),
        n_permutations=n_perm,
        scoring="roc_auc",
        random_state=0,
        n_jobs=n_jobs,
    )
    return float(p)


def evaluate(
    F: dict[str, np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    families,
    n_perm: int = 0,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(summary, per_fold)``: summary has one row per family (``<metric>_mean``,
    ``<metric>_std`` and, with ``n_perm > 0``, ``perm_p``)."""
    summary, per_fold = [], []
    for fam in families:
        X = F[fam]
        trials = cross_validate(X, y, folds).assign(family=fam)
        per_fold.append(trials)
        row = {"family": fam, "dim": X.shape[1]}
        for metric in METRICS:
            row[f"{metric}_mean"] = trials[metric].mean()
            row[f"{metric}_std"] = trials[metric].std(ddof=0)
        if n_perm > 0:
            row["perm_p"] = permutation_p(X, y, folds, n_perm, n_jobs)
        summary.append(row)
    cols = ["family", "repeat", "fold", *METRICS]
    return pd.DataFrame(summary), pd.concat(per_fold, ignore_index=True)[cols]


def cell(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def markdown_table(summary: pd.DataFrame, labels) -> str:
    """Rows of ``labels`` = ``[(family, label)]`` as ``Feature set | Accuracy | F1 | ROC-AUC``
    (+ ``perm p`` when present)."""
    by_family = summary.set_index("family")
    perm = "perm_p" in by_family.columns
    head = ["Feature set", "Accuracy", "F1", "ROC-AUC"] + (["perm p"] if perm else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for fam, label in labels:
        if fam not in by_family.index:
            continue
        r = by_family.loc[fam]
        cells = [label] + [cell(r[f"{m}_mean"], r[f"{m}_std"]) for m in METRICS]
        if perm:
            cells.append(f"{r['perm_p']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
