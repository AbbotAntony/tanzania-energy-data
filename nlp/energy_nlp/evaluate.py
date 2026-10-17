"""Evaluation of model predictions against hand labels (Tasks 3 and 4)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def kappa(y_true, y_pred, labels) -> float:
    """Cohen's kappa; NaN when undefined (both raters use one and the same single label)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = cohen_kappa_score(y_true, y_pred, labels=list(labels))
    return float(k)


def scores(y_true, y_pred, labels) -> dict[str, float]:
    """Accuracy, macro-F1 (over ``labels``, absent classes score 0) and Cohen's kappa."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0),
        "kappa": kappa(y_true, y_pred, labels),
    }


def per_class(y_true, y_pred, labels) -> pd.DataFrame:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return pd.DataFrame({"label": list(labels), "precision": p, "recall": r, "f1": f, "support": n})


def confusion(y_true, y_pred, labels) -> pd.DataFrame:
    """Rows = hand label, columns = prediction."""
    m = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(m, index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="pred"))


def bootstrap_scores(y_true, y_pred, labels, n_boot: int = 2000, seed: int = 42,
                     level: float = 0.95) -> pd.DataFrame:
    """Point estimate and percentile bootstrap interval for each of ``scores``.

    Resamples where kappa is undefined are skipped for kappa (not counted as 0);
    ``n_valid`` reports how many resamples each interval uses."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    point = scores(y_true, y_pred, labels)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        for k, v in scores(y_true[i], y_pred[i], labels).items():
            boots[k].append(v)
    a = (1 - level) / 2
    rows = []
    for k, v in boots.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        lo, hi = (np.quantile(v, [a, 1 - a]) if len(v) else (np.nan, np.nan))
        rows.append({"metric": k, "value": point[k], "lo": lo, "hi": hi, "n_valid": len(v)})
    return pd.DataFrame(rows)


def weighted_accuracy(y_true, y_pred, weights) -> float:
    """Accuracy re-weighted to the corpus with sampling weights (Horvitz-Thompson style)."""
    w = np.asarray(weights, dtype=float)
    return float((w * (np.asarray(y_true) == np.asarray(y_pred))).sum() / w.sum())


def weighted_distribution(values, weights, labels) -> pd.Series:
    """Estimated corpus share of each label from a weighted sample."""
    s = pd.Series(np.asarray(weights, dtype=float)).groupby(np.asarray(values)).sum()
    return (s / s.sum()).reindex(list(labels), fill_value=0.0)
