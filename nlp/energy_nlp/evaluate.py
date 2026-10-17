"""Evaluation of model predictions against hand labels (Tasks 3 and 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def scores(y_true, y_pred, labels) -> dict[str, float]:
    """Accuracy, macro-F1 (over ``labels``, absent classes score 0) and Cohen's kappa."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=list(labels))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0),
        "kappa": 0.0 if np.isnan(kappa) else kappa,
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
    """Point estimate and percentile bootstrap interval for each of ``scores``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    point = scores(y_true, y_pred, labels)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        for k, v in scores(y_true[i], y_pred[i], labels).items():
            boots[k].append(v)
    a = (1 - level) / 2
    return pd.DataFrame([{"metric": k, "value": point[k], "lo": np.quantile(v, a), "hi": np.quantile(v, 1 - a)}
                         for k, v in boots.items()])


def weighted_accuracy(y_true, y_pred, weights) -> float:
    """Accuracy re-weighted to the corpus with sampling weights (Horvitz-Thompson style)."""
    w = np.asarray(weights, dtype=float)
    return float((w * (np.asarray(y_true) == np.asarray(y_pred))).sum() / w.sum())


def weighted_distribution(values, weights, labels) -> pd.Series:
    """Estimated corpus share of each label from a weighted sample."""
    s = pd.Series(np.asarray(weights, dtype=float)).groupby(np.asarray(values)).sum()
    return (s / s.sum()).reindex(list(labels), fill_value=0.0)
