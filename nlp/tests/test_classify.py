"""Tests for the Task 4 classifier evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_nlp import classify as cl


def test_corrected_ttest_matches_formula_and_is_wider_than_naive():
    rng = np.random.default_rng(0)
    diff = rng.normal(0.1, 0.05, 50)
    mean, lo, hi, p = cl.corrected_ttest(diff, n_train=120, n_test=30)
    se = np.sqrt(np.var(diff, ddof=1) * (1 / 50 + 30 / 120))
    assert mean == pytest.approx(diff.mean())
    assert hi - mean == pytest.approx(2.0096 * se, rel=1e-3)          # t(49, .975)
    naive = 2.0096 * np.std(diff, ddof=1) / np.sqrt(50)
    assert hi - mean > naive                                            # correction widens the interval
    assert 0 <= p < 0.05


def test_corrected_ttest_no_difference():
    mean, lo, hi, p = cl.corrected_ttest(np.array([0.1, -0.1, 0.05, -0.05, 0.0]), 80, 20)
    assert mean == pytest.approx(0.0) and lo < 0 < hi and p > 0.5


def _toy(n_per=20, seed=0):
    rng = np.random.default_rng(seed)
    pos = [f"growth success doubled record boost {rng.integers(1e6)}" for _ in range(n_per)]
    neu = [f"tender meeting bids invited committee {rng.integers(1e6)}" for _ in range(n_per)]
    return pd.Series(pos + neu), pd.Series(["positive"] * n_per + ["neutral"] * n_per)


def test_cross_validate_beats_baseline_on_separable_toy_data():
    X, y = _toy()
    folds, oof = cl.cross_validate(X, y, ("neutral", "positive"), n_splits=5, n_repeats=2, seed=1)
    assert set(folds["model"]) == {"tfidf_logreg", "majority_class"}
    assert len(folds) == 2 * 5 * 2
    assert folds.loc[folds["model"] == "tfidf_logreg", "accuracy"].mean() == 1.0
    assert folds.loc[folds["model"] == "majority_class", "accuracy"].mean() == pytest.approx(0.5)
    assert oof.shape == (40, 2) and oof.notna().all().all()             # every article predicted once per repeat
    s = cl.summarise(folds).set_index(["metric", "model"])
    assert s.loc[("macro_f1", "difference (model - baseline)"), "mean"] > 0.5
    pc = cl.pooled_per_class(y, oof, ("neutral", "positive"))
    assert (pc["f1"] == 1.0).all() and pc["support"].tolist() == [20, 20]


def test_cross_validate_is_reproducible():
    X, y = _toy()
    a, _ = cl.cross_validate(X, y, ("neutral", "positive"), n_splits=5, n_repeats=1, seed=3)
    b, _ = cl.cross_validate(X, y, ("neutral", "positive"), n_splits=5, n_repeats=1, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_top_terms_binary_orientation():
    X, y = _toy()
    t = cl.top_terms(X, y, n=3).set_index("label")["top_terms"]
    assert any(w in t["positive"] for w in ("growth", "success", "doubled", "record", "boost"))
    assert any(w in t["neutral"] for w in ("tender", "meeting", "bids", "invited", "committee"))
