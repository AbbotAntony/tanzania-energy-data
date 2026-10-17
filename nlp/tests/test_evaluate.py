"""Tests for evaluation metrics and the sentiment helpers that need no model download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_nlp import evaluate as ev
from energy_nlp import sentiment as se

L = ("negative", "neutral", "positive")


def test_scores_perfect_and_hand_computed():
    y = ["positive", "neutral", "negative", "positive"]
    assert ev.scores(y, y, L) == {"accuracy": 1.0, "macro_f1": 1.0, "kappa": 1.0}
    # 3 of 4 right; positive: P=1, R=.5, F=2/3; neutral: P=.5, R=1, F=2/3; negative: F=1
    p = ["positive", "neutral", "negative", "neutral"]
    s = ev.scores(y, p, L)
    assert s["accuracy"] == 0.75
    assert s["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_macro_f1_counts_absent_predicted_class_as_zero():
    y = ["positive", "positive", "negative"]
    p = ["positive", "positive", "positive"]
    s = ev.scores(y, p, L)
    # positive F = 0.8, negative F = 0, neutral (absent everywhere) F = 0
    assert s["macro_f1"] == pytest.approx(0.8 / 3)


def test_kappa_matches_definition():
    y = ["positive"] * 5 + ["negative"] * 5
    p = ["positive"] * 4 + ["negative"] + ["negative"] * 4 + ["positive"]
    # po = .8, pe = .5 -> kappa = .6
    assert ev.scores(y, p, L)["kappa"] == pytest.approx(0.6)


def test_confusion_orientation():
    m = ev.confusion(["positive", "negative"], ["neutral", "negative"], L)
    assert m.loc["positive", "neutral"] == 1 and m.loc["negative", "negative"] == 1
    assert m.values.sum() == 2


def test_per_class_support():
    t = ev.per_class(["positive", "positive", "neutral"], ["positive", "neutral", "neutral"], L)
    assert t.set_index("label")["support"].to_dict() == {"negative": 0, "neutral": 1, "positive": 2}


def test_bootstrap_interval_contains_point_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = rng.choice(L, 120)
    p = np.where(rng.random(120) < 0.7, y, rng.choice(L, 120))
    a = ev.bootstrap_scores(y, p, L, n_boot=300, seed=3)
    assert ((a["lo"] <= a["value"]) & (a["value"] <= a["hi"])).all()
    pd.testing.assert_frame_equal(a, ev.bootstrap_scores(y, p, L, n_boot=300, seed=3))


def test_weighted_accuracy_and_distribution():
    y = ["positive", "negative", "neutral"]
    p = ["positive", "positive", "neutral"]
    assert ev.weighted_accuracy(y, p, [1, 3, 1]) == pytest.approx(2 / 5)
    d = ev.weighted_distribution(["positive", "positive", "neutral"], [2, 1, 1], L)
    assert d.to_dict() == {"negative": 0.0, "neutral": 0.25, "positive": 0.75}


@pytest.mark.parametrize("c, label", [(0.05, "positive"), (0.049, "neutral"), (-0.05, "negative"), (0.0, "neutral")])
def test_vader_cutoffs(c, label):
    assert se.vader_label(c) == label


def test_gas_sentences_keeps_only_substantive():
    text = ("Tanzania has gold, coal and natural gas. Natural gas output doubled this year. "
            "Roads were built.")
    assert se.gas_sentences(text) == ["Natural gas output doubled this year."]


def test_mean_probs():
    m = se.mean_probs([{"negative": .1, "neutral": .3, "positive": .6},
                       {"negative": .3, "neutral": .3, "positive": .4}])
    assert m == pytest.approx({"negative": .2, "neutral": .3, "positive": .5})


def test_cap_words():
    assert se.cap_words("a b c", 5) == "a b c"
    assert se.cap_words("a b c d e f", 4) == "a b c d"


def test_in_chunks_preserves_order_and_reports_progress():
    seen = []
    out = se._in_chunks(list(range(7)), 3, lambda c: [x * 10 for x in c], lambda i, n: seen.append((i, n)))
    assert out == [0, 10, 20, 30, 40, 50, 60]
    assert seen == [(3, 7), (6, 7), (7, 7)]
