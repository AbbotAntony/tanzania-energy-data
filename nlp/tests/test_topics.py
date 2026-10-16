"""Tests for Task 2 topic-model evaluation code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from energy_nlp import topics as tp


def test_npmi_hand_computed():
    # 4 docs x 3 terms. Terms 0 and 1 always co-occur; term 2 never meets term 0.
    X = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1]]))
    # pair (0,1): p0=.5, p1=.75, p01=.5 -> log(.5/.375)/-log(.5)
    expected01 = np.log(0.5 / 0.375) / -np.log(0.5)
    assert tp.npmi_coherence(np.array([[0, 1]]), X)[0] == pytest.approx(expected01)
    assert tp.npmi_coherence(np.array([[0, 2]]), X)[0] == -1.0     # never co-occur
    # topic of three terms = mean of its three pairs
    p12 = np.log(0.25 / (0.75 * 0.5)) / -np.log(0.25)
    assert tp.npmi_coherence(np.array([[0, 1, 2]]), X)[0] == pytest.approx((expected01 - 1 + p12) / 3)


def test_npmi_perfect_pair_is_one():
    X = sparse.csr_matrix(np.array([[1, 1], [1, 1], [0, 0]]))
    assert tp.npmi_coherence(np.array([[0, 1]]), X)[0] == pytest.approx(1.0)


def test_topic_diversity():
    assert tp.topic_diversity(np.array([[0, 1], [2, 3]])) == 1.0
    assert tp.topic_diversity(np.array([[0, 1], [1, 2]])) == 0.75


def test_match_similarity_ignores_topic_order():
    rng = np.random.default_rng(0)
    H = rng.random((4, 30))
    assert tp.match_similarity(H, H[[2, 0, 3, 1]]) == pytest.approx(1.0)
    assert tp.stability([H, H[::-1], H]) == pytest.approx(1.0)
    assert tp.match_similarity(H, rng.random((4, 30))) < 1.0


def test_doc_proportions_rows_sum_to_one():
    W = np.array([[1.0, 3.0], [0.0, 0.0]])
    P = tp.doc_proportions(W)
    assert P.tolist() == [[0.25, 0.75], [0.5, 0.5]]


def test_period_shares_mean_and_interval():
    props = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
    periods = pd.Series(["A", "A", "B", "B"])
    out = tp.period_shares(props, periods, n_boot=200, seed=1)
    a0 = out[(out["period"] == "A") & (out["topic"] == 0)].iloc[0]
    assert a0["share"] == 0.5 and a0["n_articles"] == 2
    assert a0["lo"] <= 0.5 <= a0["hi"]
    b0 = out[(out["period"] == "B") & (out["topic"] == 0)].iloc[0]
    assert b0["lo"] == b0["hi"] == 0.5             # no variation -> zero-width interval
    # reproducible with a fixed seed
    pd.testing.assert_frame_equal(out, tp.period_shares(props, periods, n_boot=200, seed=1))


def test_share_change_by_group_sums_to_total_change():
    flag = pd.Series([True, False, True, True, True, False, True, False])
    group = pd.Series(["x", "y", "y", "x", "x", "y", "z", "z"])
    period = pd.Series(["A"] * 4 + ["B"] * 4)
    out = tp.share_change_by_group(flag, group, period, "A", "B").set_index("group")
    total = flag[period == "B"].mean() - flag[period == "A"].mean()
    assert out["contribution"].sum() == pytest.approx(total)
    assert out.loc["x", "A"] == 0.5 and out.loc["x", "B"] == 0.25
    assert out.loc["z", "A"] == 0.0 and out.loc["z", "B"] == 0.25


def test_person_names_rule():
    texts = ["Minister Doto Biteko said", "Dr Biteko opened", "Mr Biteko and President Samia"] * 2
    names = tp.person_names(texts, min_docs=5)
    assert "biteko" in names and "doto" not in names     # doto only twice
    assert tp.person_names(["Mr Julius Nyerere"] * 5) == set()   # protected: dam name


def test_person_names_drops_mostly_lowercase_words():
    texts = ["Mr Mark Rutte spoke", "a mark of respect", "will mark the day", "an indelible mark",
             "Mr Mark Rutte", "Mr Mark Rutte", "Mr Mark Rutte", "Mr Mark Rutte"]
    names = tp.person_names(texts, min_docs=5)
    assert "rutte" in names and "mark" not in names     # mark: 5 of 8 uses capitalised


def test_share_change_with_intervals_brackets_point_estimate():
    rng = np.random.default_rng(0)
    n = 400
    flag = pd.Series(rng.random(n) < 0.6)
    group = pd.Series(rng.choice([-1, 0, 1, 2], n))
    period = pd.Series(np.repeat(["A", "B"], n // 2))
    out = tp.share_change_with_intervals(flag, group, period, "A", "B", n_boot=300, seed=1)
    assert set(out["group"]) == {-1, 0, 1, 2}
    assert (out["lo"] <= out["contribution"]).all() and (out["contribution"] <= out["hi"]).all()
    point = tp.share_change_by_group(flag, group, period, "A", "B")
    pd.testing.assert_series_equal(out["contribution"], point["contribution"])
