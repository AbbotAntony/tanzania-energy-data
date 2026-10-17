"""Tests for the second-pass near-duplicate detection and its application in prep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_nlp import dedup, embed, prep


@pytest.mark.parametrize("shared, cos, days, title, kind", [
    (0.75, 0.90, 200, 0.0, "repost"),                 # reprint months later still a repost
    (0.12, 0.987, 2, 0.89, "paraphrased_repost"),     # S002/S072-like rewrite
    (0.12, 0.987, 30, 0.89, None),                    # same story a month later: not a repost
    (0.12, 0.987, 2, 0.20, None),                     # different headline
    (0.50, 0.90, 100, 0.0, "partial_reuse"),
    (0.20, 0.95, 1, 0.3, None),
])
def test_classify_pair(shared, cos, days, title, kind):
    assert dedup.classify_pair(shared, cos, days, title) == kind


def test_candidate_pairs_and_default_decisions():
    base = " ".join(f"w{i}" for i in range(80))
    df = pd.DataFrame({
        "post_id": [1, 2, 3],
        "pub_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-06-01"]),
        "title": ["Gas output doubles", "Gas output doubles again", "Unrelated"],
        "text": [base, base + " extra words here", "completely different text about roads and bridges"],
        "url": ["u1", "u2", "u3"],
    })
    emb = np.array([[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    pairs = dedup.candidate_pairs(df, emb)
    assert pairs[["post_a", "post_b", "kind"]].values.tolist() == [[1, 2, "repost"]]
    d = dedup.default_decisions(pairs)
    assert d["decision"].tolist() == ["remove_later"]


def test_near_duplicate_groups_keep_earliest_and_skip_keep_both():
    df = pd.DataFrame({"post_id": [10, 20, 30, 40, 50],
                       "pub_date": pd.to_datetime(["2023-04-26", "2023-04-26", "2023-10-16", "2024-01-01",
                                                   "2024-01-02"])})
    decisions = pd.DataFrame({"post_a": [20, 10, 40, 99], "post_b": [30, 20, 50, 10],
                              "kind": ["repost", "repost", "partial_reuse", "repost"],
                              "decision": ["remove_later", "remove_later", "keep_both", "remove_later"]})
    out = prep.near_duplicate_groups(decisions, df).sort_values("post_id")
    # 10/20/30 form one group (chained); same-day tie -> lower id kept; 99 is not in the corpus
    assert out.values.tolist() == [[20, 10], [30, 10]]


def test_cached_embeddings_reuses_by_id(tmp_path, monkeypatch):
    calls = []

    def fake(texts):
        calls.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts])

    monkeypatch.setattr(embed, "embed_documents", fake)
    path, ids = tmp_path / "e.npy", tmp_path / "ids.csv"
    a = embed.cached_embeddings([1, 2], ["a", "bb"], path, ids)
    b = embed.cached_embeddings([2, 3, 1], ["bb", "ccc", "a"], path, ids)     # only 3 is new
    assert calls == [["a", "bb"], ["ccc"]]
    assert b.tolist() == [a[1].tolist(), [3.0, 1.0], a[0].tolist()]
