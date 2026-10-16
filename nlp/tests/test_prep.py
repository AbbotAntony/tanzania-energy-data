"""Tests for Task 1 data prep."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from energy_nlp import prep

BODY = " ".join(f"word{i}" for i in range(60))


def test_clean_html_drops_cross_links_and_captions():
    html = (
        "<p>THE government said natural gas use will grow.</p>\n"
        '<p>ALSO READ: <a href="x">Biteko stresses natural gas use</a></p>\n'
        '<p><strong><a href="y">ALSO READ: LNG project inches closer</a></strong></p>\n'
        '<p><strong>Read also: </strong><strong><a href="z">Other story</a></strong></p>\n'
        '<div class="wp-caption"><img src="a.jpg"><p class="wp-caption-text">Minister at Kinyerezi</p></div>\n'
        "<figure><img src='b.jpg'><figcaption>JNHPP dam</figcaption></figure>\n"
        "<p>He added that prices would fall.</p>"
    )
    assert prep.clean_html(html) == ("THE government said natural gas use will grow. "
                                     "He added that prices would fall.")


def test_clean_html_keeps_long_paragraph_starting_with_related():
    para = "Related challenges remain for the sector, " + BODY
    assert prep.clean_html(f"<p>{para}</p>") == para


def test_normalise_maps_typography_and_handles_missing():
    assert prep.normalise("Samia’s “bold” move​ – now&amp;then") == \
        "Samia's \"bold\" move - now&then"
    assert prep.normalise(math.nan) == ""
    assert prep.normalise(None) == ""


@pytest.mark.parametrize("text, expected", [
    ("DAR ES SALAAM: THE government", ("DAR ES SALAAM", "THE government")),
    ("DODOMA : DEPUTY Prime Minister", ("DODOMA", "DEPUTY Prime Minister")),
    ("THE government said: yes", ("", "THE government said: yes")),
    ("Speaking: the minister", ("", "Speaking: the minister")),
])
def test_split_dateline(text, expected):
    assert prep.split_dateline(text) == expected


def test_strong_hits_ignores_weak_topics():
    assert prep.strong_hits("Mtwara pipeline project") == 0
    assert prep.strong_hits("natural gas from Mtwara via the natural gas pipeline; TPDC said") == 3


def _docs(rows):
    return pd.DataFrame(rows, columns=["post_id", "pub_date", "text"]).assign(
        pub_date=lambda d: pd.to_datetime(d["pub_date"]))


def test_find_duplicates_keeps_earliest_and_groups_transitively():
    a = BODY
    b = BODY + " extra"                      # repost of a
    c = BODY + " extra more"                 # repost of b
    df = _docs([(30, "2024-01-03", c), (10, "2024-01-01", a), (20, "2024-01-02", b),
                (40, "2024-01-01", "a completely different story about hydropower dams")])
    dups = prep.find_duplicates(df).set_index("post_id")
    assert set(dups.index) == {20, 30}
    assert (dups["dup_of"] == 10).all()
    assert dups["jaccard"].min() >= prep.DUP_JACCARD


def test_find_duplicates_tie_on_date_keeps_lower_id():
    df = _docs([(7, "2024-05-01", BODY), (5, "2024-05-01", BODY)])
    assert prep.find_duplicates(df).to_dict("records") == [{"post_id": 7, "dup_of": 5, "jaccard": 1.0}]


def test_find_duplicates_none():
    df = _docs([(1, "2024-01-01", BODY), (2, "2024-01-01", "short unrelated text about gas prices")])
    assert prep.find_duplicates(df).empty


def _corpus():
    dates = ["2022-07-05", "2022-07-20", "2022-09-01", "2022-09-02", "2022-10-01"]
    df = pd.DataFrame({
        "pub_date": pd.to_datetime(dates),
        "topics": ["natural_gas", "natural_gas;jnhpp", "jnhpp", "natural_gas", "jnhpp"],
        "include": [True, True, True, False, True],
    })
    df["month"] = df["pub_date"].dt.to_period("M")
    df["quarter"] = df["pub_date"].dt.to_period("Q")
    df["in_main_window"] = df["quarter"] >= prep.MAIN_WINDOW_START
    return df


def test_counts_by_month_fills_gaps_and_flags():
    out = prep.counts_by_period(_corpus(), "M", last_complete=pd.Period("2022-09", "M"), min_articles=2)
    assert out["month"].tolist() == ["2022-07", "2022-08", "2022-09", "2022-10"]
    assert out["n_articles"].tolist() == [2, 0, 1, 1]          # excluded row not counted
    assert out["empty"].tolist() == [False, True, False, False]
    assert out["low_count"].tolist() == [False, True, True, True]
    assert out["partial"].tolist() == [False, False, False, True]
    assert out["natural_gas"].tolist() == [2, 0, 0, 0]
    assert out["jnhpp"].tolist() == [1, 0, 1, 1]
    assert out["hfo"].sum() == 0                               # every tag column present


def test_counts_by_quarter_scales_threshold():
    out = prep.counts_by_period(_corpus(), "Q", min_articles=1)   # quarter floor = 3
    assert out["quarter"].tolist() == ["2022Q3", "2022Q4"]
    assert out["n_articles"].tolist() == [3, 1]
    assert out["low_count"].tolist() == [False, True]
    assert not out["partial"].any()


def test_prep_flow_adds_up():
    df = pd.DataFrame({
        "text_source": ["html"] * 5,
        "off_topic": [False, True, False, False, False],
        "is_duplicate": [False, False, True, False, False],
        "in_main_window": [False, True, True, True, True],
        "passing_mention": [False, False, False, True, False],
    })
    df["include"] = ~df["off_topic"] & ~df["is_duplicate"]
    n = dict(prep.prep_flow(df).values)
    assert n["included articles"] == 3
    assert n["scraped articles"] - n["off-topic after removing links/captions"] \
        - n["near-duplicates removed"] == n["included articles"]
    assert n["  of which before 2022-Q3 (descriptive only)"] \
        + n["  of which in main window (2022-Q3 on)"] == n["included articles"]
