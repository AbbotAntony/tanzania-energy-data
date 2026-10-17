"""Tests for the Task 5 coverage-energy alignment helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_nlp import link


def _bot(tmp_path):
    rows = []
    def add(year, month, cat, v, latest=1, ptype="quarter_ending"):
        rows.append({"year": year, "month": month, "fy": None, "variable": "gas_consumption", "category": cat,
                     "value_mmscf": v, "latest_edition": latest, "x_period_type": ptype})
    for (y, m), tot, pw in [((2023, 3), 100.0, 80.0), ((2023, 6), 110.0, 85.0), ((2023, 12), 110.0, 85.0),
                            ((2024, 3), 90.0, 70.0)]:
        add(y, m, "TOTAL", tot); add(y, m, "Power generating plants", pw)
    add(2023, 3, "TOTAL", 999.0, latest=0)                      # older edition ignored
    add(2023, None, "TOTAL", 5000.0, ptype="fiscal_year")        # FY totals ignored
    path = tmp_path / "bot.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_bot_quarterly_latest_edition_gaps_and_repeat_flag(tmp_path):
    q = link.bot_quarterly(_bot(tmp_path))
    assert [str(p) for p in q.index] == ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"]
    assert q.loc[pd.Period("2023Q1", "Q"), "gas_total"] == 100.0
    assert np.isnan(q.loc[pd.Period("2023Q3", "Q"), "gas_total"])          # missing quarter kept as NaN
    assert q["repeat_of_previous"].tolist() == [False, False, False, False, False]  # gap breaks the repeat


def test_repeat_flag_on_consecutive_equal_quarters(tmp_path):
    rows = [{"year": 2022, "month": m, "variable": "gas_consumption", "category": c, "value_mmscf": v,
             "latest_edition": 1, "x_period_type": "quarter_ending"}
            for m, tot in [(6, 18.0), (9, 19.0), (12, 19.0)]
            for c, v in [("TOTAL", tot), ("Power generating plants", tot - 3)]]
    path = tmp_path / "bot.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert link.bot_quarterly(path)["repeat_of_previous"].tolist() == [False, False, True]


def test_focus_hits_title_or_three_mentions():
    texts = pd.Series(["natural gas once", "natural gas, natural gas and natural gas", "nothing"])
    titles = pd.Series(["Roads", "Budget", "Natural gas prices rise"])
    assert link.focus_hits(texts, titles, "natural_gas").tolist() == [False, True, True]


def test_coverage_quarterly_shares():
    c = pd.DataFrame({
        "quarter": pd.PeriodIndex(["2024Q1", "2024Q1", "2024Q2"], freq="Q"),
        "topics": ["natural_gas;jnhpp", "jnhpp", "natural_gas"],
        "title": ["JNHPP turbine", "x", "y"],
        "text": ["natural gas", "JNHPP", "natural gas natural gas natural gas"],
    })
    out = link.coverage_quarterly(c)
    assert out.loc[pd.Period("2024Q1", "Q"), "gas_mention_share"] == 0.5
    assert out.loc[pd.Period("2024Q1", "Q"), "jnhpp_focus_share"] == 0.5
    assert out.loc[pd.Period("2024Q2", "Q"), "gas_focus_share"] == 1.0


def test_lagged_correlations_detects_known_lead():
    rng = np.random.default_rng(1)
    idx = pd.period_range("2020Q1", periods=40, freq="Q")
    x = pd.Series(rng.normal(size=40).cumsum(), index=idx)
    y = x.shift(2)                                    # y follows x two quarters later
    out = link.lagged_correlations(x, y).set_index("lag_quarters")
    assert out.loc[2, "r"] == pytest.approx(1.0)
    assert abs(out.loc[0, "r"]) < 0.5
    assert out.loc[2, "r_crit_5pct"] == pytest.approx(1.96 / np.sqrt(out.loc[2, "n"]))


def test_calendar_year_totals_needs_four_quarters():
    idx = pd.period_range("2023Q1", "2024Q3", freq="Q")
    s = pd.Series([1, 2, 3, 4, 5, 6, 7], index=idx, dtype=float)
    assert link.calendar_year_totals(s).to_dict() == {2023: 10.0}
