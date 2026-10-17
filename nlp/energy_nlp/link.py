"""Task 5: line up news coverage with the gas consumption series (descriptive only).

Gas data: Bank of Tanzania Consolidated Zonal Economic Report, Table 3.13
(source TPDC), quarter-ending values, latest edition (the project convention:
flag suspicious values, never override them).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from parsers.news import TOPICS

# JNHPP milestones as reported in the corpus (post_id = the Daily News article cited).
MILESTONES = [
    ("2022-12-22", "Reservoir impounding begins", 14378),
    ("2024-02-14", "Turbine 9 switched on (235 MW)", 51960),
    ("2024-07-30", "3 turbines, 705 MW", 64176),
    ("2024-12-20", "5 turbines, ~1,175 MW", 80383),
    ("2025-04-05", "All 9 turbines (2,115 MW)", 123600),
    ("2026-08-22", "Official commissioning", 157661),
]


def bot_quarterly(path: Path) -> pd.DataFrame:
    """Quarterly gas consumption (MMscf): total and power generation, latest edition.

    Every quarter from the first to the last report is present; missing quarters
    are NaN. ``repeat_of_previous`` flags a quarter whose total equals the
    previous quarter's exactly (a known source repeat), kept as published.
    """
    g = pd.read_csv(path)
    q = g[(g["x_period_type"] == "quarter_ending") & (g["latest_edition"] == 1)
          & (g["variable"] == "gas_consumption")]
    q = q.assign(quarter=pd.PeriodIndex.from_fields(year=q["year"].astype(int), month=q["month"].astype(int),
                                                    freq="Q"))
    wide = q.pivot_table(index="quarter", columns="category", values="value_mmscf", aggfunc="first")
    out = pd.DataFrame({"gas_total": wide["TOTAL"], "gas_power": wide["Power generating plants"]})
    out = out.reindex(pd.period_range(out.index.min(), out.index.max(), freq="Q"))
    out.index.name = "quarter"
    out["repeat_of_previous"] = out["gas_total"].eq(out["gas_total"].shift(1)) & out["gas_total"].notna()
    return out


def focus_hits(texts: pd.Series, titles: pd.Series, topic: str, min_hits: int = 3) -> pd.Series:
    """Article is *about* the topic: phrase in the title or >= ``min_hits`` mentions."""
    rx = TOPICS[topic]
    return titles.str.contains(rx) | (texts.map(lambda t: len(rx.findall(t))) >= min_hits)


def coverage_quarterly(corpus: pd.DataFrame) -> pd.DataFrame:
    """Per quarter: articles, and shares mentioning / focused on natural gas and JNHPP."""
    tags = corpus["topics"].str.split(";")
    full = corpus["title"] + ". " + corpus["text"]
    df = pd.DataFrame({
        "quarter": corpus["quarter"],
        "gas_mention": tags.map(lambda t: "natural_gas" in t),
        "gas_focus": focus_hits(full, corpus["title"], "natural_gas"),
        "jnhpp_mention": tags.map(lambda t: "jnhpp" in t),
        "jnhpp_focus": focus_hits(full, corpus["title"], "jnhpp"),
    })
    out = df.groupby("quarter").agg(n_articles=("gas_mention", "size"),
                                    **{f"{c}_share": (c, "mean") for c in df.columns[1:]})
    return out


def lagged_correlations(x: pd.Series, y: pd.Series, lags=range(-3, 4)) -> pd.DataFrame:
    """Pearson r between quarter-on-quarter changes of ``x`` (lagged by k) and ``y``.

    Positive k means x leads y by k quarters (x at t-k against y at t). Changes,
    not levels, are used so that two trending series do not correlate by
    construction. ``r_crit`` is the two-sided 5% threshold for n pairs, ignoring
    autocorrelation (so it is, if anything, too lenient).
    """
    dx, dy = x.diff(), y.diff()
    rows = []
    for k in lags:
        pair = pd.concat([dx.shift(k), dy], axis=1).dropna()
        n = len(pair)
        r = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if n >= 3 else np.nan
        rows.append({"lag_quarters": k, "r": r, "n": n, "r_crit_5pct": 1.96 / np.sqrt(n) if n else np.nan})
    return pd.DataFrame(rows)


def calendar_year_totals(q: pd.Series) -> pd.Series:
    """Sum of the four quarters for years with all four quarters reported."""
    s = q.dropna()
    years = s.groupby(s.index.year)
    return years.sum()[years.size() == 4]


def milestone_table(corpus: pd.DataFrame) -> pd.DataFrame:
    """Milestones with the citing article's date and URL (from the full corpus)."""
    ref = corpus.set_index("post_id")
    rows = []
    for date, event, pid in MILESTONES:
        rows.append({"date": date, "event": event, "source_post_id": pid,
                     "source_pub_date": ref.at[pid, "pub_date"].date(), "source_url": ref.at[pid, "url"]})
    return pd.DataFrame(rows)
