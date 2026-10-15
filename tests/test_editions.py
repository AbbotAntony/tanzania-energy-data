"""latest_edition flag: the most recent report wins, all rows are kept."""

from __future__ import annotations

import pandas as pd

from pipeline.editions import mark_latest_edition


def test_latest_edition_is_document_covering_latest_period() -> None:
    df = pd.DataFrame([
        # FY2021/22 report: households 2021/22 = 1407
        {"source": "EWURA", "url": "a", "pub_date": "2022-06-30", "variable": "customers",
         "category": "Households", "fy": "2021/22", "year": 2022},
        # FY2022/23 report revises it to 970 and also covers 2022/23
        {"source": "EWURA", "url": "b", "pub_date": "2024-06-14", "variable": "customers",
         "category": "Households", "fy": "2021/22", "year": 2022},
        {"source": "EWURA", "url": "b", "pub_date": "2024-06-14", "variable": "customers",
         "category": "Households", "fy": "2022/23", "year": 2023},
    ])
    out = mark_latest_edition(df)
    assert len(out) == 3
    flags = dict(zip(zip(out.url, out.fy), out.latest_edition))
    assert flags == {("a", "2021/22"): 0, ("b", "2021/22"): 1, ("b", "2022/23"): 1}


def test_exactly_one_latest_row_per_observation() -> None:
    df = pd.DataFrame([
        {"source": "S", "url": u, "pub_date": d, "variable": "v", "category": "c",
         "fy": None, "year": 2020}
        for u, d in (("x", "2021-01-01"), ("y", "2022-01-01"), ("z", None))
    ])
    out = mark_latest_edition(df)
    assert out.latest_edition.sum() == 1
    assert out.loc[out.latest_edition == 1, "url"].item() == "y"
