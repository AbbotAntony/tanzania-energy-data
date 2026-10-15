"""EWURA petroleum report and cap-price parsers, on page text cut from real PDFs."""

from __future__ import annotations

from parsers import ewura_cap_prices, ewura_petroleum
from tests.conftest import load_pages, make_doc


def _one(records, **want):
    hits = [r for r in records
            if all((r.extra.get(k[2:]) if k.startswith("x_") else getattr(r, k)) == v
                   for k, v in want.items())]
    assert len(hits) == 1, f"{want}: {len(hits)} matches"
    return hits[0]


# ---------------------------------------------------------------- petroleum
def test_fy_imports_by_product_with_split_section_label() -> None:
    """FY2022/23 report prints "Transit Im-" / "ports" across two lines."""
    recs = ewura_petroleum.parse_pages(load_pages("ewura/petroleum_fy2223_pages.json"),
                                       make_doc("Petroleum FY 2022/23", "petroleum_report"))
    hfo_local = _one(recs, variable="petroleum_imports", category="HFO", fy="2022/23", x_market="Local")
    assert hfo_local.value == 14427625 and hfo_local.unit == "litres" and hfo_local.year == 2023
    assert _one(recs, variable="petroleum_imports", category="HFO", fy="2022/23",
                x_market="Transit").value == 3569277


def test_fy_totals_and_consumption() -> None:
    recs = ewura_petroleum.parse_pages(load_pages("ewura/petroleum_fy2223_pages.json"),
                                       make_doc("Petroleum FY 2022/23", "petroleum_report"))
    assert _one(recs, variable="petroleum_imports_total", fy="2021/22",
                x_market="Total").value == 7591418185
    assert _one(recs, variable="petroleum_consumption", category="HFO",
                fy="2022/23").value == 33283991


def test_calendar_year_tables_and_jet_a1() -> None:
    recs = ewura_petroleum.parse_pages(load_pages("ewura/petroleum_2021_pages.json"),
                                       make_doc("Petroleum 2021", "petroleum_report"))
    hfo = _one(recs, variable="petroleum_imports", category="HFO", year=2020, x_market="Local")
    assert hfo.value == 26795952 and hfo.fy is None and hfo.extra["period_type"] == "calendar"
    # the "1" in "Jet A-1" must not be read as a value
    assert _one(recs, variable="petroleum_imports", category="Jet A-1", year=2020,
                x_market="Local").value == 118254496
    assert _one(recs, variable="petroleum_imports_total", year=2021,
                x_market="Transit").value == 3112410734


def test_canon_product() -> None:
    assert ewura_petroleum.canon_product("Jet A-1/ Kerosene") == "Jet A-1/Kerosene"
    assert ewura_petroleum.canon_product("Dizeli") == "Diesel (AGO)"
    assert ewura_petroleum.canon_product("Taa") == "Kerosene"


# ---------------------------------------------------------------- cap prices
def test_cap_prices_2026() -> None:
    recs = ewura_cap_prices.parse_pages(load_pages("ewura/cap_2026_05_pages.json"),
                                        make_doc("Cap Prices May 2026", "cap_prices"))
    r = _one(recs, variable="cap_price_retail", category="Diesel (AGO)", x_port="Dar es Salaam")
    assert r.value == 4248 and r.unit == "TZS/litre" and (r.year, r.month) == (2026, 5)
    assert r.extra["effective_date"] == "2026-05-06"
    w = _one(recs, variable="cap_price_wholesale", category="Kerosene", x_port="Tanga")
    assert w.value is None  # printed "-"


def test_subsidy_columns_are_labelled() -> None:
    recs = ewura_cap_prices.parse_pages(load_pages("ewura/cap_2022_07_subsidy_pages.json"),
                                        make_doc("Cap Prices July 2022", "cap_prices"))
    without = _one(recs, variable="cap_price_retail", category="Petrol (PMS)",
                   x_port="Dar es Salaam", x_price_basis="cap price without subsidy")
    with_s = _one(recs, variable="cap_price_retail", category="Petrol (PMS)",
                  x_port="Dar es Salaam", x_price_basis="cap price with subsidy")
    assert (without.value, with_s.value) == (3497, 3220)


def test_floor_prices_kept_separate_and_levy_table_ignored() -> None:
    recs = ewura_cap_prices.parse_pages(load_pages("ewura/cap_2022_02_floor_pages.json"),
                                        make_doc("Cap Prices Feb 2022", "cap_prices"))
    cap = _one(recs, variable="cap_price_wholesale", category="Petrol (PMS)",
               x_port="Dar es Salaam", x_price_basis="cap price")
    floor = _one(recs, variable="cap_price_wholesale", category="Petrol (PMS)",
                 x_port="Dar es Salaam", x_price_basis="floor price")
    assert (cap.value, floor.value) == (2350.54, 2227.54)


def test_swahili_notice() -> None:
    recs = ewura_cap_prices.parse_pages(load_pages("ewura/cap_2025_01_swahili_pages.json"),
                                        make_doc("Bei Kikomo Januari 2025", "cap_prices"))
    retail = [r for r in recs if r.variable == "cap_price_retail" and r.extra["port"] == "Dar es Salaam"]
    assert sorted(r.value for r in retail) == [2644, 2676, 2793]
    wholesale = _one(recs, variable="cap_price_wholesale", category="Petrol (PMS)",
                     x_port="Dar es Salaam")
    assert wholesale.value == 2662.34 and (wholesale.year, wholesale.month) == (2025, 1)
