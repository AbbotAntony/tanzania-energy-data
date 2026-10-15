"""EWURA petroleum report and cap-price parsers, on page text cut from real PDFs."""

from __future__ import annotations

from parsers import ewura_cap_prices, ewura_petroleum
from pipeline.provenance import Record
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
    assert ewura_petroleum.canon_product("MSP") == "Petrol (PMS)"
    # running page footer, not a product
    assert ewura_petroleum.canon_product(
        "Downstream Petroleum Sub Sector Performance Review Report for Year") is None


def _assert_no_year_values(recs) -> None:
    years = [(r.variable, r.year, r.category, r.value_as_printed) for r in recs
             if r.variable.startswith("petroleum_imports") and r.value is not None
             and (ewura_petroleum._BARE_YEAR.match(r.value_as_printed or "") or r.value < 10_000)]
    assert not years, f"year labels read as litre values: {years}"


def test_page_footer_year_not_read_as_petrol_imports() -> None:
    """2016 report: footer "...Review Report for Year 2016 | 11" gave Petrol 2015=11, 2016=2016."""
    recs = ewura_petroleum.parse_pages(load_pages("ewura/petroleum_2016_footer_pages.json"),
                                       make_doc("Petroleum 2016", "petroleum_report"))
    _assert_no_year_values(recs)
    assert not [r for r in recs if r.variable == "petroleum_imports"]
    assert _one(recs, variable="petroleum_imports_total", year=2015,
                x_market="Local").value == 3338097164


def test_totals_header_row_year_not_read_as_total_imports() -> None:
    """FY2020 report: "Description Local Imports Transit Imports Total" + "2019 ..." gave TOTAL=2019."""
    recs = ewura_petroleum.parse_pages(load_pages("ewura/petroleum_2020_header_total_pages.json"),
                                       make_doc("Petroleum FY 2020", "petroleum_report"))
    _assert_no_year_values(recs)
    assert not [r for r in recs if r.variable == "petroleum_imports" and r.category == "TOTAL"]
    assert _one(recs, variable="petroleum_imports_total", year=2019,
                x_market="Local").value == 3537891891
    assert _one(recs, variable="petroleum_imports", category="HFO", year=2019,
                x_market="Local").value == 19838051


def test_pre_2019_consumption_caption() -> None:
    """2018 report captions the table just "Petroleum Product Consumption"."""
    recs = ewura_petroleum.parse_pages(load_pages("ewura/petroleum_2018_consumption_pages.json"),
                                       make_doc("Petroleum 2018", "petroleum_report"))
    assert _one(recs, variable="petroleum_consumption", category="HFO", year=2017).value == 75162726
    assert _one(recs, variable="petroleum_consumption", category="HFO", year=2018).value == 42790511


def test_appendix_annual_totals_2013_14_edition() -> None:
    """Local and transit side by side; the row-total column must equal the product sum."""
    recs = ewura_petroleum.parse_pages(load_pages("ewura/petroleum_2013_14_appendix_pages.json"),
                                       make_doc("Petroleum 2013/14", "petroleum_report"))
    hfo12 = _one(recs, variable="petroleum_imports", category="HFO", year=2012, x_market="Local")
    assert hfo12.value == 103239448 and "sum_check_failed" not in hfo12.qa_flag
    assert _one(recs, variable="petroleum_imports", category="HFO", year=2013,
                x_market="Transit").value == 1533708
    assert _one(recs, variable="petroleum_imports", category="Petrol (PMS)", year=2013,
                x_market="Local").value == 871554958
    assert all("sum_check_failed" not in r.qa_flag for r in recs)


def test_appendix_and_consumption_2015_ocr_edition() -> None:
    """Scanned 2015 report: rotated OCR of Appendices 3/4 (two years per page) and Table 15."""
    recs = ewura_petroleum.parse_pages(load_pages("ewura/petroleum_2015_ocr_appendix_pages.json"),
                                       make_doc("Petroleum 2015", "petroleum_report"))
    hfo15 = _one(recs, variable="petroleum_imports", category="HFO", year=2015, x_market="Local")
    assert hfo15.value == 253635446 and "sum_check_failed" not in hfo15.qa_flag
    assert _one(recs, variable="petroleum_imports", category="HFO", year=2014,
                x_market="Local").value == 168861198
    # transit total less HFO "localized" is printed negative
    assert _one(recs, variable="petroleum_imports", category="HFO", year=2015,
                x_market="Transit").value == -19595274
    assert all("sum_check_failed" not in r.qa_flag for r in recs)
    assert _one(recs, variable="petroleum_consumption", category="HFO", year=2014).value == 162459821
    assert _one(recs, variable="petroleum_consumption", category="HFO", year=2015).value == 223890081


def _narr(text: str, page: int = 1) -> Record:
    return Record(source="EWURA", url="https://www.ewura.go.tz/n.pdf", doc_title="Report",
                  retrieved_date="2026-10-15", variable="explanatory_text", value=None,
                  unit="text", pdf_page=page, value_as_printed=text)


def _imp(variable: str, year: int, value: float, category: str = "HFO") -> Record:
    return Record(source="EWURA", url="u", doc_title="d", retrieved_date="2026-10-15",
                  variable=variable, value=value, unit="litres", year=year,
                  category=category, extra={"market": "Local"})


def test_estimate_hfo_imports_chain_and_share_check() -> None:
    records = [_imp("petroleum_imports", 2015, 253635446),
               _imp("petroleum_imports", 2019, 19838051),
               _imp("petroleum_imports_total", 2016, 3302298898, "All products"),
               _imp("petroleum_imports_total", 2017, 3193252759, "All products")]
    narrative = [
        _narr("Whilst diesel ... increased, Heavy Fuel Oil (HFO) imports decreased by 75%. As a"),
        _narr("The proportions of diesel, petrol, JetA1/kerosene and HFO in the local imports in "
              "the year under review were 55%, 34%, 9% and 2% respectively"),
        _narr("local imports for diesel, kerosene, Jet-A1 and Heavy Furnace Oil (HFO) decreased "
              "by 3.9%, 54%, 18% and 30% respectively"),
        _narr("The proportion of diesel, petrol, JetA1/kerosene and HFO in the local imports in "
              "the year under review were 55%, 37%, 7% and 1% respectively"),
        _narr("Likewise, HFO importation decreased by 4% due to the continued increase"),
    ]
    est = {r.year: r for r in ewura_petroleum.estimate_hfo_imports(records, narrative)}
    assert est[2016].value == round(253635446 * 0.25)
    assert est[2017].value == round(253635446 * 0.25 * 0.70)
    assert est[2018].value == round(19838051 / 0.96)
    assert all(r.variable == "petroleum_imports_estimated" and "estimate" in r.qa_flag
               for r in est.values())
    assert "inside range" in est[2016].extra["share_check"]
    assert "inside range" in est[2017].extra["share_check"]
    # without the source sentence there is no estimate
    assert ewura_petroleum.estimate_hfo_imports(records, narrative[1:2]) == []


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
