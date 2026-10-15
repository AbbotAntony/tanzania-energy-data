"""Parse EWURA (Mid and) Downstream Petroleum Sub-Sector Performance Reports.

Extracted (units as published, normally litres):
  * total local / transit imports per period       -> petroleum_imports_total
  * local / transit imports by product per period  -> petroleum_imports (HFO, diesel...)
  * consumption by product per period              -> petroleum_consumption
  * sentences about HFO / power-generation fuel mix -> narrative records
  * annual "Total" rows of the monthly import appendices (2013/14 and 2015
    editions)                                       -> petroleum_imports
  * HFO local imports 2016-2018, estimated from stated % changes
                                                    -> petroleum_imports_estimated

The 2016-2018 editions present product splits only as charts, so exact
product-level imports exist for 2012-2015 (appendices) and from 2019 onward.
The 2015 edition is a scan and is read with OCR; its appendix tables are
printed sideways and are OCR'd after rotating the page.
"""

from __future__ import annotations

import logging
import re

from parsers.ewura_gas import fy_end_year
from parsers.pdf_text import PageText, fiscal_years_in, parse_number, read_pages
from parsers.text_tables import Cell, TableBlock, find_tables, tokenise
from pipeline.provenance import Record
from pipeline.storage import StoredDoc

log = logging.getLogger(__name__)

_IMPORTS_CAPTION = re.compile(r"^Table\s*\d+\s*[:.]?.*Local and Transit.*Imports", re.I)
# "Petroleum Products Consumption in ..." (2019+), "Sales/Consumption of Petroleum
# Products in Years 2016 and 2015", "Comparison of sales Volume/Consumption of ..."
# (2015, 2017) and a bare "Petroleum Product Consumption" (2018).
_CONSUMPTION_CAPTION = re.compile(
    r"^Table\s*\d+\s*[:.]?\s*(?:Comparison of\s+)?(?:(?:sales\s*(?:volume\s*)?/\s*)?"
    r"Consumption of Petroleum Products|Petroleum Products? Consumption(?:\s+in\b|\s*$))", re.I)
_YEAR_ROW = re.compile(r"^\s*(?:Year\s+|FY\s*)?(20\d\d)(?:\s*/\s*(\d\d))?\s+([\d,.\s%()-]*\d[\d,.\s%()-]*)$")
_CONSUMPTION_ROW = re.compile(r"^\s*(?:Year\s+|Total\s+|FY\s*)?(20\d\d)(?:\s*/\s*(\d\d))?\s+"
                              r"([\d,.\s%()-]*\d[\d,.\s%()-]*)$")
_CAL_YEAR = re.compile(r"\b(20\d\d)\b(?!\s*/)")
_BARE_YEAR = re.compile(r"^(?:19|20)\d\d$")


def is_year_token(cell: Cell) -> bool:
    """A printed bare year ("2016") is a label, never a litre value."""
    return bool(_BARE_YEAR.match(cell.printed))


def canon_product(label: str) -> str | None:
    s = label.lower().replace(" ", "")
    if "total" in s:
        return "TOTAL"
    if "petroleum" in s:
        return None  # "Petroleum Products", running page headers: not a product
    if "jet" in s and "kerosene" in s:
        return "Jet A-1/Kerosene"
    if "jet" in s:
        return "Jet A-1"
    if "hfo" in s or "heavyfuel" in s or "fueloil" in s:
        return "HFO"
    if "diesel" in s or "dizeli" in s or "ago" in s or "gasoil" in s:
        return "Diesel (AGO)"
    if "petrol" in s or "pms" in s or "mogas" in s or s == "msp":  # also Swahili "Petroli"
        return "Petrol (PMS)"
    if "kerosene" in s or s in ("ik", "taa"):  # Swahili "Mafuta ya Taa"
        return "Kerosene"
    if s == "ido":
        return "IDO"
    return None


def periods_in(block: TableBlock) -> tuple[list[str], str]:
    """Return column periods and their type ("fy" or "calendar") from caption/header."""
    text = " ".join([block.caption, *block.header_lines])
    fys = list(dict.fromkeys(fiscal_years_in(text)))
    if fys:
        return fys, "fy"
    years = list(dict.fromkeys(_CAL_YEAR.findall(" ".join(block.header_lines) or block.caption)))
    if len(years) < 2:
        years = list(dict.fromkeys(_CAL_YEAR.findall(block.caption)))
    return years, "calendar"


class _Mk:
    def __init__(self, doc: StoredDoc) -> None:
        self.doc = doc

    def rec(self, block: TableBlock, variable: str, cell: Cell, period: str, kind: str,
            category: str, flags: list[str] | None = None, **extra: object) -> Record:
        flags = list(flags or [])
        if block.cid_decoded:
            flags.append("text_glyph_decoded")
        fy = period if kind == "fy" else None
        year = fy_end_year(fy) if fy else int(period)
        return Record(
            source="EWURA", url=self.doc.ref.url, doc_title=self.doc.ref.title,
            pub_date=self.doc.ref.pub_date, retrieved_date=self.doc.retrieved_date,
            pdf_page=block.page_no, table_no=block.table_no, variable=variable,
            value=cell.value, unit="litres", fy=fy, year=year, category=category,
            value_as_printed=cell.printed, qa_flag=";".join(flags),
            extra={"table_caption": block.caption[:200], "period_type": kind, **extra},
        )


def _imports(block: TableBlock, mk: _Mk) -> list[Record]:
    """Handle both the totals-by-year layout and the by-product layout."""
    out: list[Record] = []
    for line in block.body:  # year rows have no text label, so read raw lines
        m = _YEAR_ROW.match(line)
        if m:
            out += _totals_row(block, mk, m)
    periods, kind = periods_in(block)
    section = "Local"
    for row in block.rows:
        if re.search(r"transit\s*im", row.line, re.I):  # also "Transit Im-" / "ports"
            section = "Transit"
        product = canon_product(row.label)
        values = [c for c in row.cells if not c.is_pct]
        if product is None or not periods or len(values) < len(periods):
            continue
        if any(is_year_token(c) for c in values):
            # A totals-by-year row ("Description Local Imports ... Total" + "2019 ...")
            # or a page footer ("... Report for Year 2016 | 11"): handled above / noise.
            continue
        for period, cell in zip(periods, values):
            out.append(mk.rec(block, "petroleum_imports", cell, period, kind, product,
                              market=section))
        if product == "TOTAL":
            section = "Transit"
    return out


def _totals_row(block: TableBlock, mk: _Mk, m: re.Match[str]) -> list[Record]:
    start, fy_end, rest = m.group(1), m.group(2), m.group(3)
    period, kind = (f"{start}/{fy_end}", "fy") if fy_end else (start, "calendar")
    _, cells = tokenise(rest)
    values = [c for c in cells if not c.is_pct]
    out = []
    for market, cell in zip(("Local", "Transit", "Total"), values):
        out.append(mk.rec(block, "petroleum_imports_total", cell, period, kind, "All products",
                          market=market))
    return out


def _consumption(block: TableBlock, mk: _Mk) -> list[Record]:
    """Wide layout: a header line lists products, each following row is a year."""
    products: list[str] = []
    for line in block.body:
        found = [p for p in (canon_product(t) for t in re.findall(r"[A-Za-z][A-Za-z\-0-9]*", line))
                 if p]
        if len(found) >= 3:
            products = found
            break
    out: list[Record] = []
    if not products:
        return out
    for line in block.body:
        m = _CONSUMPTION_ROW.match(line)
        if not m:
            continue
        start, fy_end = m.group(1), m.group(2)
        period, kind = (f"{start}/{fy_end}", "fy") if fy_end else (start, "calendar")
        _, cells = tokenise(m.group(3))
        values = [c for c in cells if not c.is_pct]
        flags = [] if len(values) == len(products) else ["column_count_mismatch"]
        for product, cell in zip(products, values):
            out.append(mk.rec(block, "petroleum_consumption", cell, period, kind, product, flags))
    return out


# ------------------------------------------------------------ monthly import appendices
# 2013/14 edition: one year per page, local and transit side by side, each with a
# row-total column. 2015 edition: one market per page, two years side by side.
_APPENDIX_HEADER = re.compile(r"\bMonth\s+(.*?)\s+Month\b")
_APPENDIX_TOTAL = re.compile(
    r"\bTotal(?:\s+Year\s+20\d\d)?((?:\s+(?:-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-)){4,5})(?=\s|$)")
# 2015 transit HFO is printed negative: transit imports less HFO "localized" for
# the local market (Appendix 4). Kept as printed.
_GRAND_TOTAL = re.compile(r"Grand Total\s+(\d{1,3}(?:,\d{3})+)")


def _appendix_sides(text: str) -> list[tuple[str, int]]:
    """(market, year) of the left and right halves of a monthly-imports appendix page."""
    one_year = re.search(r"Imports for (20\d\d)", text, re.I)
    if one_year and re.search(r"Local imports", text, re.I) and re.search(r"Transit imports", text, re.I):
        year = int(one_year.group(1))
        return [("Local", year), ("Transit", year)]
    return [(m.group(1).title(), int(m.group(2))) for m in
            re.finditer(r"(LOCAL|TRANSIT)\s+IMPORTS\s+FO[RF]\s*YEAR\s+(20\d\d)", text, re.I)]


def appendix_totals(pages: list[PageText], doc: StoredDoc) -> list[Record]:
    """Annual "Total" rows of the monthly imports-by-product appendix tables."""
    out: list[Record] = []
    for page in pages:
        text = " ".join(page.text.split())
        header = _APPENDIX_HEADER.search(text)
        sides = _appendix_sides(text)
        if not header or len(sides) != 2:
            continue
        products = [p for p in (canon_product(t) for t in
                                re.findall(r"[A-Za-z][A-Za-z\-0-9]*", header.group(1))) if p]
        totals = _APPENDIX_TOTAL.findall(text)
        grands = [parse_number(g) for g in _GRAND_TOTAL.findall(text)]
        if len(totals) != 2 or len(products) != 4:
            log.warning("%s p%d: appendix layout not recognised", doc.path.name, page.page_no)
            continue
        for i, ((market, year), group) in enumerate(zip(sides, totals)):
            printed = group.split()
            values = [0.0 if p == "-" else parse_number(p) for p in printed]
            check = values[4] if len(values) == 5 else (grands[i] if i < len(grands) else None)
            flags = ["appendix_annual_total"]
            if check is None:
                flags.append("sum_not_checked")
            elif abs(sum(values[:4]) - check) > 2:
                flags.append("sum_check_failed")
            for product, value, shown in zip(products, values, printed):
                out.append(Record(
                    source="EWURA", url=doc.ref.url, doc_title=doc.ref.title,
                    pub_date=doc.ref.pub_date, retrieved_date=doc.retrieved_date,
                    pdf_page=page.page_no, variable="petroleum_imports", value=value,
                    unit="litres", year=year, category=product, value_as_printed=shown,
                    qa_flag=";".join(flags),
                    extra={"table_caption": "Appendix: monthly imports by product, annual Total row",
                           "period_type": "calendar", "market": market},
                ))
    return out


# ------------------------------------------------------------ HFO 2016-2018 estimates
# The 2016-2018 editions show product imports only as charts, but state HFO's
# change in words. Each phrase is matched verbatim in the narrative records so
# the estimate stays traceable; a missing phrase skips the estimate.
_HFO_CHANGES = (
    # (year estimated, phrase, % change, anchor year)
    (2016, "Heavy Fuel Oil (HFO) imports decreased by 75%", -75.0, 2015),
    (2017, "Heavy Furnace Oil (HFO) decreased by 3.9%, 54%, 18% and 30%", -30.0, 2016),
    (2018, "HFO importation decreased by 4%", -4.0, 2019),
)
# Stated HFO share of local imports (rounded to whole %), used as a cross-check.
_HFO_SHARES = {
    2016: ("HFO in the local imports in the year under review were 55%, 34%, 9% and 2%", 2.0),
    2017: ("HFO in the local imports in the year under review were 55%, 37%, 7% and 1%", 1.0),
}


def _find_sentence(narrative_records: list[Record], phrase: str) -> Record | None:
    for r in narrative_records:
        if phrase in " ".join((r.value_as_printed or "").split()):
            return r
    return None


def _latest(records: list[Record], **want: object) -> Record | None:
    hits = [r for r in records if r.value is not None and all(
        (r.extra.get(k[2:]) if k.startswith("x_") else getattr(r, k)) == v for k, v in want.items())]
    return max(hits, key=lambda r: r.pub_date or "", default=None)


def estimate_hfo_imports(records: list[Record], narrative_records: list[Record]) -> list[Record]:
    """HFO local imports for 2016-2018 from stated % changes, chained from exact years."""
    known: dict[int, tuple[float, str]] = {}
    for year in (2015, 2019):
        r = _latest(records, variable="petroleum_imports", category="HFO", year=year, x_market="Local")
        if r is not None:
            known[year] = (r.value, f"{r.value:,.0f} L ({r.doc_title}, p{r.pdf_page})")
    out: list[Record] = []
    for year, phrase, pct, anchor in _HFO_CHANGES:
        sentence = _find_sentence(narrative_records, phrase)
        if sentence is None or anchor not in known:
            log.warning("HFO %d estimate skipped: %s", year, "phrase not found" if sentence is None
                        else f"no {anchor} value")
            continue
        base, base_note = known[anchor]
        factor = 1 + pct / 100
        value = base * factor if anchor < year else base / factor
        how = (f"{anchor} HFO local imports {base_note} x (1 {pct:+.0f}%)" if anchor < year else
               f"{anchor} HFO local imports {base_note} / (1 {pct:+.0f}%)")
        known[year] = (value, f"{value:,.0f} L (estimate)")
        extra = {"table_caption": "", "period_type": "calendar", "market": "Local"}
        flags = ["estimate", "derived_from_pct_change"]
        if year in _HFO_SHARES:
            share_phrase, share = _HFO_SHARES[year]
            total = _latest(records, variable="petroleum_imports_total", year=year, x_market="Local")
            if _find_sentence(narrative_records, share_phrase) and total is not None:
                lo, hi = total.value * (share - 0.5) / 100, total.value * (share + 0.5) / 100
                inside = lo <= value <= hi
                extra["share_check"] = (f"stated HFO share {share:.0f}% of local imports "
                                        f"{total.value:,.0f} L -> {lo / 1e6:.1f}-{hi / 1e6:.1f} M L; "
                                        f"estimate {'inside' if inside else 'OUTSIDE'} range")
                if not inside:
                    flags.append("share_check_failed")
        out.append(Record(
            source="EWURA", url=sentence.url, doc_title=sentence.doc_title,
            pub_date=sentence.pub_date, retrieved_date=sentence.retrieved_date,
            pdf_page=sentence.pdf_page, variable="petroleum_imports_estimated", value=round(value),
            unit="litres", year=year, category="HFO",
            value_as_printed=" ".join(sentence.value_as_printed.split()),
            conversion_note=how, qa_flag=";".join(flags), extra=extra,
        ))
    return out


_NARRATIVE = re.compile(r"(\bHFO\b|heavy fuel|fuel oil|power generation|electricity generation)", re.I)


def narrative(pages: list[PageText], doc: StoredDoc, flag: str = "") -> list[Record]:
    out: list[Record] = []
    for page in pages:
        text = " ".join(page.text.split())
        for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z])", text):
            if _NARRATIVE.search(sentence) and 60 <= len(sentence) <= 1200 and "(cid:" not in sentence:
                out.append(Record(
                    source="EWURA", url=doc.ref.url, doc_title=doc.ref.title,
                    pub_date=doc.ref.pub_date, retrieved_date=doc.retrieved_date,
                    pdf_page=page.page_no, variable="explanatory_text", value=None,
                    unit="text", category="narrative", value_as_printed=sentence,
                    qa_flag=flag or ("text_glyph_decoded" if page.cid_decoded else ""),
                ))
    return out


def parse_pages(pages: list[PageText], doc: StoredDoc) -> list[Record]:
    mk = _Mk(doc)
    records: list[Record] = []
    for block in find_tables(pages, _IMPORTS_CAPTION):
        records += _imports(block, mk)
    for block in find_tables(pages, _CONSUMPTION_CAPTION):
        records += _consumption(block, mk)
    records += appendix_totals(pages, doc)
    return _dedupe(records)


def _repair_rotated(pages: list[PageText], path) -> list[PageText]:
    """pdfplumber reads sideways tables back to front ("latoT"); pdfium reads them in order."""
    bad = {p.page_no for p in pages if "latoT" in p.text}
    if not bad:
        return pages
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    return [PageText(p.page_no, pdf[p.page_no - 1].get_textpage().get_text_range(), False)
            if p.page_no in bad else p for p in pages]


def _ocr_scanned(path) -> list[PageText]:
    """OCR a scanned report; sideways appendix tables are OCR'd again after rotating."""
    from parsers.ocr import ocr_page_rotated, ocr_pages

    one_line_header = re.compile(r"\bMonth\b.*\bMonth\b")
    out = []
    for page in ocr_pages(path, max_pages=500):
        if (re.search(r"APPENDIX", page.text, re.I) and re.search(r"IMPORT", page.text, re.I)
                and not one_line_header.search(page.text)):
            turned = ocr_page_rotated(path, page.page_no, 90)
            if one_line_header.search(turned.text):
                page = turned
        out.append(page)
    return out


def _dedupe(records: list[Record]) -> list[Record]:
    """A table caption repeated on one page yields duplicate rows; keep the first."""
    seen: set[tuple] = set()
    out = []
    for r in records:
        key = (r.pdf_page, r.variable, r.category, r.fy, r.year, r.extra.get("market"),
               r.value_as_printed)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def parse(doc: StoredDoc) -> dict[str, list[Record]]:
    pages = read_pages(doc.path)
    scanned = not any(p.text.strip() for p in pages)
    pages = _ocr_scanned(doc.path) if scanned else _repair_rotated(pages, doc.path)
    records = parse_pages(pages, doc)
    if scanned:
        for r in records:
            r.qa_flag = ";".join(f for f in (r.qa_flag, "ocr_extracted") if f)
    log.info("%s: %d values", doc.path.name, len(records))
    return {"ewura_petroleum": records,
            "ewura_petroleum_narrative": narrative(pages, doc, "ocr_extracted" if scanned else "")}
