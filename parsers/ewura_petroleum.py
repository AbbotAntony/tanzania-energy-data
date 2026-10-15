"""Parse EWURA (Mid and) Downstream Petroleum Sub-Sector Performance Reports.

Extracted (units as published, normally litres):
  * total local / transit imports per period       -> petroleum_imports_total
  * local / transit imports by product per period  -> petroleum_imports (HFO, diesel...)
  * consumption by product per period              -> petroleum_consumption
  * sentences about HFO / power-generation fuel mix -> narrative records

Editions before 2019 present product splits only as charts, so product-level
imports exist from calendar 2019 onward; earlier years yield totals only.
"""

from __future__ import annotations

import logging
import re

from parsers.ewura_gas import fy_end_year
from parsers.pdf_text import PageText, fiscal_years_in, read_pages
from parsers.text_tables import Cell, TableBlock, find_tables, tokenise
from pipeline.provenance import Record
from pipeline.storage import StoredDoc

log = logging.getLogger(__name__)

_IMPORTS_CAPTION = re.compile(r"^Table\s*\d+\s*[:.]?.*Local and Transit.*Imports", re.I)
_CONSUMPTION_CAPTION = re.compile(r"^Table\s*\d+\s*[:.]?\s*Petroleum Products? Consumption in", re.I)
_YEAR_ROW = re.compile(r"^\s*(?:Year\s+|FY\s*)?(20\d\d)(?:\s*/\s*(\d\d))?\s+([\d,.\s%()-]*\d[\d,.\s%()-]*)$")
_CAL_YEAR = re.compile(r"\b(20\d\d)\b(?!\s*/)")


def canon_product(label: str) -> str | None:
    s = label.lower().replace(" ", "")
    if "total" in s:
        return "TOTAL"
    if "jet" in s and "kerosene" in s:
        return "Jet A-1/Kerosene"
    if "jet" in s:
        return "Jet A-1"
    if "hfo" in s or "heavyfuel" in s or "fueloil" in s:
        return "HFO"
    if "diesel" in s or "dizeli" in s or "ago" in s or "gasoil" in s:
        return "Diesel (AGO)"
    if "petrol" in s or "pms" in s or "mogas" in s:  # also Swahili "Petroli"
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
        m = _YEAR_ROW.match(line)
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


_NARRATIVE = re.compile(r"(\bHFO\b|heavy fuel|fuel oil|power generation|electricity generation)", re.I)


def narrative(pages: list[PageText], doc: StoredDoc) -> list[Record]:
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
                    qa_flag="text_glyph_decoded" if page.cid_decoded else "",
                ))
    return out


def parse_pages(pages: list[PageText], doc: StoredDoc) -> list[Record]:
    mk = _Mk(doc)
    records: list[Record] = []
    for block in find_tables(pages, _IMPORTS_CAPTION):
        records += _imports(block, mk)
    for block in find_tables(pages, _CONSUMPTION_CAPTION):
        records += _consumption(block, mk)
    return _dedupe(records)


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
    records = parse_pages(pages, doc)
    log.info("%s: %d values", doc.path.name, len(records))
    return {"ewura_petroleum": records, "ewura_petroleum_narrative": narrative(pages, doc)}
