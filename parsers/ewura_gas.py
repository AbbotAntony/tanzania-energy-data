"""Parse EWURA Natural Gas Sub-Sector Regulatory Performance Reports.

Tables extracted (names vary by edition; matched by caption):
  * supply by processing plant, multi-year   -> gas_supply_by_plant (MMscf)
  * production by gas field, report year      -> gas_produced_by_field (MMscf)
  * consumption by customer category x seller -> gas_consumption (MMscf, %)
  * consumption by category, multi-year       -> gas_consumption (MMscf, %)
  * transmission pipeline capacity/utilisation -> pipeline_* (MMscfd, %)
  * processing plant capacity utilisation     -> plant_* (MMscfd, %)
  * customers per category                    -> customers (count)
  * narrative sentences explaining changes    -> separate "narrative" records

Every value keeps its published unit. Internal totals are checked and any
mismatch is written to ``qa_flag`` (never corrected).
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from parsers.pdf_text import PageText, fy_label, read_pages
from parsers.text_tables import Cell, Row, TableBlock, find_tables, unique_years
from pipeline.provenance import Record
from pipeline.storage import StoredDoc

log = logging.getLogger(__name__)

TOTAL_TOLERANCE = 0.005  # 0.5 % relative difference before a total is flagged


# ---------------------------------------------------------------- canonical labels
def canon_plant(label: str) -> str | None:
    s = label.lower()
    if "total" in s:
        return "TOTAL"
    if "madimba" in s:
        return "TPDC Madimba"
    if "songas" in s or "paet" in s:
        return "Songas"
    if "songo" in s:
        return "TPDC Songo Songo"
    if "maurel" in s or "m&p" in s:
        return "Maurel & Prom (M&P)"
    return None


def canon_field(label: str) -> str | None:
    s = label.lower()
    if "total" in s:
        return "TOTAL"
    if "mnazi" in s:
        return "Mnazi Bay Field"
    if "songo" in s:
        return "Songo Songo Field"
    return None


def canon_category(label: str) -> str | None:
    s = label.lower()
    if "total" in s:
        return "TOTAL"
    if "power" in s:
        return "Power"
    if "industr" in s:
        return "Industries"
    if "cng" in s:
        return "CNG Vehicles"
    if "household" in s and "commercial" in s:
        return "Households, Commercial and Institutions"
    if "household" in s and "institution" in s:
        return "Households and Institutions"
    for key, name in (("household", "Households"), ("commercial", "Commercial"),
                      ("institution", "Institutions")):
        if key in s:
            return name
    return None


def canon_operator(label: str) -> str | None:
    s = label.lower()
    if "tpdc" in s:
        return "TPDC"
    if "songas" in s:
        return "Songas"
    if "m&p" in s or "maurel" in s:
        return "M&P"
    return None


# ---------------------------------------------------------------- helpers
def report_fy(title: str) -> str | None:
    """FY covered by a report: "...Year Ended 30th June 2024" -> "2023/24"."""
    m = re.search(r"june\s*,?\s*(20\d\d)", title, re.I)
    if m:
        return fy_label(int(m.group(1)) - 1)
    m = re.search(r"(20\d\d)\s*[-/]\s*(?:20)?(\d\d)", title)
    return fy_label(int(m.group(1))) if m else None


def fy_end_year(fy: str | None) -> int | None:
    return int(fy[:4]) + 1 if fy else None


class _Builder:
    """Creates Records that share one document's provenance."""

    def __init__(self, doc: StoredDoc, fy: str | None) -> None:
        self.doc = doc
        self.fy = fy

    def record(self, block: TableBlock, variable: str, cell: Cell | None, unit: str,
               category: str, fy: str | None, flags: list[str] | None = None,
               **extra: object) -> Record | None:
        if cell is None:
            return None
        flags = list(flags or [])
        if block.cid_decoded:
            flags.append("text_glyph_decoded")
        return Record(
            source="EWURA", url=self.doc.ref.url, doc_title=self.doc.ref.title,
            pub_date=self.doc.ref.pub_date, retrieved_date=self.doc.retrieved_date,
            pdf_page=block.page_no, table_no=block.table_no, variable=variable,
            value=cell.value, unit=unit, fy=fy, year=fy_end_year(fy), category=category,
            value_as_printed=cell.printed, qa_flag=";".join(flags),
            extra={"table_caption": block.caption[:200], "report_fy": self.fy, **extra},
        )


def _values(row: Row) -> list[Cell]:
    return [c for c in row.cells if not c.is_pct]


def _pct(row: Row) -> list[Cell]:
    return [c for c in row.cells if c.is_pct]


def _align(cells: list[Cell], n: int) -> tuple[list[Cell | None], list[str]]:
    """Take ``n`` cells; if fewer exist, right-align and flag the row."""
    if len(cells) >= n:
        return list(cells[:n]), []
    pad: list[Cell | None] = [None] * (n - len(cells))
    return pad + list(cells), ["short_row_right_aligned"]


def _pairs(cells: list[Cell], n: int) -> list[tuple[Cell | None, Cell | None]]:
    """Walk (volume, percent) pairs; a percent where a volume is due means a blank volume."""
    out: list[tuple[Cell | None, Cell | None]] = []
    it = iter(cells)
    pending: Cell | None = next(it, None)
    for _ in range(n):
        vol: Cell | None = None
        if pending is not None and not pending.is_pct:
            vol, pending = pending, next(it, None)
        pct: Cell | None = None
        if pending is not None and pending.is_pct:
            pct, pending = pending, next(it, None)
        out.append((vol, pct))
    return out


# ---------------------------------------------------------------- table parsers
def _supply_by_plant(b: TableBlock, mk: _Builder) -> list[Record]:
    years = unique_years(b.years)
    if not years:
        return []
    measure = ("processed" if "process" in b.caption.lower() else
               "produced" if "production" in b.caption.lower() else "supplied (exported)")
    out = []
    for row in b.rows:
        plant = canon_plant(row.label)
        if plant is None:
            continue
        cells, flags = _align(_values(row), len(years))
        for fy, cell in zip(years, cells):
            out.append(mk.record(b, "gas_supply_by_plant", cell, "MMscf", plant, fy, flags,
                                 measure_as_labelled=measure))
    return out


def _produced_by_field(b: TableBlock, mk: _Builder) -> list[Record]:
    out = []
    for row in b.rows:
        field = canon_field(row.label)
        vals = _values(row)
        if field and vals:
            out.append(mk.record(b, "gas_produced_by_field", vals[0], "MMscf", field, mk.fy))
    return out


_SELLERS = ("TPDC", "PAET", "M&P", "Total")


def _consumption_by_seller(b: TableBlock, mk: _Builder) -> list[Record]:
    out = []
    for row in b.rows:
        cat = canon_category(row.label)
        if cat is None:
            continue
        cells, flags = _align(_values(row), len(_SELLERS))
        for seller, cell in zip(_SELLERS, cells):
            out.append(mk.record(b, "gas_consumption", cell, "MMscf", cat, mk.fy, flags,
                                 seller=seller))
        for cell in _pct(row)[:1]:
            out.append(mk.record(b, "gas_consumption_share", cell, "%", cat, mk.fy,
                                 seller="Total"))
    return out


def _consumption_multi_year(b: TableBlock, mk: _Builder) -> list[Record]:
    years = unique_years(b.years)
    if not years:
        return []
    has_pct = any(c.is_pct for r in b.rows for c in r.cells)
    out = []
    for row in b.rows:
        cat = canon_category(row.label)
        if cat is None:
            continue
        if has_pct and cat != "TOTAL":
            for fy, (vol, pct) in zip(years, _pairs(row.cells, len(years))):
                flags = [] if vol else ["blank_in_source"]
                vol = vol or Cell(None, "", False)  # keep an explicit missing row
                out.append(mk.record(b, "gas_consumption", vol, "MMscf", cat, fy, flags,
                                     seller="Total"))
                out.append(mk.record(b, "gas_consumption_share", pct, "%", cat, fy, seller="Total"))
        else:
            cells, flags = _align(_values(row), len(years))
            for fy, cell in zip(years, cells):
                out.append(mk.record(b, "gas_consumption", cell, "MMscf", cat, fy, flags,
                                     seller="Total"))
    return out


def _pipeline(b: TableBlock, mk: _Builder) -> list[Record]:
    years = unique_years(b.years)
    out = []
    for row in b.rows:
        op = canon_operator(row.label)
        vals = _values(row)
        if op is None or len(vals) < 3:
            continue
        out.append(mk.record(b, "pipeline_installed_capacity", vals[0], "MMscfd", op, mk.fy))
        out.append(mk.record(b, "pipeline_availability", vals[1], "%", op, mk.fy))
        if len(years) >= 2:
            cells, flags = _align(vals[2:], len(years))
            for fy, cell in zip(years, cells):
                out.append(mk.record(b, "pipeline_capacity_utilisation", cell, "%", op, fy, flags))
        else:
            out.append(mk.record(b, "pipeline_capacity_utilisation", vals[2], "%", op, mk.fy))
    return out


def _plant_utilisation(b: TableBlock, mk: _Builder) -> list[Record]:
    years = unique_years(b.years, order="appearance")
    if len(years) < 2:
        return []
    out = []
    for row in b.rows:
        plant = canon_plant(row.label)
        cells = row.cells  # installed, then (MMscfd, %) per year; % sign optional
        if plant is None or plant == "TOTAL" or len(cells) < 1 + 2 * len(years):
            continue
        out.append(mk.record(b, "plant_installed_capacity", cells[0], "MMscfd", plant, mk.fy))
        for k, fy in enumerate(years):
            out.append(mk.record(b, "plant_average_throughput", cells[1 + 2 * k], "MMscfd", plant, fy))
            out.append(mk.record(b, "plant_capacity_utilisation", cells[2 + 2 * k], "%", plant, fy))
    return out


_CUSTOMER_COLS = ("Power Plants", "Industries", "Households", "Commercial",
                  "Institutions", "CNG Filling Stations")
_FY_ROW = re.compile(r"^\s*(20\d\d)\s*/\s*(\d\d)\s+((?:[\d,]+\s+){5}[\d,]+)\s*$")


def _customers(b: TableBlock, mk: _Builder, page: PageText) -> list[Record]:
    """Rows start with an FY label, e.g. "2022/23 11 54 1,514 1 8 3"."""
    out = []
    start = page.text.find(b.caption)
    for line in page.text[start:].splitlines()[1:12]:
        m = _FY_ROW.match(line)
        if not m:
            continue
        fy = fy_label(int(m.group(1)))
        for col, tok in zip(_CUSTOMER_COLS, m.group(3).split()):
            cell = Cell(float(tok.replace(",", "")), tok, False)
            out.append(mk.record(b, "customers", cell, "count", col, fy))
    return out


TableParser = Callable[[TableBlock, _Builder], list[Record]]
SPECS: tuple[tuple[re.Pattern[str], TableParser], ...] = (
    (re.compile(r"^Table\s*\d+\s*[:.]?\s*(Natural Gas Processing Plants Production"
                r"|Natural Gas Production\b|Comparison of Natural Gas (Supply|Exported))", re.I),
     _supply_by_plant),
    (re.compile(r"^Table\s*\d+\s*[:.]?\s*Produced Natural Gas in FY", re.I), _produced_by_field),
    (re.compile(r"^Table\s*\d+\s*[:.]?\s*Natural Gas Consumption by Categor", re.I),
     _consumption_by_seller),
    (re.compile(r"^(Annex\s*\d+\s*[:.]?\s*Natural Gas Consumption and Contribution"
                r"|Table\s*\d+\s*[:.]?\s*Natural Gas Consumption\s*$)", re.I),
     _consumption_multi_year),
    (re.compile(r"^Table\s*\d+\s*[:.]?\s*Transmission Pipeline Availability", re.I), _pipeline),
    (re.compile(r"^Table\s*\d+\s*[:.]?\s*Gas Processing Plants Capacity Utili[sz]ation", re.I),
     _plant_utilisation),
)
_CUSTOMER_CAPTION = re.compile(r"^Table\s*\d+\s*[:.]?\s*Natural Gas Customer Categories", re.I)


# ---------------------------------------------------------------- QA
def flag_totals(records: list[Record]) -> None:
    """Flag TOTAL rows that do not equal the sum of their component rows."""
    groups: dict[tuple, list[Record]] = {}
    for r in records:
        if r.unit in ("MMscf",) and r.value is not None:
            key = (r.pdf_page, r.table_no, r.variable, r.fy, r.extra.get("seller"))
            groups.setdefault(key, []).append(r)
    for rows in groups.values():
        totals = [r for r in rows if r.category == "TOTAL"]
        parts = [r for r in rows if r.category != "TOTAL"]
        if len(totals) == 1 and parts:
            total, s = totals[0], sum(p.value for p in parts)
            if total.value and abs(s - total.value) / abs(total.value) > TOTAL_TOLERANCE:
                total.qa_flag = _join(total.qa_flag, f"total_ne_sum_of_rows(sum={s:,.2f})")
    _flag_seller_sums(records)


def _flag_seller_sums(records: list[Record]) -> None:
    """Flag rows where TPDC + PAET + M&P != the row's Total column."""
    rows: dict[tuple, dict[str, Record]] = {}
    for r in records:
        if r.variable == "gas_consumption" and "seller" in r.extra and r.value is not None:
            key = (r.pdf_page, r.table_no, r.fy, r.category)
            rows.setdefault(key, {})[r.extra["seller"]] = r
    for by_seller in rows.values():
        total = by_seller.get("Total")
        parts = [v for k, v in by_seller.items() if k != "Total"]
        if total and len(parts) >= 2 and total.value:
            s = sum(p.value for p in parts)
            if abs(s - total.value) / abs(total.value) > TOTAL_TOLERANCE:
                total.qa_flag = _join(total.qa_flag, f"total_ne_sum_of_sellers(sum={s:,.2f})")


def _join(existing: str, flag: str) -> str:
    return f"{existing};{flag}" if existing else flag


# ---------------------------------------------------------------- narrative
_EXPLAIN = re.compile(r"(Julius Nyerere|JNHPP|hydro|Power Purchase Agreement|\bPPA\b|"
                      r"decreas|declin|reduc|drop)", re.I)


def narrative(pages: list[PageText], mk: _Builder) -> list[Record]:
    """Sentences that explain changes in gas production/consumption."""
    out: list[Record] = []
    for page in pages:
        text = " ".join(page.text.split())
        for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z])", text):
            if (_EXPLAIN.search(sentence) and re.search(r"\bgas\b", sentence, re.I)
                    and 60 <= len(sentence) <= 1200 and "(cid:" not in sentence):
                out.append(Record(
                    source="EWURA", url=mk.doc.ref.url, doc_title=mk.doc.ref.title,
                    pub_date=mk.doc.ref.pub_date, retrieved_date=mk.doc.retrieved_date,
                    pdf_page=page.page_no, variable="explanatory_text", value=None, unit="text",
                    fy=mk.fy, year=fy_end_year(mk.fy), category="narrative",
                    value_as_printed=sentence,
                    qa_flag="text_glyph_decoded" if page.cid_decoded else "",
                ))
    return out


# ---------------------------------------------------------------- entry points
def parse_pages(pages: list[PageText], doc: StoredDoc) -> list[Record]:
    """Parse already-extracted pages of one gas report (testable without a PDF)."""
    mk = _Builder(doc, report_fy(doc.ref.title))
    records: list[Record] = []
    for caption_re, parser in SPECS:
        for block in find_tables(pages, caption_re):
            records += [r for r in parser(block, mk) if r is not None]
    for block in find_tables(pages, _CUSTOMER_CAPTION, max_lines=12):
        page = next(p for p in pages if p.page_no == block.page_no)
        records += [r for r in _customers(block, mk, page) if r is not None]
    flag_totals(records)
    log.info("%s: %d values", doc.path.name, len(records))
    return records


def parse(doc: StoredDoc) -> dict[str, list[Record]]:
    """Parse one gas report PDF into numeric and narrative datasets."""
    pages = read_pages(doc.path)
    return {
        "ewura_gas": parse_pages(pages, doc),
        "ewura_gas_narrative": narrative(pages, _Builder(doc, report_fy(doc.ref.title))),
    }
