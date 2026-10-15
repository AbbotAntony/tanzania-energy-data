"""Parse Bank of Tanzania Monthly Economic Review statistical tables.

Extracted:
  * Table A7 "Imports of Goods": fuel and lubricants / petroleum products,
    USD million, cumulative over the 12 months ending in the stated month
    ("Year ending August")                     -> petroleum_imports_value
  * Table A1 "Selected Economic Indicators": TZS/USD exchange rate, annual
    average and end of period                    -> exchange_rate

Older editions print these tables rotated with letter-spaced text, which
pdfplumber cannot read as rows; those issues yield no records and are logged.
"""

from __future__ import annotations

import logging
import re

from parsers.pdf_text import PageText, read_pages
from parsers.text_tables import Cell, Row, TableBlock, tokenise
from pipeline.provenance import Record
from pipeline.storage import StoredDoc

log = logging.getLogger(__name__)

_IMPORTS_CAPTION = re.compile(r"^Table\s*A\s*\d+\s*(\([a-z]+\))?\s*:?\s*(Tanzania\s+)?Imports? of Goods", re.I)
_INDICATORS_CAPTION = re.compile(r"^Table\s*A\s*1\s*:?\s*Select(ed)? Economic Indicators", re.I)
_MONTHS = ("january february march april may june july august september october "
           "november december").split()
_YEAR_ENDING = re.compile(r"year\s+ending\s+(" + "|".join(_MONTHS) + r")", re.I)
_YEAR = re.compile(r"\b(20\d\d)([rp])?\b")

_IMPORT_ROWS = (
    (re.compile(r"^o/w\s+petroleum products|^oil imports", re.I), "Petroleum products"),
    (re.compile(r"^fuel and lubricants", re.I), "Fuel and lubricants"),
    (re.compile(r"^total imports", re.I), "Total goods imports (f.o.b.)"),
)


def _header_years(lines: list[str]) -> list[tuple[int, str]]:
    """Years on the header line with the most year tokens, e.g. "2022 ... 2026p"."""
    best: list[tuple[int, str]] = []
    for line in lines:
        found = [(int(y), s or "") for y, s in _YEAR.findall(line)]
        if len(found) > len(best):
            best = found
    return best


def _unit(block: TableBlock) -> tuple[str, list[str]]:
    text = " ".join([block.caption, *block.header_lines]).lower()
    if re.search(r"millions? of u(sd|ds)", text):
        flags = ["unit_label_typo_UDS"] if "uds" in text else []
        return "USD million", flags
    return "unknown", ["unit_not_found"]


def _rec(doc: StoredDoc, block: TableBlock, variable: str, cell: Cell, year: int,
         month: int | None, unit: str, category: str, flags: list[str], **extra: object) -> Record:
    return Record(
        source="Bank of Tanzania", url=doc.ref.url, doc_title=doc.ref.title,
        pub_date=doc.ref.pub_date, retrieved_date=doc.retrieved_date, pdf_page=block.page_no,
        table_no=None, variable=variable, value=cell.value, unit=unit, year=year, month=month,
        category=category, value_as_printed=cell.printed, qa_flag=";".join(flags),
        extra={"table_caption": block.caption[:200], **extra},
    )


def _imports(block: TableBlock, doc: StoredDoc) -> list[Record]:
    header_text = " ".join([block.caption, *block.header_lines])
    m = _YEAR_ENDING.search(header_text)
    years = _header_years(block.header_lines)
    if not m or not years:
        log.warning("%s: imports table without 'Year ending' header", doc.path.name)
        return []
    month = _MONTHS.index(m.group(1).lower()) + 1
    unit, uflags = _unit(block)
    out = []
    for row in block.rows:
        cat = next((name for rx, name in _IMPORT_ROWS if rx.search(row.label)), None)
        values = [c for c in row.cells if not c.is_pct]
        if cat is None or len(values) < len(years):
            continue
        for (year, status), cell in zip(years, values):
            flags = uflags + (["provisional"] if status == "p" else ["revised"] if status == "r" else [])
            out.append(_rec(doc, block, "imports_value_12m_cumulative", cell, year, month, unit,
                            cat, flags, period_label=f"year ending {m.group(1).title()} {year}"))
    return out


def _exchange_rates(block: TableBlock, doc: StoredDoc) -> list[Record]:
    years = _header_years(block.header_lines + block.body[:3])
    out: list[Record] = []
    in_fx = False
    for line in block.body:
        low = line.lower()
        if "exchange rate" in low:
            in_fx = True
            continue
        if not in_fx:
            continue
        kind = ("Annual average" if low.startswith("annual average") else
                "End of period" if low.startswith("end of period") else None)
        if kind is None:
            if re.match(r"^\s*\d+\.\d+\s", line):  # next numbered indicator: section over
                break
            continue
        _, cells = tokenise(line)
        values = [c for c in cells if not c.is_pct]
        for (year, status), cell in zip(years, values[-len(years):] if years else []):
            flags = ["provisional"] if status == "p" else ["revised"] if status == "r" else []
            out.append(_rec(doc, block, "exchange_rate", cell, year, None, "TZS/USD",
                            kind, flags, period_label=str(year)))
    return out


def raw_block(pages: list[PageText], caption_re: re.Pattern[str], n: int) -> TableBlock | None:
    """Caption plus the next ``n`` raw lines.

    MER tables number their rows ("1.1 Annual change ..."), which the generic
    table finder treats as section headings, so raw lines are used here.
    """
    for page in pages:
        lines = page.text.splitlines()
        for i, line in enumerate(lines):
            if caption_re.search(line.strip()) and not re.search(r"\.{5,}", line):
                body = lines[i + 1:i + 1 + n]
                rows = [Row(*tokenise(b), b) for b in body]
                return TableBlock(line.strip(), page.page_no, None, body[:4], rows,
                                  page.cid_decoded, [], body)
    return None


def parse_pages(pages: list[PageText], doc: StoredDoc) -> list[Record]:
    records: list[Record] = []
    block = raw_block(pages, _IMPORTS_CAPTION, 45)
    if block:
        records += _imports(block, doc)
    block = raw_block(pages, _INDICATORS_CAPTION, 50)
    if block:
        records += _exchange_rates(block, doc)
    if not records:
        log.warning("no MER tables recognised in %s (%s)", doc.path.name, doc.ref.title)
    return records


def parse(doc: StoredDoc) -> dict[str, list[Record]]:
    records = parse_pages(read_pages(doc.path), doc)
    return {
        "bot_trade": [r for r in records if r.variable.startswith("imports_")],
        "bot_exchange_rate": [r for r in records if r.variable == "exchange_rate"],
    }
