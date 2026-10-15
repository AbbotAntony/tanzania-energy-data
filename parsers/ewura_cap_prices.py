"""Parse EWURA monthly cap-price public notices.

Each notice has "Table 1: Retail Prices - TZS/Litre" and "Table 2: Wholesale
Prices - TZS/Litre" with ports as rows (Dar es Salaam, Tanga, Mtwara) and
products as columns (Petrol, Diesel, Kerosene). Prices apply from the
effective date in the notice title, normally the first Wednesday of the month.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from parsers.ewura_petroleum import canon_product
from parsers.pdf_text import PageText, read_pages
from parsers.text_tables import TableBlock, find_tables
from pipeline.provenance import Record
from pipeline.storage import StoredDoc

log = logging.getLogger(__name__)

_PREFIX = r"^(Table\s*[A-Z0-9]{1,2}|Jedwali(\s*Na\.?)?\s*\d+|[A-Z])\s*[:.]\s*"
_CAPTIONS = {
    # "Table 1: Retail Prices - TZS/Litre" (2022+), "TABLE A: RETAIL PRICES" (2021),
    # "A: RETAIL PRICES" (2017), "TABLE 4: RETAIL CAP PRICES" (late 2022),
    # "Jedwali 1: Bei Kikomo za Rejareja" (Swahili notices)
    "retail": re.compile(_PREFIX + r"(Retail\s+(Cap\s+)?Prices|Bei Kikomo za Rejareja)", re.I),
    "wholesale": re.compile(_PREFIX + r"(Wholesale\s+(Cap\s+)?Prices|Bei Kikomo za Jumla)", re.I),
}
_PORTS = {"dar": "Dar es Salaam", "tanga": "Tanga", "mtwara": "Mtwara"}
_MONTHS: dict[str, int] = {}
for _i, (_en, _sw) in enumerate(zip(
        "january february march april may june july august september october november december".split(),
        "januari februari machi aprili mei juni julai agosti septemba oktoba novemba desemba".split()), 1):
    _MONTHS[_en] = _MONTHS[_sw] = _i
_DATE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(" + "|".join(_MONTHS) +
                   r")\s*,?\s*(20\d\d)", re.I)


def effective_date(text: str) -> str | None:
    """Find the first "6th May 2026" / "01 Januari 2025" date and return ISO format."""
    m = _DATE.search(text)
    if not m:
        return None
    month = _MONTHS[m.group(2).lower()]
    return datetime(int(m.group(3)), month, int(m.group(1))).date().isoformat()


def _unit(block: TableBlock) -> str:
    text = " ".join([block.caption, *block.header_lines])
    m = re.search(r"(TZS|TSh[s]?)\s*/\s*(Litre|Ltr|L)\b", text, re.I)
    return "TZS/litre" if m else "TZS/litre (assumed; unit not in caption)"


def price_columns(header: str) -> list[tuple[str, str]]:
    """Return (product, price_basis) per column.

    In 2022 the Government subsidised fuel and notices printed each product
    twice: "Price without subsidy" then "Price with subsidy". A repeated
    product is labelled accordingly; otherwise the basis is "cap price".
    """
    products = [p for p in (canon_product(t) for t in header.split()) if p]
    subsidised = "subsidy" in header.lower()
    columns, seen = [], {}
    for p in products:
        seen[p] = seen.get(p, 0) + 1
        repeated = products.count(p) > 1
        if subsidised and repeated:
            basis = "cap price without subsidy" if seen[p] == 1 else "cap price with subsidy"
        else:
            basis = "cap price"
        columns.append((p, basis))
    return columns


_TABLE_END = re.compile(r"DIRECTOR GENERAL|SERVICE LEV|^Eng\.", re.I)
_FLOOR = re.compile(r"^\s*FLOOR PRICES?\b", re.I)


def _rows_with_basis(block: TableBlock):
    """Yield (row, is_floor_price) in order, stopping at the signature/levy tables.

    Some 2022 notices print "FLOOR PRICES" rows under the wholesale cap prices.
    """
    by_line = {row.line: row for row in block.rows}
    floor = False
    for line in block.body:
        if _TABLE_END.search(line):
            return
        if _FLOOR.search(line):
            floor = True
        if line in by_line:
            yield by_line[line], floor


def _parse_block(block: TableBlock, kind: str, doc: StoredDoc, eff: str | None) -> list[Record]:
    columns = price_columns(" ".join(block.header_lines))
    out: list[Record] = []
    for row, floor in _rows_with_basis(block):
        port = next((name for key, name in _PORTS.items() if key in row.label.lower()), None)
        if port is None or not columns:
            continue
        if floor:
            columns = [(p, "floor price") for p, _ in columns]
        # Fewer cells than columns happens when a port has no kerosene (last column).
        flags = [] if len(row.cells) == len(columns) else ["fewer_cells_than_columns_left_aligned"]
        if block.cid_decoded:
            flags.append("text_glyph_decoded")
        for (product, basis), cell in zip(columns, row.cells):
            out.append(Record(
                source="EWURA", url=doc.ref.url, doc_title=doc.ref.title,
                pub_date=doc.ref.pub_date, retrieved_date=doc.retrieved_date,
                pdf_page=block.page_no, table_no=block.table_no,
                variable=f"cap_price_{kind}", value=cell.value, unit=_unit(block),
                year=int(eff[:4]) if eff else None, month=int(eff[5:7]) if eff else None,
                category=product, value_as_printed=cell.printed, qa_flag=";".join(flags),
                extra={"port": port, "price_basis": basis, "effective_date": eff},
            ))
    return out


def parse_pages(pages: list[PageText], doc: StoredDoc) -> list[Record]:
    first = pages[0].text if pages else ""
    eff = effective_date(first) or effective_date(doc.ref.title)
    records: list[Record] = []
    for kind, caption in _CAPTIONS.items():
        blocks = find_tables(pages, caption, max_lines=80)  # 2021 Table A lists every town
        if blocks:
            records += _parse_block(blocks[0], kind, doc, eff)
    if not records:
        log.warning("no cap-price tables recognised in %s", doc.path.name)
    return records


def parse(doc: StoredDoc) -> dict[str, list[Record]]:
    return {"ewura_cap_prices": parse_pages(read_pages(doc.path), doc)}
