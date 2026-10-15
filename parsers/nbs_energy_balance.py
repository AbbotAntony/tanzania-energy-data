"""Parse the NBS 2022 Energy Balance workbook (Excel).

Sheets: T1/T3 aggregated and T2/T4 detailed balances for Mainland Tanzania (T1,
T2) and the United Republic of Tanzania (T3, T4). Each sheet holds one or more
blocks; a block starts with a header row whose first cell names the unit
("Thousand Tonnes of Oil Equivalent (ktoe)", "Terajoules (TJ)", "GWh") and
whose other cells name products; each following row is a flow (Production,
Imports, ..., electricity transformation, final consumption by sector).

Every cell becomes one record (flow x product x unit). The workbook itself
supplies the TJ and GWh blocks, so no conversion is done here. Excel errors
such as "#VALUE!" become empty values with a flag. Flows describing power
generation are marked so gas, oil and fuel oil used for electricity can be
filtered directly.
"""

from __future__ import annotations

import logging
import re

from pipeline.provenance import Record
from pipeline.storage import StoredDoc

log = logging.getLogger(__name__)

SHEETS = {
    "T1": ("Mainland Tanzania", "aggregated"),
    "T2": ("Mainland Tanzania", "detailed"),
    "T3": ("United Republic of Tanzania", "aggregated"),
    "T4": ("United Republic of Tanzania", "detailed"),
}
YEAR = 2022
_UNITS = ((re.compile(r"ktoe|oil equivalent", re.I), "ktoe"),
          (re.compile(r"terajoule|\bTJ\b", re.I), "TJ"),
          (re.compile(r"\bGWh\b", re.I), "GWh"))
POWER_FLOWS = re.compile(r"^(electricity plants|main activity producer electricity|"
                         r"autoproducer electricity)", re.I)


def block_unit(label: object) -> str | None:
    text = str(label or "")
    return next((unit for rx, unit in _UNITS if rx.search(text)), None)


def clean_label(label: object) -> str:
    return " ".join(str(label).split())


def parse_sheet(rows: list[tuple], sheet: str, doc: StoredDoc) -> list[Record]:
    geography, detail = SHEETS[sheet]
    out: list[Record] = []
    unit: str | None = None
    products: list[tuple[int, str]] = []
    for row_no, row in enumerate(rows, 1):
        label = row[1] if len(row) > 1 else None
        if label is None:
            continue
        new_unit = block_unit(label)
        if new_unit:  # header row of a new block
            unit = new_unit
            products = [(j, clean_label(c)) for j, c in enumerate(row) if j > 1 and c is not None]
            continue
        if unit is None or not products:
            continue
        flow = clean_label(label)
        for j, product in products:
            cell = row[j] if j < len(row) else None
            value, flags = _value(cell)
            if value is None and not flags:
                continue
            out.append(Record(
                source="NBS", url=doc.ref.url, doc_title=doc.ref.title, pub_date=doc.ref.pub_date,
                retrieved_date=doc.retrieved_date, pdf_page=None, table_no=None,
                variable="energy_balance", value=value, unit=unit, year=YEAR,
                category=product, value_as_printed=str(cell), qa_flag=";".join(flags),
                extra={"flow": flow, "geography": geography, "table": f"{sheet} ({detail})",
                       "sheet_row": row_no, "power_generation_flow": bool(POWER_FLOWS.search(flow))},
            ))
    return out


def _value(cell: object) -> tuple[float | None, list[str]]:
    if isinstance(cell, bool) or cell is None:
        return None, []
    if isinstance(cell, (int, float)):
        return float(cell), []
    text = str(cell).strip()
    if text.startswith("#"):
        return None, [f"excel_error_{text}"]
    return None, []


def parse(doc: StoredDoc) -> list[Record]:
    import openpyxl

    wb = openpyxl.load_workbook(doc.path, data_only=True, read_only=True)
    records: list[Record] = []
    for sheet in SHEETS:
        if sheet in wb.sheetnames:
            records += parse_sheet(list(wb[sheet].iter_rows(values_only=True)), sheet, doc)
        else:
            log.warning("%s: sheet %s missing", doc.path.name, sheet)
    log.info("%s: %d energy-balance cells", doc.path.name, len(records))
    return records
