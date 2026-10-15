"""Parse Bank of Tanzania Consolidated Zonal Economic Reports (quarterly).

Energy tables (captions and numbering change between editions):
  * natural gas production (by field, or South Eastern zone total) and
    consumption by sector (from about 2023)            -> gas_production / gas_consumption
  * electricity generation by zone                       -> electricity_generation_by_zone
  * electricity generation by source (from about 2021):
    TANESCO / private plants x hydro / thermal / solar   -> electricity_generation_by_source

Columns are periods: quarters ("Mar-26", quarter ending) or fiscal years
("2024/25", June editions). Units are kept as printed; mislabelled energy
units such as "Megawatts" are flagged, never converted. The data originate
from TPDC (gas) and TANESCO (electricity), as stated under each table.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from parsers.ewura_gas import fy_end_year
from parsers.pdf_text import PageText, read_pages
from parsers.text_tables import Cell, TableBlock, find_tables, tokenise
from pipeline.provenance import Record
from pipeline.storage import StoredDoc

log = logging.getLogger(__name__)

_MON = "jan feb mar apr may jun jul aug sep oct nov dec".split()
_PERIOD_TOKEN = re.compile(
    r"\b(?:(?P<mon>" + "|".join(_MON) + r")[a-z]*\s*-\s*(?P<yy>\d{2})"
    r"|(?P<fy>20\d\d/\d\d))(?P<status>[rp])?\b", re.I)

_GAS_CAPTION = re.compile(r"^Table\s*[\d.]+[a-z]?\s*:?\s*(Natural Gas Production"
                          r"|(Production of )?Electricity and (Natural )?Gas"
                          r"|Production of Electricity and (Natural )?Gas)", re.I)
_ZONE_CAPTION = re.compile(r"^Table\s*[\d.]+[a-z]?\s*:?\s*Electricity Generation by Zone", re.I)
_SOURCE_CAPTION = re.compile(r"^Table\s*[\d.]+[a-z]?\s*:?\s*Electricity Generation by Source", re.I)

_ZONES = ("Central", "Dar es Salaam", "Lake", "Northern", "South Eastern",
          "Southern Highlands", "Eastern")


@dataclass(frozen=True)
class Period:
    label: str
    year: int
    month: int | None
    fy: str | None
    status: str  # "", "revised", "provisional"

    @property
    def order(self) -> tuple[int, int]:
        return (self.year, self.month or 6)


def parse_period(m: re.Match[str]) -> Period:
    status = {"r": "revised", "p": "provisional"}.get((m.group("status") or "").lower(), "")
    if m.group("fy"):
        fy = m.group("fy")
        return Period(fy, fy_end_year(fy), None, fy, status)
    month = _MON.index(m.group("mon").lower()[:3]) + 1
    return Period(f"{m.group('mon')[:3].title()}-{m.group('yy')}", 2000 + int(m.group("yy")),
                  month, None, status)


def column_periods(lines: list[str]) -> list[Period]:
    """Longest strictly increasing run of period labels on any header line.

    Header lines also carry percentage-change labels ("Mar-25 to Mar-26"), which
    repeat earlier periods; the increasing-run rule stops before them.
    """
    best: list[Period] = []
    for line in lines:
        run: list[Period] = []
        for m in _PERIOD_TOKEN.finditer(line):
            p = parse_period(m)
            if run and p.order <= run[-1].order:
                break
            run.append(p)
        if len(run) > len(best):
            best = run
    return best


# ---------------------------------------------------------------- text repair
_GLUED = re.compile(r"(\d,\d{3}\.\d)(?=\d{1,3}(?:,\d{3})+\.\d)")
_SPLIT_DIGIT = re.compile(r"(?<![\d,.])(\d) (\d,\d{3}\.\d)\b")


def repair_numbers(text: str) -> tuple[str, bool]:
    """Fix two extraction faults seen in BoT PDFs (one-decimal tables):
    "423,596.11,313,257.1" -> "423,596.1 1,313,257.1" and "3 4,770.7" -> "34,770.7".
    """
    fixed = _SPLIT_DIGIT.sub(r"\1\2", _GLUED.sub(r"\1 ", text))
    return fixed, fixed != text


# ---------------------------------------------------------------- units
def unit_from(text: str, kind: str) -> tuple[str, list[str]]:
    """Return (unit as published, flags) for 'gas' or 'electricity'."""
    t = text.lower()
    if kind == "gas":
        if re.search(r"million (standard|cubic) (cubic|standard) feet|mmscf|(mn|mill)\.?\s*std\.?\s*cubic\s*f", t):
            return "MMscf", []
        if re.search(r"\bmscf\b", t):  # printed "MSCF" (thousand scf) for MMscf-sized values
            return "MSCF (as printed)", ["unit_label_suspect_MSCF_for_MMscf"]
        return "unknown", ["unit_not_found"]
    if re.search(r"000\s*mwh", t):
        return "000 MWh", []  # 2016-17 editions print thousands of MWh
    if re.search(r"megawatts? hours?|\bmwh\b", t):
        return "MWh", []
    if re.search(r"000'?\s*kwh", t):
        return "000 kWh", []  # numerically equal to MWh; not converted
    if re.search(r"\bkwh\b", t):
        return "kWh (as printed)", ["unit_label_suspect"]
    if "megawatts" in t:
        return "Megawatts (as printed)", ["unit_label_suspect_power_unit_for_energy"]
    return "unknown", ["unit_not_found"]


# ---------------------------------------------------------------- records
class _Mk:
    def __init__(self, doc: StoredDoc, repaired_lines: set[str]) -> None:
        self.doc = doc
        self.repaired = repaired_lines

    def rec(self, block: TableBlock, variable: str, cell: Cell, period: Period, unit: str,
            category: str, line: str, flags: list[str] | None = None, **extra: object) -> Record:
        flags = list(flags or [])
        if line in self.repaired:
            flags.append("numbers_repaired_from_extraction")
        if period.status:
            flags.append(period.status)
        return Record(
            source="Bank of Tanzania", url=self.doc.ref.url, doc_title=self.doc.ref.title,
            pub_date=self.doc.ref.pub_date, retrieved_date=self.doc.retrieved_date,
            pdf_page=block.page_no, table_no=None, variable=variable, value=cell.value,
            unit=unit, year=period.year, month=period.month, fy=period.fy, category=category,
            value_as_printed=cell.printed, qa_flag=";".join(flags),
            extra={"table_caption": block.caption[:200], "period_label": period.label,
                   "period_type": "fiscal_year" if period.fy else "quarter_ending",
                   "original_source": _original_source(block), **extra},
        )


_SOURCES: dict[tuple[int, str], str] = {}


def _original_source(block: TableBlock) -> str:
    return _SOURCES.get((block.page_no, block.caption), "")


def _register_sources(pages: list[PageText], blocks: list[TableBlock]) -> None:
    """Remember the "Source: ..." line printed below each table (the original data owner)."""
    by_page = {p.page_no: p.text.splitlines() for p in pages}
    for b in blocks:
        lines = by_page.get(b.page_no, [])
        start = next((i for i, l in enumerate(lines) if l.strip() == b.caption), 0)
        src = next((l for l in lines[start:] if l.strip().lower().startswith("source")), "")
        _SOURCES[(b.page_no, b.caption)] = src.split(":", 1)[-1].strip()[:150]


_ROW_UNITS = re.compile(r"\b000'?\s*KWh\b|\b(Mn|Mill)\.?\s*std\.?\s*cubic\s*f(ee)?t\b|\bMWh\b", re.I)


def _values(line: str, n: int) -> list[Cell]:
    """First ``n`` non-percentage numbers (2016 rows print a unit, e.g. "000' KWh", first).

    A row with fewer numbers than period columns has blank cells whose position
    cannot be inferred from text (e.g. imports from a new partner country), so
    it is skipped and logged rather than guessed.
    """
    _, cells = tokenise(_ROW_UNITS.sub(" ", line))
    values = [c for c in cells if not c.is_pct]
    if len(values) < n:
        log.warning("row skipped, fewer values than %d period columns: %r", n, line.strip())
        return []
    return values[:n]


def _label(line: str) -> str:
    label, _ = tokenise(_ROW_UNITS.sub(" ", line))
    return label.replace("---", "").strip()


# ---------------------------------------------------------------- gas
_GAS_ROWS = (
    (re.compile(r"songo\s*songo", re.I), "production", "Songo Songo"),
    (re.compile(r"^songas\b", re.I), "production", "Songo Songo"),  # 2016-17 label
    (re.compile(r"mnazi|manzi", re.I), "production", "Mnazi Bay"),  # "Manzi" typo in Dec-16
    (re.compile(r"total production", re.I), "production", "TOTAL"),
    (re.compile(r"power", re.I), "consumption", "Power generating plants"),
    (re.compile(r"industr", re.I), "consumption", "Industries"),
    (re.compile(r"vehicle", re.I), "consumption", "Vehicles"),
    (re.compile(r"household", re.I), "consumption", "Households"),
    (re.compile(r"^others?\b", re.I), "consumption", "Others"),
    (re.compile(r"total consumption", re.I), "consumption", "TOTAL"),
)


def _gas(block: TableBlock, mk: _Mk, periods: list[Period]) -> list[Record]:
    """Gas rows; older editions give only a zone total (South Eastern / Eastern)."""
    out: list[Record] = []
    unit, uflags = unit_from(" ".join(block.body), "gas")
    in_gas = not re.search(r"electricity", block.caption, re.I)
    for line in block.body:
        if "natural gas" in line.lower():  # start of the gas section in combined tables
            in_gas = True
        if not in_gas or not re.search(r"\d", line):
            continue
        label = _label(line)
        target = next(((var, cat) for rx, var, cat in _GAS_ROWS if rx.search(label)), None)
        if target is None and label.split(" ")[0] in ("South", "Eastern"):
            target = ("production", "TOTAL")  # gas-producing zone total; label kept below
        if target is None:
            continue
        var, cat = target
        for period, cell in zip(periods, _values(line, len(periods))):
            out.append(mk.rec(block, f"gas_{var}", cell, period, unit, cat, line, uflags,
                              label_as_printed=label))
    return out


# ---------------------------------------------------------------- electricity by zone
_PLANT_NAMES = {"tanesco": "TANESCO", "iptl": "IPTL", "songas": "Songas", "symbion": "Symbion",
                "aggreko": "Aggreko", "nyakato": "Nyakato", "hale": "Hale (hydro)",
                "pangani": "New Pangani Falls (hydro)", "nyumba": "Nyumba ya Mungu (hydro)"}


def canon_plant(label: str) -> str:
    """One name per power plant across editions ("Tanesco"/"TANESCO", "... Power Plant")."""
    low = label.lower()
    return next((name for key, name in _PLANT_NAMES.items() if key in low), label)
def _zone_rows(block: TableBlock, mk: _Mk, periods: list[Period], stop_at_gas: bool) -> list[Record]:
    out: list[Record] = []
    unit, uflags = unit_from(" ".join([block.caption, *block.body]), "electricity")
    # Section prefixes are used only when the table prints them (2022+ editions);
    # in older editions zone rows and "Total" include imported electricity.
    section, last_zone = "", ""
    for line in block.body:
        low = line.lower().strip()
        if stop_at_gas and "natural gas" in low:
            break
        if low.startswith("imported"):
            section = "Imported"
        elif low.startswith("exported"):
            section = "Exported"
        elif low.startswith("generated domestically"):
            section = "Generated domestically"
        if not re.search(r"\d", line):
            continue
        label = _label(line)
        prefix = f"{section}: " if section else ""
        zone = next((z for z in _ZONES if label.lower().startswith(z.lower())), None)
        if zone:
            category, last_zone = f"{prefix}{zone}", zone
        elif label.lower().startswith("o/w imported"):
            category = f"{last_zone or 'Total'}: of which imported"
        elif label.lower().startswith("total"):
            category = f"{prefix}Total (as printed)" if label.lower() == "total" else label
            last_zone = "Total"
        elif section in ("Imported", "Exported") and label:
            category = f"{prefix}{label}"  # imports/exports by country
        elif label and re.search(r"\bMWh\b", line):
            # 2016-17 editions list individual plants (TANESCO, IPTL, Songas, Symbion...)
            for period, cell in zip(periods, _values(line, len(periods))):
                out.append(mk.rec(block, "electricity_generation_by_plant", cell, period, unit,
                                  canon_plant(label), line, uflags, zone=last_zone,
                                  label_as_printed=label))
            continue
        else:
            continue
        for period, cell in zip(periods, _values(line, len(periods))):
            out.append(mk.rec(block, "electricity_generation_by_zone", cell, period, unit,
                              category, line, uflags))
    return out


# ---------------------------------------------------------------- electricity by source
_BLOCK_PERIOD = re.compile(r"^\s*(?:" + _PERIOD_TOKEN.pattern + r")", re.I)


def _by_source(block: TableBlock, mk: _Mk) -> list[Record]:
    """Each period block starts with its label; the national Total is the last column."""
    out: list[Record] = []
    unit, uflags = unit_from(" ".join([block.caption, *block.body]), "electricity")
    period: Period | None = None
    owner = ""
    group: list[tuple[str, Record]] = []
    for line in block.body:
        m = _BLOCK_PERIOD.match(line)
        if m:
            out += _check_totals(group)
            period, group = parse_period(m), []
            line = line[m.end():]
        if period is None or not re.search(r"\d", line):
            continue
        label = _label(line).lower()
        if "tanesco" in label:
            owner, cat = "TANESCO plants", "TANESCO plants: all sources"
        elif "private" in label:
            owner, cat = "Private plants", "Private plants: all sources"
        elif label.split(" ")[0] in ("hydro", "thermal", "solar", "wind"):
            cat = f"{owner}: {label.split(' ')[0]}"
        elif label.startswith("total"):
            cat = "Total (as printed)"
        elif label.startswith("imported"):
            cat = "Imported"
        elif label.startswith("exported"):
            cat = "Exported"
        else:
            continue
        _, cells = tokenise(line)
        numbers = [c for c in cells if not c.is_pct]
        if not numbers:
            continue
        rec = mk.rec(block, "electricity_generation_by_source", numbers[-1], period, unit,
                     cat, line, uflags, column="national total")
        out.append(rec)
        group.append((cat, rec))
    return out + _check_totals(group)


def _check_totals(group: list[tuple[str, Record]]) -> list[Record]:
    """Label whether the printed Total includes imports (it does in some editions)."""
    vals = {cat: r.value for cat, r in group if r.value is not None}
    total = next((r for cat, r in group if cat == "Total (as printed)"), None)
    if total is None or total.value is None:
        return []
    domestic = vals.get("TANESCO plants: all sources", 0) + vals.get("Private plants: all sources", 0)
    imported = vals.get("Imported", 0)
    tol = max(1.0, 0.0005 * total.value)
    if abs(total.value - domestic) <= tol:
        total.extra["total_definition"] = "domestic generation (TANESCO + private)"
    elif abs(total.value - domestic - imported) <= tol:
        total.extra["total_definition"] = "domestic generation plus imports"
    else:
        total.qa_flag = ";".join(filter(None, [total.qa_flag, "total_ne_components"]))
    return []


# ---------------------------------------------------------------- entry points
def parse_pages(pages: list[PageText], doc: StoredDoc) -> list[Record]:
    repaired: set[str] = set()
    fixed_pages = []
    for p in pages:
        lines = []
        for line in p.text.splitlines():
            new, changed = repair_numbers(line)
            if changed:
                repaired.add(new)
            lines.append(new)
        fixed_pages.append(PageText(p.page_no, "\n".join(lines), p.cid_decoded))
    mk = _Mk(doc, repaired)
    records: list[Record] = []
    gas_blocks = _unique(find_tables(fixed_pages, _GAS_CAPTION, max_lines=40))
    zone_blocks = _unique(find_tables(fixed_pages, _ZONE_CAPTION, max_lines=40))
    source_blocks = _unique(find_tables(fixed_pages, _SOURCE_CAPTION, max_lines=60))
    _register_sources(fixed_pages, gas_blocks + zone_blocks + source_blocks)
    for block in gas_blocks:
        periods = column_periods(block.header_lines + block.body[:6])
        records += _gas(block, mk, periods)
        if re.search(r"electricity", block.caption, re.I):
            records += _zone_rows(block, mk, periods, stop_at_gas=True)
    for block in zone_blocks:
        records += _zone_rows(block, mk, column_periods(block.header_lines + block.body[:6]),
                              stop_at_gas=False)
    for block in source_blocks:
        records += _by_source(block, mk)
    if not records:
        log.warning("no energy tables recognised in %s", doc.path.name)
    return records


def _unique(blocks: list[TableBlock]) -> list[TableBlock]:
    """A caption repeated on the same page (e.g. in a chart title) is parsed once."""
    seen: set[tuple[int, str]] = set()
    out = []
    for b in blocks:
        key = (b.page_no, b.caption[:40])
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


def parse(doc: StoredDoc) -> dict[str, list[Record]]:
    records = parse_pages(read_pages(doc.path), doc)
    gas = [r for r in records if r.variable.startswith("gas_")]
    elec = [r for r in records if r.variable.startswith("electricity_")]
    return {"bot_gas": gas, "bot_electricity": elec}
