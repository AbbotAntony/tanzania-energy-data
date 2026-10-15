"""Extract gas volumes from TPDC audited financial statements (scanned; OCR).

The volumes are written as sentences in the "Report by those charged with
governance", e.g.:

  "the total amount of gas dispatched to power plants was 38,622.32 MMSCF up
   from 32,661.00 MMSCF recorded in 2021/22. For industrial segment, a total
   of 6,409.75 MMSCF down from 7,990.00 MMSCF in 2021-22 ..."

Each volume becomes a record: the category comes from the nearest keyword
before the number (power, industry, households...), and the period is the
report's fiscal year unless the number follows "from"/"compared to" (prior
year) or is followed by an explicit FY. These are TPDC's own volumes, not
national totals. Every value carries the source sentence and the flags
``ocr_extracted;narrative_extraction`` and should be spot-checked.
"""

from __future__ import annotations

import logging
import re

from parsers.ewura_gas import fy_end_year, report_fy
from parsers.ocr import ocr_pages
from parsers.pdf_text import PageText, fy_label
from pipeline.provenance import Record
from pipeline.storage import StoredDoc

log = logging.getLogger(__name__)

# Upper-case section headings only (prose mentions "statement of financial position" too);
# OCR sometimes drops the spaces ("STATEMENTOFDIRECTORS'RESPONSIBILITIES").
_STOP = re.compile(r"^\s*(\d+(\.\d+)?\s*)?(CONSOLIDATED\s*)?(STATEMENT\s*OF\s*DIRECTORS.{0,25}RESPONSIBILIT"
                   r"|STATEMENT\s*OF\s*FINANCIAL\s*POSITION)")
_VOLUME = re.compile(
    r"(?P<num>\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>MMSCF(?!D)|MMscf(?!d)|mmscf(?!d)|MMBTU|MMBtu|mmbtu"
    r"|million standard cubic feet)\b")  # Bcf/Tcf figures are reserves, not annual flows
_PRIOR = re.compile(r"(from|compared (to|with)|against|previous(ly)?|last year|prior year)\W*$", re.I)
_FY_AFTER = re.compile(r"^\W{0,3}(?:recorded |reported )?(?:in|during|for)\s+(?:the\s+)?(?:FY\s*|year\s*)?"
                       r"(20\d\d)\s*[/-]\s*(?:20)?(\d\d)", re.I)

# (pattern, category); the last match before a number wins
_CATEGORIES = (
    (re.compile(r"power|tanesco|electricity generation", re.I), "Power generation"),
    (re.compile(r"industr", re.I), "Industries"),
    (re.compile(r"household|domestic|cooking", re.I), "Households"),
    (re.compile(r"\bCNG\b|vehicle", re.I), "CNG vehicles"),
    (re.compile(r"institution|commercial", re.I), "Commercial and institutions"),
    (re.compile(r"received for processing|plant received|received", re.I), "Received at processing plants"),
    (re.compile(r"charged into", re.I), "Charged into NNGI pipeline"),
    (re.compile(r"export", re.I), "Exported to pipeline"),
    (re.compile(r"produc", re.I), "Produced"),
    (re.compile(r"\bsold\b|\bsales?\b", re.I), "Sold (all customers)"),
)
_PLANTS = (("madimba", "Madimba"), ("songo songo", "Songo Songo"), ("mnazi", "Mnazi Bay"))


def clean_ocr(text: str) -> str:
    """Fix common OCR confusions inside numbers ("2O23" -> "2023", "1,2O0" -> "1,200")."""
    text = re.sub(r"(?<=\d)[Oo](?=[\d,.])|(?<=[\d,.])[Oo](?=\d)|(?<=\d)[Oo]\b(?=\s*/)", "0", text)
    return re.sub(r"(?<=\d)[Il](?=\d)", "1", text)


def sentences(pages: list[PageText]) -> list[tuple[int, str]]:
    """Paragraph-joined sentences with the page they start on."""
    out: list[tuple[int, str]] = []
    for page in pages:
        text = clean_ocr(" ".join(page.text.split()))
        for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", text):
            out.append((page.page_no, s))
    return out


def nearest_category(before: str, after: str) -> str | None:
    """Category keyword closest to the number, looking back to the previous
    number and forward to the next ("40,824.45 MMSCF was exported")."""
    best: tuple[int, str] | None = None
    for rx, name in _CATEGORIES:
        for m in rx.finditer(before):
            dist = len(before) - m.end()
            if best is None or dist < best[0]:
                best = (dist, name)
        m = rx.search(after)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


def _category(context: str) -> str | None:
    best, pos = None, -1
    for rx, name in _CATEGORIES:
        for m in rx.finditer(context):
            if m.start() > pos:
                best, pos = name, m.start()
    return best


def _plant(context: str) -> str:
    low = context.lower()
    hits = [(low.rfind(k), name) for k, name in _PLANTS if k in low]
    return max(hits)[1] if hits else ""


def parse_number(token: str) -> float | None:
    """"38,622.32" -> 38622.32; OCR may print a period as thousands separator ("38.622.32")."""
    if token.count(".") > 1:
        head, _, dec = token.rpartition(".")
        token = head.replace(".", ",") + "." + dec
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


# "Quantity 17,201 mmscf 12,441" (FY2016/17), "Quantity processed 24,000 12,441 ... mmscf mmscf"
# (FY2015/16), "Quantity 33,396m 25,005m ... scf" (FY2017/18; m = million)
_KPI_ROW = re.compile(r"Quantity(?:\s+processed)?\s+(?P<target>\d[\d,.]*)\s*(?:mmscf|m\s*scf|m)?\s+"
                      r"(?P<actual>\d[\d,.]*)\s*(?:mmscf|m\s*scf|m)?\b", re.I)
_KPI_UNIT = re.compile(r"mmscf|\dm\b|m\s*scf", re.I)
# FY2018/19-FY2019/20: "Processed and transported 15.48 Billion Cubic Feet (BCF) of natural gas"
_PROCESSED_BCF = re.compile(r"[Pp]rocessed and transported\s+(?P<num>\d[\d,.]*)\s*"
                            r"(?P<unit>Billion Cubic Feet|BCF)", re.I)


def kpi_rows(pages: list[PageText], doc: StoredDoc, fy: str | None) -> list[Record]:
    """Older editions (about FY2015-FY2019) report a KPI table row:
    "Quantity processed and transported via NNGI | 17,201 mmscf | 12,441 mmscf"."""
    out = []
    for page in pages:
        text = clean_ocr(" ".join(page.text.split()))
        for m in _KPI_ROW.finditer(text):
            context = text[m.start():m.start() + 200]
            if not _KPI_UNIT.search(context):
                continue  # a "Quantity" row about something other than gas volumes
            for measure in ("target", "actual"):
                out.append(Record(
                    source="TPDC", url=doc.ref.url, doc_title=doc.ref.title, pub_date=doc.ref.pub_date,
                    retrieved_date=doc.retrieved_date, pdf_page=page.page_no,
                    variable=f"tpdc_gas_kpi_{measure}", value=parse_number(m.group(measure)),
                    unit="MMscf", fy=fy, year=fy_end_year(fy),
                    category="Processed and transported via NNGI",
                    value_as_printed=m.group(measure), qa_flag="ocr_extracted;kpi_table",
                    extra={"sentence": context, "report_fy": fy},
                ))
        for m in _PROCESSED_BCF.finditer(text):
            out.append(Record(
                source="TPDC", url=doc.ref.url, doc_title=doc.ref.title, pub_date=doc.ref.pub_date,
                retrieved_date=doc.retrieved_date, pdf_page=page.page_no,
                variable="tpdc_gas_volume", value=parse_number(m.group("num")),
                unit="Bcf", fy=fy, year=fy_end_year(fy),
                category="Processed and transported via NNGI",
                value_as_printed=f"{m.group('num')} {m.group('unit')}",
                qa_flag="ocr_extracted;narrative_extraction",
                extra={"sentence": text[max(0, m.start() - 60):m.end() + 80], "report_fy": fy},
            ))
    return out


def extract(pages: list[PageText], doc: StoredDoc) -> list[Record]:
    fy = report_fy(doc.ref.title)
    prior = fy_label(int(fy[:4]) - 1) if fy else None
    records: list[Record] = kpi_rows(pages, doc, fy)
    for page_no, sentence in sentences(pages):
        prev_end = 0
        sentence_cat = _category(sentence)
        if _KPI_ROW.search(sentence):
            continue  # handled by kpi_rows (target vs actual)
        matches = list(_VOLUME.finditer(sentence))
        for k, m in enumerate(matches):
            before = sentence[prev_end:m.start()]
            after = sentence[m.end():matches[k + 1].start() if k + 1 < len(matches) else len(sentence)]
            category = nearest_category(before, after[:80]) or sentence_cat
            period, flags = fy, ["ocr_extracted", "narrative_extraction"]
            explicit = _FY_AFTER.match(sentence[m.end():])
            if explicit:
                period = fy_label(int(explicit.group(1)))
            elif _PRIOR.search(before):
                period = prior
            if category is None:
                flags.append("category_not_identified")
            records.append(Record(
                source="TPDC", url=doc.ref.url, doc_title=doc.ref.title, pub_date=doc.ref.pub_date,
                retrieved_date=doc.retrieved_date, pdf_page=page_no, variable="tpdc_gas_volume",
                value=parse_number(m.group("num")), unit=_unit(m.group("unit")),
                fy=period, year=fy_end_year(period), category=category or "unidentified",
                value_as_printed=f"{m.group('num')} {m.group('unit')}", qa_flag=";".join(flags),
                extra={"sentence": sentence[:600], "plant": _plant(before), "report_fy": fy},
            ))
            prev_end = m.end()
    return records


def _unit(printed: str) -> str:
    p = printed.lower()
    if "mmscf" in p or "million standard" in p:
        return "MMscf"
    return "MMBtu"


def flag_repeated_values(records: list[Record], min_value: float = 100.0) -> None:
    """Flag a volume printed identically for different fiscal years.

    Two-column OCR can attach a prior-year sentence to the wrong year, and some
    editions repeat last year's figure (e.g. 12,441 MMscf in FY2015/16 and
    FY2016/17). Values are kept; the flag asks for a check against the page.
    """
    seen: dict[tuple[float, str], set[str]] = {}
    for r in records:
        if r.value is not None and r.value >= min_value and r.fy:
            seen.setdefault((r.value, r.unit), set()).add(r.fy)
    for r in records:
        if r.value is not None and len(seen.get((r.value, r.unit), ())) > 1:
            r.qa_flag = ";".join(filter(None, [r.qa_flag, "same_value_reported_for_other_year"]))


_SECTION_START = re.compile(r"REPORT\s*OF\s*THE\s*DIRECTORS|REPORT\s*BY\s*THOSE\s*CHARGED", re.I)


def parse(doc: StoredDoc) -> list[Record]:
    pages = ocr_pages(doc.path, stop=_STOP, max_pages=60, section_start=_SECTION_START)
    records = extract(pages, doc)
    log.info("%s: %d gas volumes from %d OCR pages", doc.path.name, len(records), len(pages))
    return records
