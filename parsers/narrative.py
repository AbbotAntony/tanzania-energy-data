"""Keyword-sentence extraction from long policy documents (English and Swahili).

Budget speeches and master plans state many energy figures in prose. Rather
than guess at table layouts, sentences mentioning gas, HFO or power generation
together with a number are kept with their page so they can be cited or
followed up. Values are not parsed out of these sentences.
"""

from __future__ import annotations

import re

from parsers.pdf_text import PageText
from pipeline.provenance import Record
from pipeline.storage import StoredDoc

TOPIC = re.compile(
    r"natural gas|\bgas\b|\bHFO\b|heavy fuel|fuel oil|\bMMscf|\bGWh\b|\bMW\b|Songas|Kinyerezi|Ubungo"
    r"|Julius Nyerere|JNHPP|Songo Songo|Mnazi Bay|GASCO|TPDC"
    r"|gesi asilia|futi za ujazo|mafuta mazito|uzalishaji wa umeme|Bwawa la Julius Nyerere", re.I)
NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def extract_sentences(pages: list[PageText], doc: StoredDoc, source: str,
                      min_len: int = 50, max_len: int = 1200) -> list[Record]:
    out: list[Record] = []
    for page in pages:
        text = " ".join(page.text.split())
        for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z(])", text):
            if (min_len <= len(sentence) <= max_len and TOPIC.search(sentence)
                    and NUMBER.search(sentence) and "(cid:" not in sentence):
                out.append(Record(
                    source=source, url=doc.ref.url, doc_title=doc.ref.title, pub_date=doc.ref.pub_date,
                    retrieved_date=doc.retrieved_date, pdf_page=page.page_no,
                    variable="explanatory_text", value=None, unit="text", category="narrative",
                    value_as_printed=sentence,
                    qa_flag="text_glyph_decoded" if page.cid_decoded else "",
                    extra={"topics": ",".join(sorted({m.group(0).lower() for m in TOPIC.finditer(sentence)}))[:200]},
                ))
    return out
