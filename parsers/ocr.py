"""OCR for scanned PDFs (RapidOCR, ONNX runtime; pip-installed, no system tools).

Pages are rendered with pypdfium2 and OCR'd; recognised text boxes are grouped
into lines by vertical position. Results are cached per file hash and page
because OCR takes roughly 15 s per page.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from parsers.pdf_text import PageText
from pipeline import config

log = logging.getLogger(__name__)

OCR_CACHE_DIR: Path = config.CACHE_DIR / "ocr"
RENDER_SCALE = 2.5  # about 180 dpi; lower scales misread small digits
LINE_TOLERANCE_PX = 14


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR  # imported lazily: slow to load

    return RapidOCR()


def boxes_to_lines(boxes: list[tuple[float, float, str]]) -> list[str]:
    """Group (y_centre, x_left, text) boxes into reading-order lines."""
    lines: list[str] = []
    current: list[tuple[float, str]] = []
    last_y: float | None = None
    for y, x, text in sorted(boxes):
        if last_y is not None and abs(y - last_y) > LINE_TOLERANCE_PX:
            lines.append(" ".join(t for _, t in sorted(current)))
            current = []
        current.append((x, text))
        last_y = y
    if current:
        lines.append(" ".join(t for _, t in sorted(current)))
    return lines


def _ocr_page(pdf, index: int, rotation: int = 0) -> str:
    import numpy as np

    image = np.array(pdf[index].render(scale=RENDER_SCALE, rotation=rotation)
                     .to_pil().convert("RGB"))
    result, _ = _engine()(image)
    boxes = [((b[0][1] + b[2][1]) / 2, b[0][0], text) for b, text, _conf in (result or [])]
    return "\n".join(boxes_to_lines(boxes))


def ocr_page_rotated(path: Path, page_no: int, rotation: int,
                     cache_dir: Path = OCR_CACHE_DIR) -> PageText:
    """OCR one page turned by ``rotation`` degrees (landscape tables printed sideways)."""
    import pypdfium2 as pdfium

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{digest}_p{page_no}_r{rotation}.json"
    if cache_file.exists():
        text = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        text = _ocr_page(pdfium.PdfDocument(str(path)), page_no - 1, rotation)
        cache_file.write_text(json.dumps(text, ensure_ascii=False), encoding="utf-8")
    return PageText(page_no, text, False)


_TOC_LINE = re.compile(r"\.{4,}|…|\s\d{1,3}\s*$")


def _stops(stop: re.Pattern[str], text: str) -> bool:
    """True if a non-contents line matches ``stop`` (TOC entries end in dots/page numbers)."""
    if re.search(r"\bCONTENTS\b", text, re.I):
        return False  # a contents page lists the stop heading without starting it
    return any(stop.search(line) and not _TOC_LINE.search(line) for line in text.splitlines())


def ocr_pages(path: Path, stop: re.Pattern[str] | None = None, max_pages: int = 60,
              start_page: int = 1, cache_dir: Path = OCR_CACHE_DIR,
              section_start: re.Pattern[str] | None = None) -> list[PageText]:
    """OCR pages from ``start_page`` until a page matches ``stop`` or ``max_pages``.

    With ``section_start``, ``stop`` only applies on pages after the one where
    the wanted section began (running headers can repeat the stop heading).
    """
    import pypdfium2 as pdfium

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(path))
    pages: list[PageText] = []
    started_on: int | None = None if section_start is not None else start_page - 1
    last = min(len(pdf), start_page - 1 + max_pages)
    for index in range(start_page - 1, last):
        cache_file = cache_dir / f"{digest}_p{index + 1}.json"
        if cache_file.exists():
            text = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            text = _ocr_page(pdf, index)
            cache_file.write_text(json.dumps(text, ensure_ascii=False), encoding="utf-8")
        pages.append(PageText(index + 1, text, False))
        if started_on is None and section_start is not None and section_start.search(text) \
                and not re.search(r"\bCONTENTS\b", text, re.I):
            started_on = index + 1
            if stop is not None and _stops(stop, text):
                # The stop heading is printed as a running header on the section's own
                # pages (TPDC FY2020/21): it cannot mark the end, so rely on max_pages.
                log.info("%s: stop heading is a running header; reading up to %d pages",
                         path.name, max_pages)
                stop = None
        still_in_section = section_start is not None and section_start.search(text)
        if (stop is not None and started_on is not None and index + 1 > started_on
                and index + 1 > start_page + 2 and not still_in_section and _stops(stop, text)):
            break
    log.info("OCR %s: %d pages", path.name, len(pages))
    return pages
