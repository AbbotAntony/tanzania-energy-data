"""Save raw documents unchanged under data/raw/<source>/<retrieved_date>/."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pipeline import config
from pipeline.http_client import FetchResult
from pipeline.provenance import DocumentRef

log = logging.getLogger(__name__)

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


@dataclass(frozen=True)
class StoredDoc:
    """A raw document on disk plus its provenance."""

    ref: DocumentRef
    path: Path
    sha256: str
    retrieved_date: str  # YYYY-MM-DD


def safe_filename(url: str, default: str = "index.html") -> str:
    """Derive a Windows-safe filename from the last URL path segment."""
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or default
    query = urlsplit(url).query
    if query:
        stem, dot, ext = name.partition(".")
        name = f"{stem}__{_UNSAFE.sub('_', query)}{dot}{ext}" if dot else f"{name}__{query}"
    name = _UNSAFE.sub("_", name).strip(" .")
    return name[:180] or default


def store_raw(source: str, ref: DocumentRef, fetched: FetchResult,
              raw_dir: Path = config.RAW_DIR) -> StoredDoc:
    """Copy a fetched file into the dated raw folder and append to the manifest."""
    retrieved_date = fetched.fetched_at[:10]
    dest_dir = raw_dir / source / retrieved_date
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_filename(fetched.url)
    sha = hashlib.sha256(fetched.read_bytes()).hexdigest()
    if not dest.exists():
        shutil.copyfile(fetched.path, dest)
        _append_manifest(dest_dir / "manifest.jsonl", ref, fetched, dest, sha)
    return StoredDoc(ref=ref, path=dest, sha256=sha, retrieved_date=retrieved_date)


def _append_manifest(manifest: Path, ref: DocumentRef, fetched: FetchResult,
                     dest: Path, sha: str) -> None:
    entry = {
        "source": ref.source, "url": ref.url, "final_url": fetched.final_url,
        "title": ref.title, "category": ref.category, "pub_date": ref.pub_date,
        "listing_url": ref.listing_url, "file": dest.name, "sha256": sha,
        "content_type": fetched.content_type, "retrieved_at": fetched.fetched_at,
    }
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
