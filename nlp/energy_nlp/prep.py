"""Task 1: load, clean and deduplicate the Daily News energy corpus.

Text is re-extracted from the raw WordPress HTML (by post id) so that inserted
"ALSO READ: <other headline>" links and photo captions can be dropped by markup
rather than guessed from flattened text. Topic tags are then re-matched on the
clean text with the scraper's own rules (``parsers.news``).
"""

from __future__ import annotations

import glob
import json
import re
from html import unescape
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from parsers.news import TOPICS, WEAK, match_topics

MIN_ARTICLES = 10            # months below this are flagged as too thin
MAIN_WINDOW_START = pd.Period("2022Q3", freq="Q")
DUP_JACCARD = 0.8            # word 5-gram Jaccard at or above this = same article
SHINGLE = 5

# A paragraph that is only a cross-link to another story.
LINK_PARA = re.compile(r"^\W*(also read|read also|read more|related)\b", re.I)
LINK_PARA_MAX_CHARS = 300    # never drop a long block just because it starts that way
# "DAR ES SALAAM: " / "DODOMA : " at the start of a story.
DATELINE = re.compile(r"^\s*([A-Z][A-Za-z .'\-]{1,40}?)\s?:\s+(?=\S)")
CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
    "​": "", "‌": "", "⁠": "", "﻿": "",
})


def normalise(text: str) -> str:
    """Unescape entities, map typographic characters to ASCII, collapse whitespace."""
    if not isinstance(text, str):          # missing title/text arrives as NaN
        return ""
    return " ".join(unescape(text).translate(CHAR_MAP).split())


def clean_html(html: str) -> str:
    """Article body text without cross-link paragraphs, captions or scripts."""
    soup = BeautifulSoup(html or "", "lxml")
    for el in soup.select("script, style, figure, figcaption, .wp-caption, .wp-caption-text"):
        el.decompose()
    links = [el for el in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
             if len(t := el.get_text(" ", strip=True)) <= LINK_PARA_MAX_CHARS and LINK_PARA.match(t)]
    for el in links:
        if not el.decomposed:
            el.decompose()
    return normalise(soup.get_text(" "))


def split_dateline(text: str) -> tuple[str, str]:
    """(dateline, body) - the dateline is the leading place name, if any."""
    m = DATELINE.match(text)
    if m and m.group(1).isupper():
        return m.group(1).strip(), text[m.end():]
    return "", text


def strong_hits(text: str) -> int:
    """Number of strong (non-weak) topic phrase matches in ``text``."""
    return sum(len(rx.findall(text)) for name, rx in TOPICS.items() if name not in WEAK)


def load_raw_html(raw_dir: Path) -> dict[int, str]:
    """post_id -> content HTML from every saved API page (latest retrieval wins)."""
    html: dict[int, str] = {}
    for f in sorted(glob.glob(str(raw_dir / "*" / "posts__*"))):
        for item in json.loads(Path(f).read_text(encoding="utf-8")):
            html[item["id"]] = item["content"]["rendered"]
    return html


def shingles(text: str, k: int = SHINGLE) -> frozenset[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(tuple(words[i:i + k]) for i in range(max(len(words) - k + 1, 1)))


def find_duplicates(df: pd.DataFrame, threshold: float = DUP_JACCARD) -> pd.DataFrame:
    """Near-duplicate pairs by word-shingle Jaccard.

    Returns columns ``post_id, dup_of, jaccard``: each duplicate points to the
    article kept for its group, which is the earliest published (then lowest id).
    """
    order = df.sort_values(["pub_date", "post_id"]).reset_index(drop=True)
    sh = [shingles(t) for t in order["text"]]
    size = [len(s) for s in sh]
    parent = list(range(len(order)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    best: dict[int, float] = {}
    by_len = sorted(range(len(order)), key=size.__getitem__)
    for a_pos, i in enumerate(by_len):
        for j in by_len[a_pos + 1:]:
            if size[i] < threshold * size[j]:       # Jaccard <= min/max size
                break
            jac = len(sh[i] & sh[j]) / len(sh[i] | sh[j])
            if jac >= threshold:
                ri, rj = find(i), find(j)
                parent[max(ri, rj)] = min(ri, rj)   # lower index = earlier article
                for x in (i, j):
                    best[x] = max(best.get(x, 0.0), jac)
    rows = [(order.at[i, "post_id"], order.at[find(i), "post_id"], round(best[i], 3))
            for i in range(len(order)) if find(i) != i]
    return pd.DataFrame(rows, columns=["post_id", "dup_of", "jaccard"])


def near_duplicate_groups(decisions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Apply reviewed near-duplicate decisions (dedup.py): pairs marked
    ``remove_later`` are linked into groups; in each group the earliest article
    (then lowest id) is kept. Returns ``post_id, near_dup_of`` for removed rows."""
    pairs = decisions[decisions["decision"] == "remove_later"]
    known = set(df["post_id"])
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["post_a"], pairs["post_b"]):
        if a in known and b in known:
            parent[find(a)] = find(b)
    order = df.set_index("post_id")["pub_date"]
    groups: dict[int, list[int]] = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    rows = []
    for members in groups.values():
        keep = min(members, key=lambda p: (order[p], p))
        rows += [(p, keep) for p in members if p != keep]
    return pd.DataFrame(rows, columns=["post_id", "near_dup_of"])


def build_corpus(news_csv: Path, raw_dir: Path, near_dup_decisions: Path | None = None) -> pd.DataFrame:
    """One row per scraped article, with clean text and inclusion flags.

    ``near_dup_decisions`` is the reviewed decisions file from dedup.py; when
    given, rewritten reposts marked ``remove_later`` are excluded as well."""
    src = pd.read_csv(news_csv)
    html = load_raw_html(Path(raw_dir))
    df = pd.DataFrame({
        "post_id": src["x_post_id"].astype(int),
        "pub_date": pd.to_datetime(src["pub_date"]),
        "url": src["url"],
        "title": src["doc_title"].map(normalise),
        "topics_scraper": src["x_topics"],
    })
    df["text_source"] = ["html" if p in html else "csv" for p in df["post_id"]]
    text = [clean_html(html[p]) if p in html else normalise(t)
            for p, t in zip(df["post_id"], src["x_text"])]
    df["dateline"], df["text"] = zip(*map(split_dateline, text))
    df["word_count"] = df["text"].str.split().str.len()
    df["month"] = df["pub_date"].dt.to_period("M")
    df["quarter"] = df["pub_date"].dt.to_period("Q")

    full = df["title"] + " " + df["text"]
    df["topics"] = [";".join(match_topics(t)) for t in full]
    df["off_topic"] = df["topics"] == ""                 # matched only via links/captions
    df["n_hits"] = [strong_hits(t) for t in full]
    df["topic_in_title"] = [strong_hits(t) > 0 for t in df["title"]]
    df["passing_mention"] = (df["n_hits"] == 1) & ~df["topic_in_title"]

    dups = find_duplicates(df[~df["off_topic"]])
    df = df.merge(dups, on="post_id", how="left")
    df["is_duplicate"] = df["dup_of"].notna()
    df["dup_of"] = df["dup_of"].astype("Int64")
    near = pd.DataFrame(columns=["post_id", "near_dup_of"])
    if near_dup_decisions is not None and Path(near_dup_decisions).exists():
        near = near_duplicate_groups(pd.read_csv(near_dup_decisions),
                                     df[~df["off_topic"] & ~df["is_duplicate"]])
    df = df.merge(near, on="post_id", how="left")
    df["near_dup_of"] = df["near_dup_of"].astype("Int64")
    df["is_near_duplicate"] = df["near_dup_of"].notna()
    df["in_main_window"] = df["quarter"] >= MAIN_WINDOW_START
    df["include"] = ~df["off_topic"] & ~df["is_duplicate"] & ~df["is_near_duplicate"]
    return df.sort_values(["pub_date", "post_id"]).reset_index(drop=True)


def load_corpus(path: Path, analysis_only: bool = True) -> pd.DataFrame:
    """Read data/nlp/corpus.csv back with proper types.

    ``analysis_only`` keeps included articles in the main window (2022-Q3 on).
    """
    df = pd.read_csv(path, dtype={"dateline": str, "title": str, "text": str, "topics": str},
                     keep_default_na=False,
                     na_values={"dup_of": [""], "jaccard": [""], "near_dup_of": [""]},
                     parse_dates=["pub_date"])
    if "is_near_duplicate" not in df:                     # corpus files from before the second pass
        df["is_near_duplicate"] = "False"
    for col in ("off_topic", "is_duplicate", "is_near_duplicate", "in_main_window", "include", "passing_mention",
                "topic_in_title"):
        df[col] = df[col].astype(str).eq("True")
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    if analysis_only:
        df = df[df["include"] & df["in_main_window"]].reset_index(drop=True)
    return df


def tag_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return df["topics"].str.get_dummies(sep=";").reindex(columns=list(TOPICS), fill_value=0)


def counts_by_period(df: pd.DataFrame, freq: str, last_complete: pd.Period | None = None,
                     min_articles: int = MIN_ARTICLES) -> pd.DataFrame:
    """Included-article counts per period (every period in range, zeros kept) and per tag.

    ``low_count`` flags periods below ``min_articles`` (scaled x3 for quarters);
    ``partial`` flags periods after ``last_complete``.
    """
    inc = df[df["include"]]
    col = "month" if freq == "M" else "quarter"
    span = pd.period_range(inc[col].min(), inc[col].max(), freq=freq)
    tags = tag_dummies(inc).groupby(inc[col]).sum()
    out = pd.DataFrame(index=span)
    out["n_articles"] = inc.groupby(col).size()
    out = out.join(tags).fillna(0).astype(int)
    floor = min_articles * (3 if freq == "Q" else 1)
    out.insert(1, "empty", out["n_articles"] == 0)
    out.insert(2, "low_count", out["n_articles"] < floor)
    out.insert(3, "partial", out.index > last_complete if last_complete is not None else False)
    out.insert(4, "in_main_window", out.index.asfreq("Q") >= MAIN_WINDOW_START)
    out.index.name = col
    return out.reset_index().assign(**{col: lambda d: d[col].astype(str)})


def tag_counts(df: pd.DataFrame) -> pd.DataFrame:
    inc = df[df["include"]]
    main = inc[inc["in_main_window"]]
    out = pd.DataFrame({"all_years": tag_dummies(inc).sum(), "main_window": tag_dummies(main).sum()})
    out["share_main_window"] = (out["main_window"] / len(main)).round(3)
    out.index.name = "tag"
    return out.reset_index()


def prep_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Article counts at each filtering step (CONSORT-style)."""
    steps = [
        ("scraped articles", len(df)),
        ("text re-extracted from raw HTML", int((df["text_source"] == "html").sum())),
        ("off-topic after removing links/captions", int(df["off_topic"].sum())),
        ("near-duplicates removed (word overlap)", int(df["is_duplicate"].sum())),
        ("rewritten reposts removed (reviewed, dedup.py)", int(df["is_near_duplicate"].sum())),
        ("included articles", int(df["include"].sum())),
        ("  of which before 2022-Q3 (descriptive only)", int((df["include"] & ~df["in_main_window"]).sum())),
        ("  of which in main window (2022-Q3 on)", int((df["include"] & df["in_main_window"]).sum())),
        ("  of which passing mention (1 hit, not in title)", int((df["include"] & df["passing_mention"]).sum())),
    ]
    return pd.DataFrame(steps, columns=["step", "n"])
