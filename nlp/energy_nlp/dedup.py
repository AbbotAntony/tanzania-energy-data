"""Near-duplicate detection beyond exact reposts (Task 1, second pass).

Word 5-gram overlap (prep.find_duplicates) misses rewritten reposts. This pass
combines three signals for article pairs whose embeddings are similar:
  shared text  share of the shorter article's word 5-grams found in the other
  cosine       similarity of MiniLM document embeddings
  dates/title  days apart and title word overlap
Pairs are classified, then reviewed by hand in a decisions file that prep.py
reads; only pairs marked ``remove_later`` are removed (the earliest copy stays).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from energy_nlp.prep import shingles

COSINE_FLOOR = 0.85
REPOST_SHARED = 0.70           # >= this share of the shorter text: repost
PARTIAL_SHARED = 0.40          # 0.40-0.70: partial reuse (kept)
PARAPHRASE_COSINE = 0.97       # rewritten repost: near-identical meaning,
PARAPHRASE_DAYS = 7            #   published within a week,
PARAPHRASE_TITLE = 0.50        #   with largely the same title words
TITLE_STOP = {"the", "of", "and", "in", "to", "a", "for", "on", "as", "tz", "s"}
DEFAULT_DECISION = {"repost": "remove_later", "paraphrased_repost": "remove_later",
                    "partial_reuse": "keep_both"}


def title_words(title: str) -> set[str]:
    return set(re.findall(r"[a-z]+", title.lower())) - TITLE_STOP


def classify_pair(shared: float, cosine: float, days: int, title_overlap: float) -> str | None:
    if shared >= REPOST_SHARED:
        return "repost"
    if cosine >= PARAPHRASE_COSINE and days <= PARAPHRASE_DAYS and title_overlap >= PARAPHRASE_TITLE:
        return "paraphrased_repost"
    if shared >= PARTIAL_SHARED:
        return "partial_reuse"
    return None


def candidate_pairs(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Classified pairs among rows of ``df`` (post_id, pub_date, title, text, url);
    ``emb`` holds unit-length embeddings in the same row order."""
    sim = emb @ emb.T
    sh = [shingles(t) for t in df["text"]]
    tw = [title_words(t) for t in df["title"]]
    rows = []
    for i, j in zip(*np.where(np.triu(sim >= COSINE_FLOOR, 1))):
        shared = len(sh[i] & sh[j]) / min(len(sh[i]), len(sh[j]))
        days = abs((df["pub_date"].iloc[i] - df["pub_date"].iloc[j]).days)
        title = len(tw[i] & tw[j]) / max(len(tw[i] | tw[j]), 1)
        kind = classify_pair(shared, float(sim[i, j]), days, title)
        if kind:
            a, b = df.iloc[i], df.iloc[j]
            rows.append({"kind": kind, "post_a": a["post_id"], "post_b": b["post_id"],
                         "date_a": a["pub_date"].date(), "date_b": b["pub_date"].date(), "days_apart": days,
                         "cosine": round(float(sim[i, j]), 3), "shared_text": round(shared, 2),
                         "title_overlap": round(title, 2), "url_a": a["url"], "url_b": b["url"]})
    cols = ["kind", "post_a", "post_b", "date_a", "date_b", "days_apart", "cosine", "shared_text",
            "title_overlap", "url_a", "url_b"]
    return pd.DataFrame(rows, columns=cols).sort_values(["kind", "shared_text"], ascending=[False, False])


def default_decisions(pairs: pd.DataFrame) -> pd.DataFrame:
    out = pairs[["post_a", "post_b", "kind"]].copy()
    out["decision"] = out["kind"].map(DEFAULT_DECISION)
    out["note"] = ""
    return out
