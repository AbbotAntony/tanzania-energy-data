"""Task 1, second pass: find rewritten reposts that word-overlap dedup misses.

Run after 01_prep.py, from the repo root (needs requirements-embed.txt):
    nlp/.venv/Scripts/python nlp/scripts/01b_near_duplicates.py [--write-decisions]

Candidates are found among main-window articles that passed the first pass
(on-topic, not an exact repost), whether or not an earlier review removed them,
so the list is stable across re-runs. Output: nlp/tables/near_duplicate_candidates.csv.
--write-decisions creates nlp/config/near_duplicate_decisions.csv with default
decisions (reposts and paraphrased reposts: remove_later; partial reuse:
keep_both) if it does not exist yet. That file is reviewed by hand; 01_prep.py
reads it. After changing it, re-run 01_prep.py and everything downstream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

import pandas as pd  # noqa: E402

from energy_nlp import DATA_DIR, NLP_DIR, TABLES_DIR  # noqa: E402
from energy_nlp import dedup, embed, prep  # noqa: E402

DECISIONS = NLP_DIR / "config" / "near_duplicate_decisions.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-decisions", action="store_true")
    args = ap.parse_args()

    full = prep.load_corpus(DATA_DIR / "corpus.csv", analysis_only=False)
    pool = full[~full["off_topic"] & ~full["is_duplicate"] & full["in_main_window"]].reset_index(drop=True)
    texts = (pool["title"] + ". " + pool["text"]).tolist()
    emb = embed.cached_embeddings(pool["post_id"].tolist(), texts, DATA_DIR / "emb_minilm.npy",
                                  DATA_DIR / "emb_minilm_ids.csv")
    pairs = dedup.candidate_pairs(pool, emb)
    pairs.to_csv(TABLES_DIR / "near_duplicate_candidates.csv", index=False)
    print(pairs["kind"].value_counts().to_string())

    if args.write_decisions:
        if DECISIONS.exists():
            sys.exit(f"{DECISIONS} exists (it holds reviewed decisions); edit it by hand instead")
        DECISIONS.parent.mkdir(parents=True, exist_ok=True)
        dedup.default_decisions(pairs).to_csv(DECISIONS, index=False)
        print(f"default decisions -> {DECISIONS} (review, then re-run 01_prep.py)")
    elif DECISIONS.exists():
        known = pd.read_csv(DECISIONS)
        new = pairs.merge(known[["post_a", "post_b"]], on=["post_a", "post_b"], how="left", indicator=True)
        new = new[new["_merge"] == "left_only"]
        if len(new):
            print(f"{len(new)} candidate pair(s) not yet in {DECISIONS.name} - review them:")
            print(new[["kind", "post_a", "post_b", "shared_text", "cosine"]].to_string(index=False))


if __name__ == "__main__":
    main()
