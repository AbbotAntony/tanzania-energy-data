"""Task 3: run the three tone models on every main-window article.

Run from the repo root:  nlp/.venv/Scripts/python -u nlp/scripts/03b_sentiment_predict.py
  --bench N   time each model on the first N articles only (nothing saved)

Each model's per-article output is saved to data/nlp/sentiment_<model>.csv as
soon as it finishes; a re-run skips models already saved for the same articles.
The merged file data/nlp/sentiment_predictions.csv is not committed and not
shown while hand-labelling is under way. Only aggregates are printed.

On a 4-core laptop CPU (i5-8250U) the zero-shot model dominates the run time
(about 1.4 s per article once loaded in float32).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from energy_nlp import DATA_DIR, TABLES_DIR  # noqa: E402
from energy_nlp import mentions as mn  # noqa: E402
from energy_nlp import prep, sentiment as se  # noqa: E402

MODELS = {"vader": se.score_vader, "finnews": se.score_finnews, "zeroshot": se.score_zeroshot}


def progress_printer(name: str):
    t0 = time.time()

    def report(done: int, total: int) -> None:
        el = time.time() - t0
        eta = el / done * (total - done)
        print(f"  {name}: {done}/{total}  elapsed {el / 60:.1f} min  remaining ~{eta / 60:.1f} min", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()

    import torch

    torch.manual_seed(0)
    d = prep.load_corpus(DATA_DIR / "corpus.csv")
    texts = (d["title"] + ". " + d["text"]).tolist()
    base = pd.DataFrame({"post_id": d["post_id"], "gas_mention_rule": [mn.article_mention(t) for t in texts]})
    sents = [se.gas_sentences(t) for t in texts]
    base["n_gas_sentences"] = [len(s) for s in sents]
    has = base["n_gas_sentences"] > 0
    ids = base.loc[has, "post_id"].tolist()
    docs = [s for s, h in zip(sents, has) if h]

    if args.bench:
        docs = docs[:args.bench]
        print(f"benchmark on {len(docs)} articles ({sum(map(len, docs))} sentences); full run = {len(ids)} articles")
        for name, fn in MODELS.items():
            t0 = time.time()
            fn(docs)
            per = (time.time() - t0) / len(docs)
            print(f"  {name}: {per:.2f} s/article -> full run ~{per * len(ids) / 60:.1f} min", flush=True)
        return

    print(f"{len(d)} articles, {len(ids)} with substantive gas sentences ({sum(map(len, docs))} sentences)",
          flush=True)
    out = base
    for name, fn in MODELS.items():
        path = DATA_DIR / f"sentiment_{name}.csv"
        if path.exists() and pd.read_csv(path)["post_id"].tolist() == ids:
            print(f"{name}: already done, loading {path.name}", flush=True)
            res = pd.read_csv(path)
        else:
            t0 = time.time()
            res = pd.DataFrame(fn(docs, progress=progress_printer(name))).add_prefix(f"{name}_")
            res.insert(0, "post_id", ids)
            res.round(4).to_csv(path, index=False)
            print(f"{name}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
        out = out.merge(res, on="post_id", how="left")
        # End-to-end 4-class prediction: the rule decides "about gas", the model the tone.
        out[f"{name}_tone4"] = out[f"{name}_label"].fillna("not_about_gas")

    out.round(4).to_csv(DATA_DIR / "sentiment_predictions.csv", index=False)

    # Aggregates only.
    dist = pd.DataFrame({m: out[f"{m}_tone4"].value_counts(normalize=True) for m in MODELS}).round(3)
    dist.index.name = "predicted_tone"
    dist.to_csv(TABLES_DIR / "sentiment_predicted_distribution.csv")
    sub = out[has]
    agree = pd.DataFrame([(a, b, round(cohen_kappa_score(sub[f"{a}_label"], sub[f"{b}_label"]), 3),
                           round(float((sub[f"{a}_label"] == sub[f"{b}_label"]).mean()), 3))
                          for a, b in [("vader", "finnews"), ("vader", "zeroshot"), ("finnews", "zeroshot")]],
                         columns=["model_a", "model_b", "kappa", "raw_agreement"])
    agree.to_csv(TABLES_DIR / "sentiment_model_agreement.csv", index=False)
    print("\n", dist.to_string())
    print("\nagreement between models on articles with gas sentences:\n", agree.to_string(index=False))


if __name__ == "__main__":
    main()
