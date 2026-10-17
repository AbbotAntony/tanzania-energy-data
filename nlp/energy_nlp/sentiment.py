"""Task 3: off-the-shelf tone models, applied to what an article says about gas.

The codebook codes tone *toward natural gas*, so models never see whole
articles. Only sentences the mention classifier calls substantive are scored.
An article with no substantive gas sentence is ``not_about_gas`` (by rule, not
by a model). Three models:

- VADER: lexicon baseline; mean compound score, +/-0.05 cut-offs (the authors').
- finnews: DistilRoBERTa fine-tuned on Financial PhraseBank news sentences,
  which code reported gains/losses as positive/negative (as codebook rule T2a).
  Sentence probabilities are averaged per article.
- zeroshot: DeBERTa-v3 NLI, asked directly about natural gas through the
  hypothesis template, on the article's gas sentences joined together (capped
  at ZS_MAX_WORDS).
"""

from __future__ import annotations

import numpy as np

from energy_nlp import mentions as mn

TONES = ("negative", "neutral", "positive")
FIN_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
FIN_REVISION = "ae0eab9ad336d7d548e0efe394b07c04bcaf6e91"
ZS_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
ZS_REVISION = "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8"
ZS_TEMPLATE = "The text is {} about natural gas."
ZS_WORDS = {"negative": "negative", "neutral": "neutral", "positive": "positive"}
VADER_CUT = 0.05
# Zero-shot input cap: ~190 words ~ 256 DeBERTa tokens. The median article's gas
# text is ~110 words, so only long ones are cut; this keeps CPU time manageable.
ZS_MAX_WORDS = 190


def gas_sentences(text: str) -> list[str]:
    """Sentences that say something about gas (mention classifier = substantive)."""
    return [s for s in mn.SENT_SPLIT.split(text) if mn.classify_sentence(s) == "substantive"]


def vader_label(compound: float) -> str:
    if compound >= VADER_CUT:
        return "positive"
    if compound <= -VADER_CUT:
        return "negative"
    return "neutral"


def score_vader(docs: list[list[str]], progress=None) -> list[dict]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    out = []
    for sents in docs:
        c = float(np.mean([sia.polarity_scores(s)["compound"] for s in sents]))
        out.append({"label": vader_label(c), "score": c})
    return out


def mean_probs(per_sentence: list[dict[str, float]]) -> dict[str, float]:
    """Average class probabilities over an article's sentences."""
    return {t: float(np.mean([p[t] for p in per_sentence])) for t in TONES}


def _cpu_dtype():
    """float32: the zero-shot weights are stored as float16, which CPUs without
    native half precision emulate ~13x slower. Results are unaffected."""
    import torch

    return torch.float32


def cap_words(text: str, n: int = ZS_MAX_WORDS) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n])


def _in_chunks(items: list, size: int, fn, progress) -> list:
    """Apply ``fn`` to consecutive chunks of ``items``, reporting progress after each."""
    out = []
    for start in range(0, len(items), size):
        out.extend(fn(items[start:start + size]))
        if progress:
            progress(min(start + size, len(items)), len(items))
    return out


def score_finnews(docs: list[list[str]], batch_size: int = 32, progress=None) -> list[dict]:
    from transformers import pipeline

    clf = pipeline("text-classification", model=FIN_MODEL, revision=FIN_REVISION, device=-1,
                   dtype=_cpu_dtype(), top_k=None, truncation=True)
    flat = [s for sents in docs for s in sents]
    owner = [i for i, sents in enumerate(docs) for _ in sents]
    preds = _in_chunks(flat, 256, lambda c: clf(c, batch_size=batch_size), progress)
    by_doc: list[list[dict]] = [[] for _ in docs]
    for i, p in zip(owner, preds):
        by_doc[i].append({d["label"].lower(): d["score"] for d in p})
    out = []
    for sents in by_doc:
        probs = mean_probs(sents)
        out.append({"label": max(probs, key=probs.get), "score": probs["positive"] - probs["negative"],
                    **{f"p_{t}": probs[t] for t in TONES}})
    return out


def score_zeroshot(docs: list[list[str]], batch_size: int = 8, progress=None) -> list[dict]:
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=ZS_MODEL, revision=ZS_REVISION, device=-1,
                   dtype=_cpu_dtype())
    premises = [cap_words(" ".join(sents)) for sents in docs]
    results = _in_chunks(premises, 50, lambda c: clf(c, candidate_labels=list(ZS_WORDS.values()),
                                                     hypothesis_template=ZS_TEMPLATE, multi_label=False,
                                                     batch_size=batch_size), progress)
    out = []
    for r in results:
        probs = {t: r["scores"][r["labels"].index(w)] for t, w in ZS_WORDS.items()}
        out.append({"label": max(probs, key=probs.get), "score": probs["positive"] - probs["negative"],
                    **{f"p_{t}": probs[t] for t in TONES}})
    return out
