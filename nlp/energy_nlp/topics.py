"""Task 2: topic models and their evaluation.

The NMF baseline runs on TF-IDF. Each candidate k is scored on:
- coherence: mean NPMI of top-term pairs, from document co-occurrence in this corpus
- diversity: share of unique terms across all topics' top terms
- stability: mean best-match cosine similarity of topics across random restarts
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Reporting verbs, filler and honorifics that carry no theme.
CUSTOM_STOP = {
    "said", "says", "say", "noted", "added", "according", "speaking", "explained", "stated",
    "revealed", "told", "including", "ensure", "various", "key", "major", "significant",
    "expected", "efforts", "continue", "continued", "currently", "time", "year", "years", "new",
    "mr", "mrs", "ms", "dr", "eng", "hon", "prof", "cent", "percent", "per", "dar", "es", "salaam",
    "tanzania", "tanzanian", "tanzanians", "country", "countries", "also", "well", "like",
    "however", "important", "made", "make", "making", "would", "could", "one", "two", "today",
    "yesterday", "week", "month", "months", "day", "days", "able", "use", "used", "using",
    "among", "already", "since", "within", "way", "part", "number", "total", "many", "much",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | CUSTOM_STOP)
TOKEN_PATTERN = r"(?u)\b[a-z][a-z\-]*[a-z]\b"

# Officials' names. Without this NMF builds "person" topics (one per minister),
# which track cabinet reshuffles rather than themes.
TITLED_NAME = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Eng|Prof|Hon|Ambassador|President|Minister|Secretary|Commissioner|"
    r"Director|Chairperson|Chairman|Speaker)\.?\s+((?:[A-Z][a-z]+(?:-[A-Z][a-z]+)?\s?){1,3})")
NOT_NAMES = {
    "mr", "mrs", "ms", "dr", "eng", "engineer", "prof", "hon", "ambassador", "general", "tax",
    "julius", "nyerere",       # part of "Julius Nyerere Hydropower Project"
}


def person_names(texts: Iterable[str], min_docs: int = 5) -> set[str]:
    """Lower-cased words that follow a title (Mr, Minister, ...) in >= ``min_docs`` texts."""
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update({w.lower() for m in TITLED_NAME.findall(t) for w in m.split()})
    return {w for w, n in counts.items() if n >= min_docs} - NOT_NAMES


def make_vectorizer(extra_stop: Iterable[str] = (), min_df: int = 5,
                    max_df: float = 0.5) -> TfidfVectorizer:
    stop = sorted(set(STOP_WORDS) | set(extra_stop))
    return TfidfVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN, stop_words=stop,
                           ngram_range=(1, 2), min_df=min_df, max_df=max_df, sublinear_tf=True)


def fit_nmf(X: sparse.spmatrix, k: int, seed: int, init: str = "nndsvda") -> tuple[np.ndarray, np.ndarray]:
    model = NMF(n_components=k, init=init, random_state=seed, max_iter=1000)
    W = model.fit_transform(X)
    return W, model.components_


def top_term_idx(H: np.ndarray, n: int = 10) -> np.ndarray:
    """Indices of each topic's ``n`` highest-weight terms (rows of H)."""
    return np.argsort(-H, axis=1)[:, :n]


def npmi_coherence(top_idx: np.ndarray, X: sparse.spmatrix) -> np.ndarray:
    """Mean NPMI over all pairs of each topic's top terms.

    Probabilities are document frequencies in ``X`` (any nonzero = present).
    Pairs that never co-occur score -1, the NPMI minimum.
    """
    B = (sparse.csc_matrix(X) > 0).astype(np.float64)
    n_docs = B.shape[0]
    p = np.asarray(B.sum(axis=0)).ravel() / n_docs
    scores = []
    for terms in top_idx:
        sub = B[:, terms]
        joint = (sub.T @ sub).toarray() / n_docs
        vals = []
        for i, j in combinations(range(len(terms)), 2):
            pij = joint[i, j]
            if pij == 0:
                vals.append(-1.0)
            elif pij == 1:
                vals.append(1.0)
            else:
                vals.append(np.log(pij / (p[terms[i]] * p[terms[j]])) / -np.log(pij))
        scores.append(np.mean(vals))
    return np.array(scores)


def topic_diversity(top_idx: np.ndarray) -> float:
    """Unique terms / all top-term slots (1 = no term shared between topics)."""
    return len(np.unique(top_idx)) / top_idx.size


def match_similarity(H1: np.ndarray, H2: np.ndarray) -> float:
    """Mean cosine similarity of topics after optimal one-to-one matching."""
    A = H1 / np.linalg.norm(H1, axis=1, keepdims=True)
    B = H2 / np.linalg.norm(H2, axis=1, keepdims=True)
    sim = A @ B.T
    rows, cols = linear_sum_assignment(-sim)
    return float(sim[rows, cols].mean())


def stability(Hs: list[np.ndarray]) -> float:
    """Mean pairwise ``match_similarity`` across runs (1 = identical topics every run)."""
    return float(np.mean([match_similarity(a, b) for a, b in combinations(Hs, 2)]))


def choose_k(sel: pd.DataFrame, k_min: int = 10, tol: float = 0.05) -> int:
    """Most stable k among k >= ``k_min`` whose coherence is within ``tol`` of the best.

    ``sel`` needs columns k, npmi, stability. The floor keeps enough topics to
    separate the energy sub-themes the research questions need.
    """
    cand = sel[sel["k"] >= k_min]
    cand = cand[cand["npmi"] >= cand["npmi"].max() - tol]
    return int(cand.sort_values(["stability", "k"], ascending=[False, True]).iloc[0]["k"])


def doc_proportions(W: np.ndarray) -> np.ndarray:
    """Rows of W scaled to sum to 1; an all-zero row is spread evenly."""
    s = W.sum(axis=1, keepdims=True)
    out = np.divide(W, s, out=np.full_like(W, 1 / W.shape[1], dtype=float), where=s > 0)
    return out


def period_shares(props: np.ndarray, periods: pd.Series, n_boot: int = 1000, seed: int = 42,
                  level: float = 0.95) -> pd.DataFrame:
    """Mean topic proportion per period with percentile bootstrap intervals.

    Articles are resampled with replacement within each period, so the interval
    reflects which articles happened to be published (and scraped) that period.
    Returns long format: period, topic, share, lo, hi, n_articles.
    """
    rng = np.random.default_rng(seed)
    a = (1 - level) / 2
    rows = []
    for period, idx in pd.Series(np.arange(len(periods))).groupby(np.asarray(periods)):
        P = props[idx.to_numpy()]
        boot = P[rng.integers(0, len(P), size=(n_boot, len(P)))].mean(axis=1)
        lo, hi = np.quantile(boot, [a, 1 - a], axis=0)
        for t in range(P.shape[1]):
            rows.append((period, t, P[:, t].mean(), lo[t], hi[t], len(P)))
    return pd.DataFrame(rows, columns=["period", "topic", "share", "lo", "hi", "n_articles"])


def share_change_by_group(flag: pd.Series, group: pd.Series, period: pd.Series,
                          before: str, after: str) -> pd.DataFrame:
    """Split the change in the share of ``flag`` articles across ``group`` values.

    share(flag) = sum over g of share(flag and group == g), so the change between
    the two periods is exactly the sum of the per-group changes (``contribution``).
    """
    rows = []
    for name in (before, after):
        sel = period == name
        n = int(sel.sum())
        part = group[sel & flag].value_counts() / n
        rows.append(part.rename(name))
    out = pd.concat(rows, axis=1).fillna(0.0)
    out["contribution"] = out[after] - out[before]
    out.index.name = "group"
    return out.sort_values("contribution", ascending=False).reset_index()
