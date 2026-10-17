"""Task 4: can a simple supervised model reproduce the hand-coded framing and tone?

TF-IDF + logistic regression, evaluated by repeated stratified k-fold
cross-validation against a majority-class baseline. With ~150 (frame) and ~70
(tone) labelled articles a single train/test split would be too noisy, so the
comparison uses all folds and the Nadeau-Bengio corrected resampled t-test,
which allows for the overlap between training sets across folds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from energy_nlp import evaluate as ev
from energy_nlp.topics import STOP_WORDS, TOKEN_PATTERN

C_GRID = (0.1, 1.0, 10.0)


def make_model(seed: int, extra_stop=()) -> GridSearchCV:
    """TF-IDF + balanced logistic regression; C tuned by an inner 3-fold CV on macro-F1."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN,
                                  stop_words=sorted(set(STOP_WORDS) | set(extra_stop)),
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=5000)),
    ])
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, {"lr__C": list(C_GRID)}, cv=inner, scoring="f1_macro")


def cross_validate(texts: pd.Series, y: pd.Series, labels, n_splits: int = 5, n_repeats: int = 10,
                   seed: int = 42, extra_stop=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold scores for the model and the majority baseline, plus pooled
    out-of-fold predictions (one column per repeat) for per-class metrics."""
    texts, y = texts.reset_index(drop=True), y.reset_index(drop=True)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows, oof = [], pd.DataFrame(index=y.index)
    for k, (tr, te) in enumerate(rskf.split(texts, y)):
        rep = k // n_splits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = make_model(seed + k, extra_stop).fit(texts[tr], y[tr])
        base = DummyClassifier(strategy="most_frequent").fit(texts[tr], y[tr])
        for name, pred in (("tfidf_logreg", model.predict(texts[te])), ("majority_class", base.predict(texts[te]))):
            rows.append({"fold": k, "repeat": rep, "model": name, "n_train": len(tr), "n_test": len(te),
                         **ev.scores(y[te], pred, labels),
                         "C": model.best_params_["lr__C"] if name == "tfidf_logreg" else np.nan})
            if name == "tfidf_logreg":
                oof.loc[te, f"rep{rep}"] = pred
    return pd.DataFrame(rows), oof


def corrected_ttest(diff: np.ndarray, n_train: int, n_test: int) -> tuple[float, float, float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on per-fold differences.

    Returns mean difference, its 95% interval and the two-sided p-value."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    j = len(diff)
    var = np.var(diff, ddof=1) * (1 / j + n_test / n_train)
    se = np.sqrt(var)
    t = diff.mean() / se if se > 0 else np.inf * np.sign(diff.mean())
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    half = stats.t.ppf(0.975, df=j - 1) * se
    return float(diff.mean()), float(diff.mean() - half), float(diff.mean() + half), float(p)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over folds per model, and the corrected test of model minus baseline."""
    out = []
    for metric in ("accuracy", "macro_f1", "kappa"):
        wide = folds.pivot_table(index="fold", columns="model", values=metric)
        for m in wide.columns:
            out.append({"metric": metric, "model": m, "mean": wide[m].mean(), "sd": wide[m].std()})
        n_tr, n_te = int(folds["n_train"].mean()), int(folds["n_test"].mean())
        d, lo, hi, p = corrected_ttest((wide["tfidf_logreg"] - wide["majority_class"]).to_numpy(), n_tr, n_te)
        out.append({"metric": metric, "model": "difference (model - baseline)", "mean": d, "lo": lo, "hi": hi,
                    "p_corrected": p})
    return pd.DataFrame(out)


def pooled_per_class(y: pd.Series, oof: pd.DataFrame, labels) -> pd.DataFrame:
    """Per-class precision/recall/F1 from each repeat's out-of-fold predictions, averaged over repeats."""
    tabs = [ev.per_class(y.reset_index(drop=True), oof[c], labels) for c in oof.columns]
    t = pd.concat(tabs).groupby("label", sort=False)[["precision", "recall", "f1"]].mean()
    return t.join(tabs[0].set_index("label")["support"]).reset_index()


def top_terms(texts: pd.Series, y: pd.Series, seed: int = 42, n: int = 10, extra_stop=()) -> pd.DataFrame:
    """Highest-weight terms per class from a model fitted on all labelled articles."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = make_model(seed, extra_stop).fit(texts, y).best_estimator_
    vocab = np.array(m.named_steps["tfidf"].get_feature_names_out())
    lr = m.named_steps["lr"]
    coef = lr.coef_ if len(lr.classes_) > 2 else np.vstack([-lr.coef_[0], lr.coef_[0]])
    return pd.DataFrame([{"label": c, "top_terms": ", ".join(vocab[np.argsort(-coef[i])[:n]])}
                         for i, c in enumerate(lr.classes_)])
