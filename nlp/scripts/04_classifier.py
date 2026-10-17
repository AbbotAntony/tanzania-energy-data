"""Task 4: TF-IDF + logistic regression on the hand labels (framing; positive vs neutral tone).

Run from the repo root after importing round-1 labels:
    nlp/.venv/Scripts/python nlp/scripts/04_classifier.py
Evaluation: 5-fold stratified CV repeated 10 times; majority-class baseline;
Nadeau-Bengio corrected t-test for the model-baseline difference. No separate
held-out test set: with ~70 tone labels it would leave too little to train on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energy_nlp import DATA_DIR, FIGURES_DIR, NLP_DIR, SEED, TABLES_DIR  # noqa: E402
from energy_nlp import classify as cl  # noqa: E402
from energy_nlp import evaluate as ev  # noqa: E402
from energy_nlp import prep, sentiment as se, topics as tp  # noqa: E402
from energy_nlp.labels import CODES  # noqa: E402
from energy_nlp.plotting import BLUE, GRAY, GRID, INK2, SURFACE, style  # noqa: E402

FRAMES = CODES["frame"]
TONES2 = ("neutral", "positive")


def plot(folds: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(folds), figsize=(8, 3.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (task, f) in zip(axes, folds.items()):
        for i, (model, colour, label) in enumerate([("majority_class", GRAY, "majority\nbaseline"),
                                                    ("tfidf_logreg", BLUE, "TF-IDF +\nlogistic reg.")]):
            v = f.loc[f["model"] == model, "macro_f1"].to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=10, color=colour, alpha=0.6,
                       edgecolors="none")
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color=colour, linewidth=2.5)
            ax.text(i + 0.28, v.mean(), f"{v.mean():.2f}", va="center", fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ["majority\nbaseline", "TF-IDF +\nlogistic reg."], fontsize=8)
        ax.set_xlim(-0.5, 1.7)
        ax.set_ylim(0, 1)
        ax.set_title(task, loc="left", fontsize=9)
        style(ax)
    axes[0].set_ylabel("macro-F1 per fold (50 folds)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    lab = pd.read_csv(NLP_DIR / "labels" / "labels_round1.csv")
    corpus = prep.load_corpus(DATA_DIR / "corpus.csv")
    names = tp.person_names(corpus["title"] + ". " + corpus["text"])
    d = lab.merge(corpus[["post_id", "title", "text"]], on="post_id", how="left")
    assert d["text"].notna().all(), "every labelled article must be in the corpus"
    full = d["title"] + ". " + d["text"]

    # Frame: all labelled articles, whole text.
    frame_folds, frame_oof = cl.cross_validate(full, d["frame"], FRAMES, seed=SEED, extra_stop=names)
    # Tone: articles coded as discussing gas, positive vs neutral, on their gas sentences.
    t = d[(d["gas_mention"] == "substantive") & d["gas_tone"].isin(TONES2)].reset_index(drop=True)
    gas_text = [se.gas_sentences(f"{a}. {b}") for a, b in zip(t["title"], t["text"])]
    fallback = sum(1 for g in gas_text if not g)
    t_text = pd.Series([" ".join(g) if g else f"{a}. {b}" for g, a, b in zip(gas_text, t["title"], t["text"])])
    tone_folds, tone_oof = cl.cross_validate(t_text, t["gas_tone"], TONES2, seed=SEED, extra_stop=names)

    summary = pd.concat([cl.summarise(frame_folds).assign(task="frame (8 classes)", n=len(d)),
                         cl.summarise(tone_folds).assign(task="tone: positive vs neutral", n=len(t))])
    # Reference: the untrained zero-shot model on the same tone articles (no training, no CV).
    zs = pd.read_csv(DATA_DIR / "sentiment_predictions.csv")[["post_id", "zeroshot_label"]]
    tz = t.merge(zs, on="post_id", how="left")
    covered = tz["zeroshot_label"].notna()
    ref = ev.scores(tz.loc[covered, "gas_tone"], tz.loc[covered, "zeroshot_label"], TONES2)
    summary = pd.concat([summary, pd.DataFrame(
        [{"metric": k, "model": "zero-shot NLI (no training, reference)", "mean": v,
          "task": "tone: positive vs neutral", "n": int(covered.sum())} for k, v in ref.items()])])
    summary = summary[["task", "n", "metric", "model", "mean", "sd", "lo", "hi", "p_corrected"]].round(4)

    per_class = pd.concat([cl.pooled_per_class(d["frame"], frame_oof, FRAMES).assign(task="frame"),
                           cl.pooled_per_class(t["gas_tone"], tone_oof, TONES2).assign(task="tone")]).round(3)
    terms = pd.concat([cl.top_terms(full, d["frame"], SEED, extra_stop=names).assign(task="frame"),
                       cl.top_terms(t_text, t["gas_tone"], SEED, extra_stop=names).assign(task="tone")])
    folds = pd.concat([frame_folds.assign(task="frame"), tone_folds.assign(task="tone")])

    summary.to_csv(TABLES_DIR / "classifier_summary.csv", index=False)
    per_class.to_csv(TABLES_DIR / "classifier_per_class.csv", index=False)
    terms.to_csv(TABLES_DIR / "classifier_top_terms.csv", index=False)
    folds.round(4).to_csv(TABLES_DIR / "classifier_folds.csv", index=False)
    plot({"Frame (8 classes, n=150)": frame_folds, f"Tone: positive vs neutral (n={len(t)})": tone_folds},
         FIGURES_DIR / "classifier_cv.png")

    pd.set_option("display.width", 200)
    print(f"tone rows: {len(t)} ({(t['gas_tone'] == 'positive').sum()} positive); "
          f"no gas sentence found (whole text used): {fallback}; zero-shot coverage {int(covered.sum())}")
    print(summary.to_string(index=False))
    print("\n", per_class.to_string(index=False))
    print("\nchosen C:", folds.groupby("task")["C"].agg(lambda s: s.value_counts().to_dict()).to_dict())


if __name__ == "__main__":
    main()
