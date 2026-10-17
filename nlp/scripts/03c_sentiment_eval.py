"""Task 3: validate the tone models (and the gas-mention rule) against hand labels.

Run after importing round-1 labels:
    nlp/.venv/Scripts/python nlp/scripts/03c_sentiment_eval.py
Three evaluations, each with bootstrap 95% intervals and a majority-class baseline:
  A  gas-mention rule vs hand gas_mention (4 classes): validates "discussed vs listed"
  B  tone model vs hand gas_tone, on articles hand-coded substantive (3 classes)
  C  end-to-end tone (rule decides "about gas", model the tone) vs hand gas_tone (4 classes)
Each is repeated without articles the coder marked unsure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energy_nlp import DATA_DIR, FIGURES_DIR, NLP_DIR, TABLES_DIR  # noqa: E402
from energy_nlp import evaluate as ev  # noqa: E402
from energy_nlp.labels import CODES  # noqa: E402
from energy_nlp.plotting import INK, INK2, SURFACE  # noqa: E402
from energy_nlp.sentiment import TONES  # noqa: E402

MODELS = ("vader", "finnews", "zeroshot")
N_BOOT = 2000
TONE4 = CODES["gas_tone"]
MENTION4 = CODES["gas_mention"]


def evaluate_task(task: str, model: str, y_true, y_pred, labels, weights, subset: str) -> tuple[list, pd.DataFrame]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    b = ev.bootstrap_scores(y_true, y_pred, labels, N_BOOT)
    rows = [{"task": task, "model": model, "subset": subset, "n": len(y_true), **r} for r in b.to_dict("records")]
    rows.append({"task": task, "model": model, "subset": subset, "n": len(y_true),
                 "metric": "accuracy_weighted_to_corpus", "value": ev.weighted_accuracy(y_true, y_pred, weights),
                 "lo": np.nan, "hi": np.nan})
    pc = ev.per_class(y_true, y_pred, labels).assign(task=task, model=model, subset=subset)
    return rows, pc


def plot_confusions(mats: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(mats), figsize=(3.4 * len(mats), 3.4))
    for ax, (title, m) in zip(np.atleast_1d(axes), mats.items()):
        share = m.div(m.sum(axis=1).replace(0, 1), axis=0)
        ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, m.values[i, j], ha="center", va="center", fontsize=8,
                        color=SURFACE if share.values[i, j] > 0.6 else INK)
        ax.set_xticks(range(m.shape[1]), [c.replace("_", " ") for c in m.columns], rotation=45, ha="right",
                      fontsize=7)
        ax.set_yticks(range(m.shape[0]), [c.replace("_", " ") for c in m.index], fontsize=7)
        ax.set_xlabel("model", fontsize=7, color=INK2)
        ax.set_ylabel("hand label", fontsize=7, color=INK2)
        ax.set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=NLP_DIR / "labels" / "labels_round1.csv")
    ap.add_argument("--out", type=Path, default=TABLES_DIR)
    ap.add_argument("--figures", type=Path, default=FIGURES_DIR)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels)
    design = pd.read_csv(NLP_DIR / "labels" / "sample_design.csv")
    pred = pd.read_csv(DATA_DIR / "sentiment_predictions.csv")
    df = lab.merge(design[["sample_id", "weight", "year", "gas_tagged"]], on="sample_id").merge(pred, on="post_id")
    assert len(df) == len(lab), "every labelled article needs a prediction"

    rows, pcs, mats = [], [], {}
    for subset, d in [("all", df), ("excl_unsure", df[~df["unsure"]])]:
        # A: mention rule
        r, pc = evaluate_task("A_gas_mention", "rule", d["gas_mention"], d["gas_mention_rule"], MENTION4,
                              d["weight"], subset)
        rows += r; pcs.append(pc)
        majority = d["gas_mention"].mode()[0]
        r, _ = evaluate_task("A_gas_mention", "majority_class", d["gas_mention"], [majority] * len(d), MENTION4,
                             d["weight"], subset)
        rows += r
        # B: tone on hand-substantive articles that the rule also found gas sentences in
        sub = d[(d["gas_mention"] == "substantive")]
        covered = sub[sub["n_gas_sentences"] > 0]
        for m in MODELS:
            r, pc = evaluate_task("B_tone_substantive", m, covered["gas_tone"], covered[f"{m}_label"], TONES,
                                  covered["weight"], subset)
            rows += r; pcs.append(pc)
        majority = covered["gas_tone"].mode()[0]
        r, _ = evaluate_task("B_tone_substantive", "majority_class", covered["gas_tone"], [majority] * len(covered),
                             TONES, covered["weight"], subset)
        rows += r
        rows.append({"task": "B_tone_substantive", "model": "coverage", "subset": subset, "n": len(sub),
                     "metric": "share_of_substantive_with_gas_sentences",
                     "value": len(covered) / max(len(sub), 1), "lo": np.nan, "hi": np.nan})
        # C: end-to-end 4-class
        for m in MODELS:
            r, pc = evaluate_task("C_tone_end_to_end", m, d["gas_tone"], d[f"{m}_tone4"], TONE4, d["weight"], subset)
            rows += r; pcs.append(pc)
        majority = d["gas_tone"].mode()[0]
        r, _ = evaluate_task("C_tone_end_to_end", "majority_class", d["gas_tone"], [majority] * len(d), TONE4,
                             d["weight"], subset)
        rows += r
        if subset == "all":
            mats["A: gas-mention rule"] = ev.confusion(d["gas_mention"], d["gas_mention_rule"], MENTION4)
            for m in MODELS:
                mats[f"B: {m} (substantive only)"] = ev.confusion(covered["gas_tone"], covered[f"{m}_label"], TONES)

    res = pd.DataFrame(rows).round(4)
    args.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out / "sentiment_eval_scores.csv", index=False)
    pd.concat(pcs).round(4).to_csv(args.out / "sentiment_eval_per_class.csv", index=False)
    long = pd.concat([m.stack().rename("n").reset_index().assign(matrix=k) for k, m in mats.items()])
    long.to_csv(args.out / "sentiment_eval_confusion.csv", index=False)
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_confusions(mats, args.figures / "sentiment_confusion.png")

    show = res[(res["subset"] == "all") & res["metric"].isin(["accuracy", "macro_f1", "kappa"])]
    print(show.pivot_table(index=["task", "model"], columns="metric", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
