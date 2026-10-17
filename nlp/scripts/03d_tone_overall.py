"""RQ3 (reframed): what is the overall tone of gas coverage? From the hand labels.

Off-the-shelf models failed validation (03c) and the supervised classifier cannot
separate positive from neutral (04), so tone is estimated from the hand labels:
weighted to the corpus with the sampling weights, with 95% intervals from a
bootstrap that resamples articles within each sampling stratum. The by-year
positive share is descriptive only (about 15 coded gas articles per year).

Run from the repo root:  nlp/.venv/Scripts/python nlp/scripts/03d_tone_overall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energy_nlp import FIGURES_DIR, NLP_DIR, SEED, TABLES_DIR  # noqa: E402
from energy_nlp.plotting import BLUE, GRAY, GRID, INK2, SURFACE, style  # noqa: E402

TONES = ("positive", "neutral", "negative")
N_BOOT = 2000


def tone_shares(df: pd.DataFrame) -> dict[str, float]:
    """Weighted share of each tone among articles hand-coded as discussing gas."""
    s = df[df["gas_mention"] == "substantive"]
    w = s["weight"]
    return {t: float((w * (s["gas_tone"] == t)).sum() / w.sum()) if w.sum() else np.nan for t in TONES}


def positive_share(df: pd.DataFrame) -> float:
    """Weighted positive / (positive + neutral) among gas-discussing articles."""
    s = df[(df["gas_mention"] == "substantive") & df["gas_tone"].isin(["positive", "neutral"])]
    return float((s["weight"] * (s["gas_tone"] == "positive")).sum() / s["weight"].sum()) if len(s) else np.nan


def stratified_boot(df: pd.DataFrame, fn, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("stratum")]
    out = []
    for _ in range(n_boot):
        b = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        out.append(fn(b))
    return np.array(out)


def plot(overall: pd.DataFrame, by_year: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    o = overall[overall["sample"] == "all labels"].set_index("tone").loc[list(TONES)]
    y = np.arange(len(o))[::-1]
    ax1.barh(y, o["share"] * 100, color=BLUE, height=0.6)
    ax1.hlines(y, o["lo"] * 100, o["hi"] * 100, color=INK2, linewidth=1.2)
    for yi, (_, r) in zip(y, o.iterrows()):
        ax1.text(r["hi"] * 100 + 2, yi, f"{r['share']:.0%}", va="center", fontsize=8, color=INK2)
    ax1.set_yticks(y, o.index, fontsize=8)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("% of gas-discussing articles (weighted; line = 95% CI)", fontsize=7, color=INK2)
    ax1.set_title("Tone toward natural gas, 2022-Q3 to 2026", loc="left", fontsize=9)
    style(ax1)
    ax1.grid(axis="x", color=GRID, linewidth=0.6)
    ax1.grid(axis="y", visible=False)

    b = by_year[by_year["sample"] == "all labels"]
    ax2.errorbar(b["year"], b["positive_share"] * 100,
                 yerr=[(b["positive_share"] - b["lo"]) * 100, (b["hi"] - b["positive_share"]) * 100],
                 fmt="o", color=BLUE, ecolor=GRAY, elinewidth=1.2, capsize=3, markersize=5)
    for _, r in b.iterrows():
        ax2.text(r["year"] + 0.12, r["positive_share"] * 100, f"n={int(r['n'])}", fontsize=7, color=INK2,
                 va="center")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(b["year"], [str(int(v)) + ("*" if v == 2022 else "") for v in b["year"]], fontsize=8)
    ax2.set_ylabel("% positive (of positive + neutral)", fontsize=7, color=INK2)
    ax2.set_title("By year: descriptive only (wide intervals)", loc="left", fontsize=9)
    style(ax2)
    fig.text(0.01, 0.01, "*2022: July-December only. Hand labels, weighted to the corpus.", fontsize=6.5,
             color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    lab = pd.read_csv(NLP_DIR / "labels" / "labels_round1.csv")
    design = pd.read_csv(NLP_DIR / "labels" / "sample_design.csv")
    df = lab.merge(design[["sample_id", "year", "stratum", "weight"]], on="sample_id")
    assert df["weight"].notna().all()

    overall, yearly = [], []
    for sample, d in [("all labels", df), ("excluding unsure", df[~df["unsure"]])]:
        point = tone_shares(d)
        boots = stratified_boot(d, lambda b: [tone_shares(b)[t] for t in TONES])
        lo, hi = np.nanquantile(boots, [0.025, 0.975], axis=0)
        n = int((d["gas_mention"] == "substantive").sum())
        counts = d.loc[d["gas_mention"] == "substantive", "gas_tone"].value_counts()
        for i, t in enumerate(TONES):
            overall.append({"sample": sample, "tone": t, "share": point[t], "lo": lo[i], "hi": hi[i],
                            "n_coded": int(counts.get(t, 0)), "n_gas_discussing": n})
        for year, g in d.groupby("year"):
            sub = g[(g["gas_mention"] == "substantive") & g["gas_tone"].isin(["positive", "neutral"])]
            bb = stratified_boot(g, positive_share)
            lo_y, hi_y = np.nanquantile(bb, [0.025, 0.975])
            yearly.append({"sample": sample, "year": int(year), "positive_share": positive_share(g),
                           "lo": lo_y, "hi": hi_y, "n": len(sub)})
    overall, yearly = pd.DataFrame(overall).round(4), pd.DataFrame(yearly).round(4)
    overall.to_csv(TABLES_DIR / "tone_overall.csv", index=False)
    yearly.to_csv(TABLES_DIR / "tone_by_year.csv", index=False)
    plot(overall, yearly, FIGURES_DIR / "tone_overall.png")
    print(overall.to_string(index=False))
    print("\n", yearly.to_string(index=False))


if __name__ == "__main__":
    main()
