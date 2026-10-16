"""Task 1: build the clean corpus and coverage tables.

Run from the repo root:  nlp/.venv/Scripts/python nlp/scripts/01_prep.py
Writes data/nlp/corpus.csv (article text, gitignored), nlp/tables/prep_*.csv
and nlp/figures/prep_coverage.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from energy_nlp import DATA_DIR, FIGURES_DIR, ROOT, TABLES_DIR  # noqa: E402
from energy_nlp import prep  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#9ec5f4", "#eb6834"


def plot_coverage(monthly: pd.DataFrame, quarterly: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]}, facecolor=SURFACE)
    m = monthly.assign(t=pd.PeriodIndex(monthly["month"], freq="M").to_timestamp())
    thin = m["low_count"] | m["partial"]
    ax1.bar(m.loc[~thin, "t"], m.loc[~thin, "n_articles"], width=24, color=BLUE, label="months with ≥10 articles")
    ax1.bar(m.loc[thin, "t"], m.loc[thin, "n_articles"], width=24, color=BLUE_LIGHT, hatch="////",
            edgecolor=BLUE, linewidth=0, label="thin (<10) or partial month")
    for t in m.loc[m["empty"], "t"]:
        ax1.plot(t, 0, marker="x", color=INK2, markersize=4, clip_on=False)
    ax1.set_ylabel("articles per month")
    ax1.set_title("Daily News energy articles (deduplicated, on-topic)", loc="left", fontsize=10)
    ax1.legend(frameon=False, loc="upper center", fontsize=8)

    qi = pd.PeriodIndex(quarterly["quarter"], freq="Q")
    q = quarterly.assign(t=qi.start_time + (qi.end_time - qi.start_time) / 2)
    q = q[q["n_articles"] > 0]
    for tag, colour, label in [("natural_gas", BLUE, "natural gas"), ("jnhpp", ORANGE, "JNHPP")]:
        share = q[tag] / q["n_articles"]
        main = q["in_main_window"]
        ax2.plot(q.loc[main, "t"], share[main], color=colour, linewidth=2, label=label)
        ok = ~q["low_count"] & ~q["partial"]
        ax2.scatter(q.loc[ok, "t"], share[ok], s=18, color=colour, zorder=3,
                    edgecolors=SURFACE, linewidths=1.5)
        ax2.scatter(q.loc[~ok, "t"], share[~ok], s=18, facecolors=SURFACE, edgecolors=colour,
                    linewidths=1.5, zorder=3)
        ax2.annotate(label, (q["t"].iloc[-1], share.iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", color=INK, fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.set_ylabel("share of quarter's articles")
    ax2.set_title("Share of articles mentioning each keyword tag, by quarter "
                  "(hollow = thin or partial quarter)", loc="left", fontsize=10)
    ax2.legend(frameon=False, loc="upper center", ncols=2, fontsize=8)

    start = prep.MAIN_WINDOW_START.start_time
    for ax in (ax1, ax2):
        ax.axvspan(m["t"].min() - pd.Timedelta(days=20), start, color="#f1f0ec", zorder=0, linewidth=0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    left = m["t"].min() - pd.Timedelta(days=20)
    ax1.text(left + (start - left) / 2, ax1.get_ylim()[1] * 0.5, "before 2022-Q3:\ndescriptive only",
             ha="center", va="center", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    for d in (DATA_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df = prep.build_corpus(ROOT / "data/processed/news_dailynews.csv", ROOT / "data/raw/dailynews")

    # Scrape date bounds the last complete month.
    retrieved = pd.to_datetime(pd.read_csv(ROOT / "data/processed/news_dailynews.csv")["retrieved_date"]).max()
    last_month = (retrieved + pd.Timedelta(days=1)).to_period("M") - 1
    last_quarter = (retrieved + pd.Timedelta(days=1)).to_period("Q") - 1

    out = df.assign(month=df["month"].astype(str), quarter=df["quarter"].astype(str))
    out.to_csv(DATA_DIR / "corpus.csv", index=False)

    monthly = prep.counts_by_period(df, "M", last_month)
    quarterly = prep.counts_by_period(df, "Q", last_quarter)
    flow = prep.prep_flow(df)
    monthly.to_csv(TABLES_DIR / "prep_articles_per_month.csv", index=False)
    quarterly.to_csv(TABLES_DIR / "prep_articles_per_quarter.csv", index=False)
    prep.tag_counts(df).to_csv(TABLES_DIR / "prep_articles_per_tag.csv", index=False)
    flow.to_csv(TABLES_DIR / "prep_flow.csv", index=False)
    # Exclusions by id/url only - no article text in committed tables.
    excl = df.loc[df["off_topic"] | df["is_duplicate"],
                  ["post_id", "pub_date", "url", "topics_scraper", "off_topic", "is_duplicate", "dup_of", "jaccard"]]
    excl.assign(pub_date=excl["pub_date"].dt.date).to_csv(TABLES_DIR / "prep_excluded.csv", index=False)
    plot_coverage(monthly, quarterly, FIGURES_DIR / "prep_coverage.png")

    print(flow.to_string(index=False))
    flagged = monthly[monthly["low_count"] | monthly["partial"]]
    print(f"\nflagged months ({len(flagged)}):")
    print(flagged[["month", "n_articles", "empty", "partial", "in_main_window"]].to_string(index=False))


if __name__ == "__main__":
    main()
