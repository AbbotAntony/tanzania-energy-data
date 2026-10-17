"""Task 5: does coverage track or anticipate the gas-to-hydro shift? (descriptive)

Run from the repo root:  nlp/.venv/Scripts/python nlp/scripts/05_link_energy.py
Uses topic tags and keyword counts only (no tone results).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energy_nlp import DATA_DIR, FIGURES_DIR, ROOT, TABLES_DIR  # noqa: E402
from energy_nlp import link, prep  # noqa: E402
from energy_nlp.plotting import BLUE, GRID, INK, INK2, ORANGE, SURFACE, style, year_ticks  # noqa: E402

BOT = ROOT / "data" / "processed" / "bot_gas.csv"
# Author's GASCO throughput (MMscf), supplied for comparison; not part of this repository's data.
GASCO = {2023: 49_148, 2024: 42_017}
LAST_FULL_COVERAGE_Q = pd.Period("2026Q3", "Q")
MEASURES = {"jnhpp_focus_share": "JNHPP focus", "jnhpp_mention_share": "JNHPP mention",
            "gas_focus_share": "gas focus", "gas_mention_share": "gas mention"}


def annual_changes(bot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [("gas_total", "BoT total consumption"), ("gas_power", "BoT gas for power")]:
        cy = link.calendar_year_totals(bot[col])
        rows.append((label, "calendar year", "2023", "2024", cy[2023], cy[2024]))
    g = pd.read_csv(BOT)
    fy = g[(g["x_period_type"] == "fiscal_year") & (g["latest_edition"] == 1) & (g["variable"] == "gas_consumption")]
    fy = fy.pivot_table(index="fy", columns="category", values="value_mmscf", aggfunc="first")
    for col, label in [("TOTAL", "BoT total consumption"), ("Power generating plants", "BoT gas for power")]:
        for a, b in [("2022/23", "2023/24"), ("2023/24", "2024/25")]:
            rows.append((label, "fiscal year (Jul-Jun)", a, b, fy.at[a, col], fy.at[b, col]))
    rows.append(("GASCO throughput (author's data)", "calendar year", "2023", "2024", GASCO[2023], GASCO[2024]))
    out = pd.DataFrame(rows, columns=["series", "basis", "from", "to", "value_from_mmscf", "value_to_mmscf"])
    out["change_pct"] = (100 * (out["value_to_mmscf"] / out["value_from_mmscf"] - 1)).round(1)
    return out


def first_break(bot: pd.DataFrame) -> str:
    """First quarter from 2024 on whose gas-for-power value is below the 2023 mean minus 2 SD."""
    base = bot.loc[bot.index.year == 2023, "gas_power"]
    later = bot[(bot.index.year >= 2024) & (bot["gas_power"] < base.mean() - 2 * base.std())]
    return str(later.index[0]) if len(later) else "none"


def plot(merged: pd.DataFrame, milestones: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1, 1]})
    x = merged.index.start_time + pd.Timedelta(days=45)
    ax = axes[0]
    for col, colour, label in [("gas_total", INK2, "total consumption"), ("gas_power", BLUE, "gas for power")]:
        ax.plot(x, merged[col], color=colour, linewidth=2, label=label)
        ok = ~merged["repeat_of_previous"]
        ax.scatter(x[ok], merged.loc[ok, col], s=14, color=colour, zorder=3)
        ax.scatter(x[~ok], merged.loc[~ok, col], s=22, facecolors=SURFACE, edgecolors=colour, zorder=3)
    ax.set_ylim(0, 25_000)
    ax.yaxis.set_major_locator(MultipleLocator(5_000))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.set_title("Natural gas consumption, MMscf per quarter (BoT/TPDC; hollow = value repeats previous "
                 "quarter in source)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    # One colour per subject in every panel (gas = blue, JNHPP = orange); solid = mention, dashed = focus.
    for ax, colour, cols, title in [
        (axes[1], ORANGE, [("jnhpp_mention_share", "mentions JNHPP"), ("jnhpp_focus_share", "JNHPP is a focus")],
         "Share of energy articles about JNHPP"),
        (axes[2], BLUE, [("gas_mention_share", "mentions natural gas"), ("gas_focus_share", "gas is a focus")],
         "Share of energy articles about natural gas"),
    ]:
        for (col, label), dash in zip(cols, ("-", "--")):
            ax.plot(x, merged[col], color=colour, linewidth=2, linestyle=dash, label=label)
            ax.scatter(x, merged[col], s=12, color=colour, zorder=3)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title(title + " (focus = in title or ≥3 mentions)", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=2)

    for i, (_, m) in enumerate(milestones.iterrows(), start=1):
        t = pd.Timestamp(m["date"])
        for ax in axes:
            ax.axvline(t, color=GRID, linewidth=1.2, zorder=0)
        axes[0].annotate(str(i), (t, 0.97), xycoords=("data", "axes fraction"), ha="center", va="top",
                         fontsize=8, color=INK, bbox=dict(boxstyle="circle,pad=0.2", fc=SURFACE, ec=GRID))
    for ax in axes:
        style(ax)
        year_ticks(ax)
    key = "   ".join(f"{i} {pd.Timestamp(m['date']):%b %Y}: {m['event']}"
                     for i, (_, m) in enumerate(milestones.iterrows(), start=1))
    fig.text(0.01, 0.005, "JNHPP milestones (from Daily News reports): " + key, fontsize=6.5, color=INK2,
             wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    corpus = prep.load_corpus(DATA_DIR / "corpus.csv")
    full = prep.load_corpus(DATA_DIR / "corpus.csv", analysis_only=False)
    cov = link.coverage_quarterly(corpus)
    cov = cov[cov.index <= LAST_FULL_COVERAGE_Q]
    bot = link.bot_quarterly(BOT)
    merged = bot.join(cov, how="outer")
    merged.index.name = "quarter"
    merged["repeat_of_previous"] = merged["repeat_of_previous"].astype("boolean").fillna(False).astype(bool)
    merged.assign(quarter=merged.index.astype(str)).round(4).to_csv(TABLES_DIR / "link_quarterly.csv",
                                                                   index=False)

    milestones = link.milestone_table(full)
    milestones.to_csv(TABLES_DIR / "jnhpp_milestones.csv", index=False)
    annual = annual_changes(bot)
    annual.to_csv(TABLES_DIR / "link_annual_change.csv", index=False)

    # Lead/lag correlations of quarter-on-quarter changes, gas for power vs coverage.
    rows = []
    # Changes are taken on the full quarterly index, so a gap gives NaN, not a jump.
    for sample, gas in [("all quarters", merged["gas_power"]),
                        ("repeat quarter set missing", merged["gas_power"].where(~merged["repeat_of_previous"]))]:
        for col, label in MEASURES.items():
            lc = link.lagged_correlations(merged[col], gas)
            rows.append(lc.assign(coverage_measure=label, gas_series="gas for power", sample=sample))
    lags = pd.concat(rows)[["sample", "coverage_measure", "gas_series", "lag_quarters", "r", "n", "r_crit_5pct"]]
    # 4 measures x 7 lags = 28 tests per sample: ~1.4 exceed the 5% threshold by chance alone.
    lags["exceeds_5pct"] = lags["r"].abs() > lags["r_crit_5pct"]
    lags.round(3).to_csv(TABLES_DIR / "link_lag_correlations.csv", index=False)

    # Timing summary.
    dpow = bot["gas_power"].diff()
    timing = pd.DataFrame([
        ("largest quarter-on-quarter fall in gas for power", str(dpow.idxmin()), round(float(dpow.min()), 1)),
        ("first quarter from 2024 with gas for power below 2023 mean - 2 SD", first_break(bot), np.nan),
        ("peak JNHPP-mention share", str(cov["jnhpp_mention_share"].idxmax()),
         round(float(cov["jnhpp_mention_share"].max()), 3)),
        ("peak JNHPP-focus share", str(cov["jnhpp_focus_share"].idxmax()),
         round(float(cov["jnhpp_focus_share"].max()), 3)),
    ], columns=["event", "quarter", "value"])
    timing.to_csv(TABLES_DIR / "link_timing.csv", index=False)

    plot(merged[merged.index >= pd.Period("2022Q1", "Q")], milestones, FIGURES_DIR / "link_gas_jnhpp.png")

    print(annual.to_string(index=False))
    print("\n", timing.to_string(index=False))
    show = lags[(lags["sample"] == "all quarters")].pivot_table(index="coverage_measure", columns="lag_quarters",
                                                                 values="r").round(2)
    print("\nr(change in coverage at t-k, change in gas for power at t); n =",
          int(lags["n"].max()), " 5% threshold ~", round(float(lags["r_crit_5pct"].min()), 2))
    print(show.to_string())


if __name__ == "__main__":
    main()
