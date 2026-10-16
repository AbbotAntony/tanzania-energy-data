"""Task 2a: NMF topic baseline, topic shares over time, and the gas-coverage drill-down.

Run from the repo root:  nlp/.venv/Scripts/python nlp/scripts/02_topics_nmf.py [--skip-sweep]
--skip-sweep reuses nlp/tables/nmf_k_selection.csv (the sweep takes ~9 minutes).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from energy_nlp import DATA_DIR, FIGURES_DIR, SEED, TABLES_DIR  # noqa: E402
from energy_nlp import prep, topics as tp  # noqa: E402
from energy_nlp.plotting import (BLUE, BLUE_LIGHT, GRAY, GRID, INK, INK2, ORANGE,  # noqa: E402
                                 SURFACE, style, year_ticks)
from parsers.news import TOPICS  # noqa: E402

K_RANGE = range(5, 26)
# k is fixed, not chosen by the sweep. Over k = 10-25 coherence is flat and the
# stability score moves as much between two runs (SD ~0.02) as it differs across k,
# so the sweep cannot pick an optimum. 17 lies on that plateau and separates the
# energy sub-themes the research questions need (LNG, CNG, exploration, gas fields).
# The sweep is reported as a diagnostic and key results are re-run at K_SENSITIVITY.
K_FINAL = 17
K_SENSITIVITY = (14, 15, 17, 19, 21)
N_RESTARTS = 10
N_TOP = 10
N_BOOT = 1000
# Gas drill-down periods: before vs during the rise; the partial quarter is left out.
ERA_A, ERA_B = "2022Q3-2024Q4", "2025Q1-2026Q3"
ERA_EDGES = (pd.Period("2024Q4", "Q"), pd.Period("2026Q3", "Q"))

# Plain-word labels for the k=17 model. The anchor term must be among the topic's
# top terms, so a refit that reorders topics fails loudly instead of mislabelling.
LABELS = {
    0: ("Industrialisation & regional growth", "industrial"),
    1: ("Upstream exploration & licensing", "exploration"),
    2: ("Electricity grid & generation", "electricity"),
    3: ("CNG for vehicles", "cng"),
    4: ("Budget & public finance", "budget"),
    5: ("Clean cooking & climate", "clean cooking"),
    6: ("EACOP crude oil pipeline", "eacop"),
    7: ("Diplomacy, trade & investment ties", "bilateral"),
    8: ("Fuel prices & Middle East shocks", "fuel prices"),
    9: ("Gas fields & production", "wells"),
    10: ("Ruling party & elections", "election"),
    11: ("JNHPP dam & Rufiji water", "rufiji"),
    12: ("Public enterprises & Treasury Registrar", "treasury registrar"),
    13: ("Local communities & CSR", "csr"),
    14: ("LNG project", "lng"),
    15: ("Zanzibar-Union affairs", "union"),
    16: ("Macroeconomic figures", "gdp"),
}



def sweep(X, k_range=K_RANGE) -> pd.DataFrame:
    def score(k):
        warnings.simplefilter("ignore", ConvergenceWarning)
        _, H = tp.fit_nmf(X, k, SEED)
        top = tp.top_term_idx(H, N_TOP)
        Hs = [tp.fit_nmf(X, k, s, init="random")[1] for s in range(N_RESTARTS)]
        return k, tp.npmi_coherence(top, X).mean(), tp.topic_diversity(top), tp.stability(Hs)
    rows = Parallel(n_jobs=-1, max_nbytes=None)(delayed(score)(k) for k in k_range)
    return pd.DataFrame(rows, columns=["k", "npmi", "diversity", "stability"]).round(4)


def era(q: pd.Series) -> pd.Series:
    return pd.Series(np.select([q <= ERA_EDGES[0], q <= ERA_EDGES[1]], [ERA_A, ERA_B], "partial"),
                     index=q.index)


def plot_k_selection(sel: pd.DataFrame, k: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), sharex=True)
    for ax, col, title in zip(axes, ["npmi", "stability", "diversity"],
                              ["Coherence (NPMI)", "Stability across 10 restarts", "Topic diversity"]):
        ax.plot(sel["k"], sel[col], color=BLUE, linewidth=2)
        ax.scatter(sel["k"], sel[col], s=14, color=BLUE, zorder=3)
        ax.scatter([k], sel.loc[sel["k"] == k, col], s=60, facecolors="none", edgecolors=INK, zorder=4)
        ax.set_title(title, loc="left", fontsize=9)
        style(ax)
    axes[1].set_xlabel("number of topics k (circled: k used, fixed at 17; diagnostic only)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_shares(shares: pd.DataFrame, robust: pd.DataFrame, order: list[int], path: Path) -> None:
    n = len(order)
    cols = 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.0 * rows), sharex=True, sharey=True)
    ymax = shares["hi"].max() * 1.05
    for ax, t in zip(axes.flat, order):
        s = shares[shares["topic"] == t]
        r = robust[robust["topic"] == t]
        x = s["period"].dt.start_time + pd.Timedelta(days=45)
        ax.fill_between(x, s["lo"], s["hi"], color=BLUE_LIGHT, linewidth=0)
        ax.plot(x, s["share"], color=BLUE, linewidth=2)
        ax.plot(r["period"].dt.start_time + pd.Timedelta(days=45), r["share"], color=GRAY,
                linewidth=1.2, linestyle="--")
        ax.set_title(LABELS[t][0], loc="left", fontsize=8)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        style(ax)
        year_ticks(ax)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2),
               plt.Rectangle((0, 0), 1, 1, color=BLUE_LIGHT),
               plt.Line2D([], [], color=GRAY, linewidth=1.2, linestyle="--")]
    list(axes.flat)[-1].legend(handles, ["all articles", "95% bootstrap interval",
                                         "excluding passing mentions"],
                               loc="center", frameon=False, fontsize=8)
    fig.suptitle("NMF topic shares by quarter (mean article proportion), 2022-Q3 to 2026-Q3",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_gas(focus_q: pd.DataFrame, contrib: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    x = focus_q["quarter"].dt.start_time + pd.Timedelta(days=45)
    for col, colour, label in [("gas_share", BLUE, "any mention"),
                               ("gas_focus_share", ORANGE, "gas is a focus (title or ≥3 mentions)")]:
        ax1.plot(x, focus_q[col], color=colour, linewidth=2, label=label)
        ax1.scatter(x, focus_q[col], s=14, color=colour, zorder=3)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.set_title("Share of articles mentioning natural gas", loc="left", fontsize=9)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax1)
    year_ticks(ax1)

    c = contrib.iloc[::-1]
    y = np.arange(len(c))
    ax2.barh(y, c["contribution"] * 100, color=BLUE, height=0.7)
    ax2.set_yticks(y, c["label"], fontsize=7)
    ax2.axvline(0, color=INK2, linewidth=0.8)
    lim = np.abs(c["contribution"]).max() * 100
    ax2.set_xlim(-lim * 0.6, lim * 1.15)
    for yi, v in zip(y, c["contribution"] * 100):
        if abs(v) >= 1:
            ax2.text(v + (0.15 if v > 0 else -0.15), yi, f"{v:+.1f}", va="center",
                     ha="left" if v > 0 else "right", fontsize=7, color=INK)
    ax2.set_xlabel("change in share of all articles that mention gas, percentage points", fontsize=7)
    ax2.set_title(f"Where the rise came from, by dominant topic\n{ERA_A} vs {ERA_B}",
                  loc="left", fontsize=9)
    style(ax2)
    ax2.grid(axis="x", color=GRID, linewidth=0.6)
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("ignore", ConvergenceWarning)

    d = prep.load_corpus(DATA_DIR / "corpus.csv")
    texts = d["title"] + ". " + d["text"]
    names = tp.person_names(texts)
    vec = tp.make_vectorizer(names)
    X = vec.fit_transform(texts)
    X.sort_indices()
    vocab = np.array(vec.get_feature_names_out())
    pd.Series(sorted(names), name="name").to_csv(TABLES_DIR / "nmf_stop_names.csv", index=False)
    print(f"{X.shape[0]} articles, {X.shape[1]} terms, {len(names)} person names removed")

    sel_path = TABLES_DIR / "nmf_k_selection.csv"
    sel = pd.read_csv(sel_path) if args.skip_sweep else sweep(X)
    sel.to_csv(sel_path, index=False)
    k = K_FINAL
    plot_k_selection(sel, k, FIGURES_DIR / "nmf_k_selection.png")

    W, H = tp.fit_nmf(X, k, SEED)
    P = tp.doc_proportions(W)
    top = tp.top_term_idx(H, 15)
    for t, (label, anchor) in LABELS.items():
        if anchor not in vocab[top[t]]:
            sys.exit(f"topic {t} ({label}) lost anchor term {anchor!r}: {list(vocab[top[t]])}")
    npmi = tp.npmi_coherence(tp.top_term_idx(H, N_TOP), X)
    dominant = P.argmax(axis=1)
    topic_table = pd.DataFrame({
        "topic": range(k), "label": [LABELS[t][0] for t in range(k)],
        "mean_share": P.mean(axis=0).round(4),
        "n_dominant": np.bincount(dominant, minlength=k),
        "npmi": npmi.round(3),
        "top_terms": [", ".join(vocab[row]) for row in top],
    }).sort_values("mean_share", ascending=False)
    topic_table.to_csv(TABLES_DIR / "nmf_topics.csv", index=False)

    doc_topics = pd.DataFrame(P, columns=[f"t{t}" for t in range(k)]).assign(
        post_id=d["post_id"], quarter=d["quarter"].astype(str), dominant=dominant)
    doc_topics.to_csv(DATA_DIR / "nmf_doc_topics.csv", index=False)

    # Topic shares by quarter, all articles and without passing mentions.
    full_q = d["quarter"] <= ERA_EDGES[1]
    shares = tp.period_shares(P[full_q], d.loc[full_q, "quarter"], N_BOOT, SEED)
    keep = full_q & ~d["passing_mention"]
    robust = tp.period_shares(P[keep], d.loc[keep, "quarter"], N_BOOT, SEED)
    out = shares.merge(robust[["period", "topic", "share", "n_articles"]], on=["period", "topic"],
                       suffixes=("", "_excl_passing"))
    out.insert(2, "label", out["topic"].map(lambda t: LABELS[t][0]))
    out.assign(period=out["period"].astype(str)).round(4).to_csv(
        TABLES_DIR / "nmf_topic_shares_quarterly.csv", index=False)
    plot_shares(shares, robust, topic_table["topic"].tolist(), FIGURES_DIR / "nmf_topic_shares.png")

    # Robustness summary: does dropping passing mentions change the trend?
    eras = era(d["quarter"])
    trend = []
    for name, mask in [("all", np.ones(len(d), bool)), ("excl_passing", ~d["passing_mention"].to_numpy())]:
        for t in range(k):
            a = P[mask & (eras == ERA_A).to_numpy(), t].mean()
            b = P[mask & (eras == ERA_B).to_numpy(), t].mean()
            trend.append((t, LABELS[t][0], name, a, b, b - a))
    trend = pd.DataFrame(trend, columns=["topic", "label", "sample", ERA_A, ERA_B, "change"])
    trend = trend.pivot_table(index=["topic", "label"], columns="sample",
                              values=[ERA_A, ERA_B, "change"]).round(4)
    trend.columns = [f"{a}_{b}" for a, b in trend.columns]
    trend = trend.reset_index()
    trend["same_direction"] = np.sign(trend["change_all"]) == np.sign(trend["change_excl_passing"])
    trend.sort_values("change_all", ascending=False).to_csv(TABLES_DIR / "nmf_topic_change_by_era.csv",
                                                            index=False)

    # Gas drill-down: what drives the rise in gas mentions?
    gas_rx = TOPICS["natural_gas"]
    gas = d["topics"].str.split(";").map(lambda ts: "natural_gas" in ts)
    gas_hits = texts.map(lambda s: len(gas_rx.findall(s)))
    gas_focus = gas & (d["title"].str.contains(gas_rx) | (gas_hits >= 3))
    focus_q = (pd.DataFrame({"quarter": d["quarter"], "gas": gas, "focus": gas_focus,
                             "passing": gas & (gas_hits == 1) & ~d["title"].str.contains(gas_rx)})
               [full_q].groupby("quarter")
               .agg(n_articles=("gas", "size"), gas_share=("gas", "mean"),
                    gas_focus_share=("focus", "mean"), gas_passing_share=("passing", "mean"))
               .reset_index())
    focus_q.assign(quarter=focus_q["quarter"].astype(str)).round(4).to_csv(
        TABLES_DIR / "gas_share_quarterly.csv", index=False)
    era_tbl = pd.DataFrame({"era": eras, "gas": gas, "focus": gas_focus,
                            "passing": gas & (gas_hits == 1) & ~d["title"].str.contains(gas_rx)})
    era_tbl = era_tbl[era_tbl["era"] != "partial"].groupby("era").agg(
        n_articles=("gas", "size"), gas_share=("gas", "mean"), gas_focus_share=("focus", "mean"),
        gas_passing_share=("passing", "mean")).round(4)
    era_tbl.to_csv(TABLES_DIR / "gas_share_by_era.csv")

    contrib = tp.share_change_with_intervals(gas, pd.Series(dominant), eras, ERA_A, ERA_B, N_BOOT, SEED)
    contrib.insert(1, "label", contrib["group"].map(lambda t: LABELS[t][0]))
    contrib = contrib.rename(columns={"group": "topic"}).round(4)
    contrib.to_csv(TABLES_DIR / "gas_rise_by_topic.csv", index=False)

    # k-sensitivity: redo the drill-down at other k; name each topic by its closest
    # k=17 topic (cosine of term weights) so contributions line up across k.
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    sens = []
    for kk in K_SENSITIVITY:
        Wk, Hk = (W, H) if kk == k else tp.fit_nmf(X, kk, SEED)
        dom_k = tp.doc_proportions(Wk).argmax(axis=1)
        ck = tp.share_change_by_group(gas, pd.Series(dom_k), eras, ERA_A, ERA_B).set_index("group")
        sim = (Hk / np.linalg.norm(Hk, axis=1, keepdims=True)) @ Hn.T
        for t in range(kk):
            sens.append((kk, t, LABELS[int(sim[t].argmax())][0], round(float(sim[t].max()), 3),
                         round(float(ck["contribution"].get(t, 0.0)), 4),
                         ", ".join(vocab[np.argsort(-Hk[t])[:8]])))
    sens = pd.DataFrame(sens, columns=["k", "topic", "closest_k17_topic", "cosine",
                                       "contribution", "top_terms"])
    sens.sort_values(["k", "contribution"], ascending=[True, False]).to_csv(
        TABLES_DIR / "nmf_k_sensitivity.csv", index=False)
    focus = ["Industrialisation & regional growth", "Clean cooking & climate",
             "Fuel prices & Middle East shocks", "LNG project", "CNG for vehicles",
             "Gas fields & production"]
    # Several topics can share a closest k=17 topic; sum them so each row is a theme.
    sens_summary = (sens[sens["closest_k17_topic"].isin(focus)]
                    .pivot_table(index="closest_k17_topic", columns="k", values="contribution",
                                 aggfunc="sum").reindex(focus).round(4))
    sens_summary.to_csv(TABLES_DIR / "nmf_k_sensitivity_summary.csv")
    plot_gas(focus_q, contrib, FIGURES_DIR / "gas_rise_drivers.png")

    print(topic_table[["topic", "label", "mean_share", "n_dominant", "npmi"]].to_string(index=False))
    print("\n", era_tbl.to_string())
    print("\n", contrib.to_string(index=False))
    print("\nk-sensitivity (contribution to gas rise, share points):\n", sens_summary.to_string())
    print("\n", trend[["label", "change_all", "change_excl_passing", "same_direction"]]
          .sort_values("change_all").to_string(index=False))


if __name__ == "__main__":
    main()
