"""Task 2c: is gas discussed or only listed? (rule-based, INDICATIVE until validated)

Classifies each article's gas mentions with energy_nlp.mentions and splits the
rise in gas-mentioning articles by mention type, overall and within the topics
of both models. Run after 02_topics_nmf.py and 02b_topics_bertopic.py:
    nlp/.venv/Scripts/python nlp/scripts/02c_gas_mentions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energy_nlp import DATA_DIR, FIGURES_DIR, SEED, TABLES_DIR  # noqa: E402
from energy_nlp import mentions as mn  # noqa: E402
from energy_nlp import prep, topics as tp  # noqa: E402
from energy_nlp.plotting import BLUE, INK, ORANGE, SURFACE, AQUA, style, year_ticks  # noqa: E402

ERA_A, ERA_B = "2022Q3-2024Q4", "2025Q1-2026Q3"
ERA_EDGES = (pd.Period("2024Q4", "Q"), pd.Period("2026Q3", "Q"))
N_BOOT = 1000
KINDS = ["substantive", "energy_mix_list", "resource_list"]
KIND_LABEL = {"substantive": "gas discussed", "energy_mix_list": "only in energy-mix lists",
              "resource_list": "only in resource lists", "none": "no gas mention"}


def plot(quarterly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = quarterly["quarter"].dt.start_time + pd.Timedelta(days=45)
    listed = quarterly["energy_mix_list"] + quarterly["resource_list"]
    for y, colour, label in [(quarterly["substantive"], BLUE, "gas discussed (≥1 sentence about gas)"),
                             (listed, ORANGE, "gas only listed (energy-mix or resource lists)"),
                             (quarterly["resource_list"], AQUA, "  of which resource lists")]:
        ax.plot(x, y, color=colour, linewidth=2, label=label)
        ax.scatter(x, y, s=12, color=colour, zorder=3)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Share of all articles, by how gas is mentioned (rule-based; indicative until validated)",
                 loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=3)
    style(ax)
    year_ticks(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    d = prep.load_corpus(DATA_DIR / "corpus.csv")
    texts = d["title"] + ". " + d["text"]
    d["gas_mention"] = texts.map(mn.article_mention)
    q = d["quarter"]
    eras = pd.Series(np.select([q <= ERA_EDGES[0], q <= ERA_EDGES[1]], [ERA_A, ERA_B], "partial"))
    d[["post_id", "gas_mention"]].to_csv(DATA_DIR / "gas_mention_rule.csv", index=False)

    # Split of the rise in "mentions gas" by mention type (sums to the total change).
    mentions_gas = d["gas_mention"] != "none"
    split = tp.share_change_with_intervals(mentions_gas, d["gas_mention"], eras, ERA_A, ERA_B, N_BOOT, SEED)
    split = split[split["group"] != "none"].rename(columns={"group": "gas_mention"})
    split.insert(1, "label", split["gas_mention"].map(KIND_LABEL))
    split.round(4).to_csv(TABLES_DIR / "gas_mention_change_by_type.csv", index=False)

    full_q = q <= ERA_EDGES[1]
    quarterly = (pd.crosstab(q[full_q], d.loc[full_q, "gas_mention"], normalize="index")
                 .reindex(columns=KINDS + ["none"], fill_value=0).reset_index())
    quarterly.assign(quarter=quarterly["quarter"].astype(str)).round(4).to_csv(
        TABLES_DIR / "gas_mention_quarterly.csv", index=False)
    plot(quarterly, FIGURES_DIR / "gas_mention_types.png")

    # Within each model's topics: is the gas rise "listed" or "discussed"?
    nmf = pd.read_csv(DATA_DIR / "nmf_doc_topics.csv")
    bt = pd.read_csv(DATA_DIR / "bertopic_doc_topics.csv")
    nmf_lab = pd.read_csv(TABLES_DIR / "nmf_topics.csv").set_index("topic")["label"]
    bt_lab = pd.read_csv(TABLES_DIR / "bertopic_topics.csv").set_index("topic")["label"]
    rows = []
    for model, topic, labels in [("NMF", nmf["dominant"], nmf_lab), ("BERTopic", bt["topic"], bt_lab)]:
        for kind in KINDS:
            c = tp.share_change_by_group(d["gas_mention"] == kind, topic, eras, ERA_A, ERA_B)
            for _, r in c.iterrows():
                rows.append((model, labels[r["group"]], kind, r["contribution"]))
    by_topic = (pd.DataFrame(rows, columns=["model", "topic", "gas_mention", "contribution"])
                .pivot_table(index=["model", "topic"], columns="gas_mention", values="contribution",
                             fill_value=0.0).reindex(columns=KINDS).round(4))
    by_topic["total"] = by_topic.sum(axis=1)
    by_topic = by_topic.reset_index().sort_values(["model", "total"], ascending=[False, False])
    by_topic.to_csv(TABLES_DIR / "gas_mention_change_by_topic.csv", index=False)

    # Among articles that mention gas, what share actually discuss it? (bootstrap the change)
    disc = (d["gas_mention"] == "substantive").to_numpy()
    ment = mentions_gas.to_numpy()
    ia, ib = np.flatnonzero((eras == ERA_A) & ment), np.flatnonzero((eras == ERA_B) & ment)
    rng = np.random.default_rng(SEED)
    boot = [disc[rng.choice(ib, len(ib))].mean() - disc[rng.choice(ia, len(ia))].mean() for _ in range(N_BOOT)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    ratio = pd.DataFrame([{
        "measure": "share of gas-mentioning articles that discuss gas",
        ERA_A: disc[ia].mean(), ERA_B: disc[ib].mean(), "change": disc[ib].mean() - disc[ia].mean(),
        "lo": lo, "hi": hi, "n_" + ERA_A: len(ia), "n_" + ERA_B: len(ib)}]).round(4)
    ratio.to_csv(TABLES_DIR / "gas_discussed_share_of_mentions.csv", index=False)
    print(ratio.to_string(index=False), "\n")

    era_tbl = pd.crosstab(eras[eras != "partial"], d.loc[eras != "partial", "gas_mention"],
                          normalize="index").reindex(columns=KINDS + ["none"]).round(4)
    print(era_tbl.to_string())
    print("\n", split[["label", ERA_A, ERA_B, "contribution", "lo", "hi"]].round(4).to_string(index=False))
    for model in ("NMF", "BERTopic"):
        print(f"\n{model}: top contributors\n",
              by_topic[by_topic["model"] == model].head(6).drop(columns="model").to_string(index=False))


if __name__ == "__main__":
    main()
