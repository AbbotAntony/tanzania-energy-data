"""Task 2b: BERTopic (all-MiniLM-L6-v2, CPU) and comparison with the NMF baseline.

Run after 02_topics_nmf.py, from the repo root (needs requirements-embed.txt):
    nlp/.venv/Scripts/python nlp/scripts/02b_topics_bertopic.py
Embeddings are cached in data/nlp/emb_minilm.npy (~5 minutes on CPU to rebuild).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bertopic import BERTopic  # noqa: E402
from bertopic.vectorizers import ClassTfidfTransformer  # noqa: E402
from hdbscan import HDBSCAN  # noqa: E402
from sklearn.decomposition import NMF  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score as ami  # noqa: E402
from umap import UMAP  # noqa: E402

from energy_nlp import DATA_DIR, FIGURES_DIR, SEED, TABLES_DIR  # noqa: E402
from energy_nlp import embed, prep, topics as tp  # noqa: E402
from energy_nlp.plotting import BLUE, BLUE_LIGHT, GRID, INK, INK2, SURFACE, style, year_ticks  # noqa: E402

MIN_CLUSTER_SIZE = 15        # ~1.3% of the corpus; 10 gives 31 topics, 25 merges all power topics
STABILITY_SEEDS = (43, 44, 45, 46)
N_TOP, N_BOOT = 10, 1000
ERA_A, ERA_B = "2022Q3-2024Q4", "2025Q1-2026Q3"
ERA_EDGES = (pd.Period("2024Q4", "Q"), pd.Period("2026Q3", "Q"))
NMF_K = 17

LABELS = {
    -1: ("Outliers (no topic)", None),
    0: ("Power generation, JNHPP & renewables", "jnhpp"),
    1: ("Economic diplomacy & trade", "diplomacy"),
    2: ("Ruling party & elections", "election"),
    3: ("CNG for vehicles", "cng stations"),
    4: ("EACOP crude oil pipeline", "eacop project"),
    5: ("Macroeconomy, debt & FDI", "debt"),
    6: ("Upstream exploration & licensing", "upstream"),
    7: ("Oil & gas sector, helium & conferences", "helium"),
    8: ("Fuel prices & Middle East shocks", "hormuz"),
    9: ("Gas fields & production", "ntorya"),
    10: ("Budget, revenue & public spending", "budget"),
    11: ("Ports, SEZs & logistics", "cargo"),
    12: ("Clean cooking", "charcoal"),
    13: ("LNG project & investment deals", "lng"),
    14: ("Public enterprises & Treasury Registrar", "treasury registrar"),
    15: ("Climate & environment", "climate"),
    16: ("China relations", "china"),
    17: ("Digital economy & AI", "digital"),
    18: ("Extractives & natural resource wealth", "extractive"),
}

def make_model(stop: list[str], seed: int) -> BERTopic:
    return BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=5, prediction_data=True),
        # Topic words come from one concatenated document per topic, so document-frequency
        # cut-offs would drop topic-specific words; frequent words are damped instead.
        vectorizer_model=CountVectorizer(token_pattern=tp.TOKEN_PATTERN, stop_words=stop,
                                         ngram_range=(1, 2), min_df=1),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        top_n_words=30,
    )


def load_embeddings(d: pd.DataFrame, texts: list[str]) -> np.ndarray:
    return embed.cached_embeddings(d["post_id"].tolist(), texts, DATA_DIR / "emb_minilm.npy",
                                   DATA_DIR / "emb_minilm_ids.csv")


def term_idx(terms: list[str], vocab_pos: dict[str, int]) -> np.ndarray:
    return np.array([vocab_pos[t] for t in terms if t in vocab_pos][:N_TOP])


def plot_shares(shares: pd.DataFrame, order: list[int], path: Path) -> None:
    cols = 4
    rows = -(-len(order) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 1.9 * rows), sharex=True, sharey=True)
    ymax = shares["hi"].max() * 1.05
    for ax, t in zip(axes.flat, order):
        s = shares[shares["topic"] == t]
        x = s["period"].dt.start_time + pd.Timedelta(days=45)
        ax.fill_between(x, s["lo"], s["hi"], color=BLUE_LIGHT, linewidth=0)
        ax.plot(x, s["share"], color=BLUE, linewidth=2)
        ax.set_title(LABELS[t][0], loc="left", fontsize=8)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        style(ax)
        year_ticks(ax)
    for ax in list(axes.flat)[len(order):]:
        ax.axis("off")
    fig.suptitle("BERTopic topic shares by quarter (share of articles; band = 95% bootstrap interval)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_gas(contrib: pd.DataFrame, path: Path) -> None:
    c = contrib.iloc[::-1]
    y = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.barh(y, c["contribution"] * 100, color=BLUE, height=0.7)
    ax.hlines(y, c["lo"] * 100, c["hi"] * 100, color=INK2, linewidth=1)
    ax.set_yticks(y, c["label"], fontsize=7)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("change in share of all articles that mention gas, percentage points\n"
                  "(line = 95% bootstrap interval)", fontsize=7)
    ax.set_title(f"BERTopic: where the gas rise came from\n{ERA_A} vs {ERA_B}", loc="left", fontsize=9)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    d = prep.load_corpus(DATA_DIR / "corpus.csv")
    texts = (d["title"] + ". " + d["text"]).tolist()
    stop = sorted(set(tp.STOP_WORDS) | tp.person_names(texts))
    E = load_embeddings(d, texts)

    model = make_model(stop, SEED)
    assign = np.array(model.fit_transform(texts, E)[0])
    terms = {t: [w for w, _ in model.get_topic(t)] for t in sorted(set(assign)) if t >= 0}
    if set(terms) != set(LABELS) - {-1}:
        sys.exit(f"BERTopic found topics {sorted(terms)}; LABELS covers {sorted(set(LABELS) - {-1})}")
    for t, (label, anchor) in LABELS.items():
        if anchor and anchor not in terms[t][:15]:
            sys.exit(f"topic {t} ({label}) lost anchor {anchor!r}: {terms[t][:15]}")

    # NMF baseline from 02_topics_nmf.py (same corpus order).
    nmf_doc = pd.read_csv(DATA_DIR / "nmf_doc_topics.csv")
    assert nmf_doc["post_id"].tolist() == d["post_id"].tolist()
    nmf_dom = nmf_doc["dominant"].to_numpy()
    nmf_lab = pd.read_csv(TABLES_DIR / "nmf_topics.csv").set_index("topic")["label"]

    # One reference vocabulary/matrix for coherence of both models.
    ref = CountVectorizer(token_pattern=tp.TOKEN_PATTERN, stop_words=stop, ngram_range=(1, 2),
                          min_df=5, max_df=0.5)
    Xr = ref.fit_transform(texts)
    vocab = ref.get_feature_names_out()
    pos = {t: i for i, t in enumerate(vocab)}
    tfidf = tp.make_vectorizer(tp.person_names(texts))
    Xt = tfidf.fit_transform(texts)
    Xt.sort_indices()
    W, H = tp.fit_nmf(Xt, NMF_K, SEED)
    tv = tfidf.get_feature_names_out()
    nmf_top = np.array([term_idx(list(tv[np.argsort(-h)[:N_TOP * 2]]), pos) for h in H])
    bt_top = np.array([term_idx(terms[t], pos) for t in sorted(terms)])

    # Stability: assignment agreement across random seeds, same measure for both.
    bt_runs = [np.array(make_model(stop, s).fit_transform(texts, E)[0]) for s in STABILITY_SEEDS]
    nmf_runs = []
    for s in STABILITY_SEEDS:
        Wr = NMF(n_components=NMF_K, init="random", random_state=s, max_iter=1000).fit_transform(Xt)
        nmf_runs.append(tp.doc_proportions(Wr).argmax(axis=1))
    core = assign >= 0
    comparison = pd.DataFrame([
        ("topics", NMF_K, len(terms)),
        ("articles without a topic (outliers)", 0.0, round(float((~core).mean()), 3)),
        ("coherence: mean NPMI of top-10 terms", round(float(tp.npmi_coherence(nmf_top, Xr).mean()), 3),
         round(float(tp.npmi_coherence(bt_top, Xr).mean()), 3)),
        ("topic diversity (unique top-10 terms)", round(tp.topic_diversity(nmf_top), 3),
         round(tp.topic_diversity(bt_top), 3)),
        ("stability: mean AMI vs 4 other seeds", round(float(np.mean([ami(nmf_dom, r) for r in nmf_runs])), 3),
         round(float(np.mean([ami(assign, r) for r in bt_runs])), 3)),
        ("agreement NMF vs BERTopic: AMI (non-outliers)", round(float(ami(nmf_dom[core], assign[core])), 3),
         round(float(ami(nmf_dom[core], assign[core])), 3)),
    ], columns=["measure", "NMF (k=17)", "BERTopic"])
    comparison.to_csv(TABLES_DIR / "topic_model_comparison.csv", index=False)

    # Topic table with the closest NMF topic.
    rows = []
    for t in sorted(set(assign)):
        sel = assign == t
        best = pd.Series(nmf_dom[sel]).value_counts()
        rows.append((t, LABELS[t][0], int(sel.sum()), round(float(sel.mean()), 4),
                     nmf_lab[best.index[0]], round(float(best.iloc[0] / sel.sum()), 3),
                     ", ".join(terms[t][:12]) if t >= 0 else ""))
    bt_table = pd.DataFrame(rows, columns=["topic", "label", "n_articles", "share",
                                           "closest_nmf_topic", "overlap", "top_terms"])
    bt_table.to_csv(TABLES_DIR / "bertopic_topics.csv", index=False)
    # Where each NMF topic's articles go (row shares) - shows which NMF themes BERTopic leaves as outliers.
    xt = pd.crosstab(pd.Series(nmf_dom).map(nmf_lab).rename("nmf_topic"),
                     pd.Series(assign).map(lambda t: LABELS[t][0]).rename("bertopic_topic"),
                     normalize="index").round(3)
    xt.to_csv(TABLES_DIR / "nmf_to_bertopic_crosstab.csv")

    pd.DataFrame({"post_id": d["post_id"], "quarter": d["quarter"].astype(str), "topic": assign}).to_csv(
        DATA_DIR / "bertopic_doc_topics.csv", index=False)

    # Shares over time: one-hot assignments, bootstrapped like NMF.
    full_q = (d["quarter"] <= ERA_EDGES[1]).to_numpy()
    ids = sorted(set(assign))
    onehot = (assign[:, None] == np.array(ids)[None, :]).astype(float)
    shares = tp.period_shares(onehot[full_q], d.loc[full_q, "quarter"], N_BOOT, SEED)
    shares["topic"] = shares["topic"].map(dict(enumerate(ids)))
    shares.insert(2, "label", shares["topic"].map(lambda t: LABELS[t][0]))
    shares.assign(period=shares["period"].astype(str)).round(4).to_csv(
        TABLES_DIR / "bertopic_topic_shares_quarterly.csv", index=False)
    plot_shares(shares, bt_table.sort_values("n_articles", ascending=False)["topic"].tolist(),
                FIGURES_DIR / "bertopic_topic_shares.png")

    # Gas drill-down with BERTopic topics.
    q = d["quarter"]
    eras = pd.Series(np.select([q <= ERA_EDGES[0], q <= ERA_EDGES[1]], [ERA_A, ERA_B], "partial"))
    gas = d["topics"].str.split(";").map(lambda ts: "natural_gas" in ts)
    contrib = tp.share_change_with_intervals(gas, pd.Series(assign), eras, ERA_A, ERA_B, N_BOOT, SEED)
    contrib.insert(1, "label", contrib["group"].map(lambda t: LABELS[t][0]))
    contrib = contrib.rename(columns={"group": "topic"}).round(4)
    contrib.to_csv(TABLES_DIR / "gas_rise_by_topic_bertopic.csv", index=False)
    plot_gas(contrib, FIGURES_DIR / "gas_rise_drivers_bertopic.png")

    print(comparison.to_string(index=False))
    print("\n", bt_table[["topic", "label", "n_articles", "closest_nmf_topic", "overlap"]].to_string(index=False))
    print("\n", contrib[["label", "contribution", "lo", "hi"]].to_string(index=False))
    ind = xt.loc["Industrialisation & regional growth"].sort_values(ascending=False).head(4)
    print("\nNMF 'Industrialisation' articles in BERTopic:\n", ind.to_string())


if __name__ == "__main__":
    main()
