"""Task 3: stratified hand-labelling sample, blind sheets, and label validation.

See nlp/labels/CODEBOOK.md. The sheet the coder sees holds only sample_id,
title and text; everything that could reveal the stratum (post id, date, URL,
weight) lives in sample_design.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CODES = {
    "gas_mention": ("substantive", "resource_list", "energy_mix_list", "none"),
    "gas_tone": ("positive", "neutral", "negative", "not_about_gas"),
    "frame": ("economic_opportunity", "supply_reliability", "prices_costs", "local_communities",
              "environment_climate", "governance_investment", "resource_wealth", "other"),
}
CODE_COLUMNS = list(CODES) + ["unsure", "notes"]
PER_STRATUM = {True: 20, False: 10}          # gas-tagged / other, per year
TRUTHY = {"1", "x", "y", "yes", "true", "t"}


def draw_sample(corpus: pd.DataFrame, seed: int = 42,
                per_stratum: dict[bool, int] = PER_STRATUM) -> pd.DataFrame:
    """Stratified sample (year x gas-tagged) with weights w = N_h / n_h.

    Rows come back in a random order and sample_id follows that order, so neither
    the id nor the row position reveals the stratum.
    """
    df = corpus.assign(year=corpus["pub_date"].dt.year,
                       gas_tagged=corpus["topics"].str.split(";").map(lambda t: "natural_gas" in t))
    rng = np.random.default_rng(seed)
    parts = []
    for (year, gas), g in df.sort_values("post_id").groupby(["year", "gas_tagged"]):
        n = min(per_stratum[gas], len(g))
        pick = g.iloc[np.sort(rng.choice(len(g), n, replace=False))]
        parts.append(pick.assign(stratum=f"{year}_{'gas' if gas else 'other'}", N_h=len(g), n_h=n))
    s = pd.concat(parts)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    s["weight"] = (s["N_h"] / s["n_h"]).round(4)
    s.insert(0, "sample_id", [f"S{i:03d}" for i in range(1, len(s) + 1)])
    return s


def design_table(sample: pd.DataFrame) -> pd.DataFrame:
    return sample[["sample_id", "post_id", "pub_date", "url", "year", "gas_tagged", "stratum",
                   "N_h", "n_h", "weight"]].assign(pub_date=lambda d: d["pub_date"].dt.date)


def blind_sheet(sample: pd.DataFrame) -> pd.DataFrame:
    """What the coder sees: id, title, text and empty code columns."""
    sheet = sample[["sample_id", "title", "text"]].copy()
    for c in CODE_COLUMNS:
        sheet[c] = ""
    return sheet


def draw_recode(design: pd.DataFrame, n: int = 30, seed: int = 43) -> list[str]:
    """sample_ids for the intra-coder reliability re-coding, in a fresh random order."""
    rng = np.random.default_rng(seed)
    ids = design["sample_id"].sort_values().to_numpy()
    return list(ids[rng.choice(len(ids), n, replace=False)])


def write_sheet(sheet: pd.DataFrame, path: Path, force: bool = False) -> None:
    """UTF-8 with BOM so Excel shows quotes and dashes correctly. Never overwrites
    an existing sheet (it may hold the coder's work) unless ``force``."""
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists - it may contain your labels. Use --force to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False, encoding="utf-8-sig")


def read_sheet(path: Path) -> pd.DataFrame:
    """Read a completed sheet as saved by Excel (UTF-8, UTF-8 with BOM, or Windows-1252)."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip().lower() for c in df.columns]
    for c in CODES:
        df[c] = df[c].str.strip().str.lower().str.replace(" ", "_")
    df["unsure"] = df["unsure"].str.strip().str.lower().isin(TRUTHY)
    df["notes"] = df["notes"].str.strip()
    return df[["sample_id"] + CODE_COLUMNS]


def validate(labels: pd.DataFrame) -> pd.DataFrame:
    """Problems to fix by hand: missing or invalid codes and codebook consistency rules."""
    probs = []
    for _, r in labels.iterrows():
        for var, allowed in CODES.items():
            if r[var] == "":
                probs.append((r["sample_id"], var, "missing"))
            elif r[var] not in allowed:
                probs.append((r["sample_id"], var, f"invalid code {r[var]!r}"))
        if r["gas_mention"] in ("resource_list", "energy_mix_list", "none") \
                and r["gas_tone"] not in ("", "not_about_gas"):
            probs.append((r["sample_id"], "gas_tone",
                          f"gas_mention={r['gas_mention']} requires gas_tone=not_about_gas (T1/T1b/T4)"))
        if r["gas_mention"] == "substantive" and r["gas_tone"] == "not_about_gas":
            probs.append((r["sample_id"], "gas_tone",
                          "gas_mention=substantive but gas_tone=not_about_gas (T4)"))
        if r["gas_mention"] == "resource_list" and r["frame"] not in ("", "resource_wealth"):
            probs.append((r["sample_id"], "frame", "gas_mention=resource_list requires frame=resource_wealth (T1)"))
    return pd.DataFrame(probs, columns=["sample_id", "variable", "problem"])
