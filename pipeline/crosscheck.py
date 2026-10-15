"""Compare the same quantity published by different sources (EWURA vs BoT).

Only explicitly mapped series are compared, because sources define concepts
differently (e.g. EWURA "supplied" gas is measured after processing while BoT
"production" is measured at the field). Each source's latest edition is used.
Nothing is reconciled: the user decides which source is authoritative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline import config

log = logging.getLogger(__name__)

REL_TOLERANCE = 0.005  # 0.5 %
PREFERRED: dict[str, str] = {"bot_gas": "bot_gas"}  # main series when sources disagree


@dataclass(frozen=True)
class SeriesSpec:
    dataset: str
    variable: str
    category: str
    value_col: str
    seller: str | None = None  # EWURA consumption tables are split by seller


@dataclass(frozen=True)
class Comparison:
    concept: str
    a: SeriesSpec
    b: SeriesSpec
    note: str = ""


COMPARISONS: tuple[Comparison, ...] = (
    Comparison("Gas consumed by power generation (FY, MMscf)",
               SeriesSpec("ewura_gas", "gas_consumption", "Power", "value_mmscf", "Total"),
               SeriesSpec("bot_gas", "gas_consumption", "Power generating plants", "value_mmscf")),
    Comparison("Gas consumed by industry (FY, MMscf)",
               SeriesSpec("ewura_gas", "gas_consumption", "Industries", "value_mmscf", "Total"),
               SeriesSpec("bot_gas", "gas_consumption", "Industries", "value_mmscf")),
    Comparison("Gas consumed by CNG vehicles (FY, MMscf)",
               SeriesSpec("ewura_gas", "gas_consumption", "CNG Vehicles", "value_mmscf", "Total"),
               SeriesSpec("bot_gas", "gas_consumption", "Vehicles", "value_mmscf")),
    Comparison("Total gas consumption (FY, MMscf)",
               SeriesSpec("ewura_gas", "gas_consumption", "TOTAL", "value_mmscf", "Total"),
               SeriesSpec("bot_gas", "gas_consumption", "TOTAL", "value_mmscf")),
    Comparison("Gas production at field (FY, MMscf)",
               SeriesSpec("ewura_gas", "gas_produced_by_field", "TOTAL", "value_mmscf"),
               SeriesSpec("bot_gas", "gas_production", "TOTAL", "value_mmscf")),
    Comparison("Gas supplied by processing plants (EWURA) vs field production (BoT) (FY, MMscf)",
               SeriesSpec("ewura_gas", "gas_supply_by_plant", "TOTAL", "value_mmscf"),
               SeriesSpec("bot_gas", "gas_production", "TOTAL", "value_mmscf"),
               note="different measurement points; some gap is expected"),
)


def _series(processed: Path, spec: SeriesSpec) -> pd.DataFrame:
    path = processed / f"{spec.dataset}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["fy", "value", "doc", "page"])
    df = pd.read_csv(path, low_memory=False)
    mask = (df["variable"] == spec.variable) & (df["category"] == spec.category) & df["fy"].notna()
    if "latest_edition" in df.columns:
        mask &= df["latest_edition"] == 1
    if spec.seller is not None and "x_seller" in df.columns:
        mask &= df["x_seller"] == spec.seller
    sub = df[mask & df[spec.value_col].notna()]
    sub = sub.drop_duplicates("fy", keep="last")
    return pd.DataFrame({"fy": sub["fy"], "value": sub[spec.value_col],
                         "doc": sub["doc_title"].str[:80], "page": sub["pdf_page"]})


def run(processed: Path = config.PROCESSED_DIR) -> pd.DataFrame:
    rows = []
    for comp in COMPARISONS:
        a, b = _series(processed, comp.a), _series(processed, comp.b)
        merged = a.merge(b, on="fy", suffixes=("_a", "_b"))
        for r in merged.itertuples():
            diff = r.value_a - r.value_b
            rel = abs(diff) / max(abs(r.value_a), abs(r.value_b)) if max(r.value_a, r.value_b) else 0
            rows.append({
                "concept": comp.concept, "fy": r.fy,
                "source_a": comp.a.dataset, "value_a": r.value_a, "doc_a": r.doc_a, "page_a": r.page_a,
                "source_b": comp.b.dataset, "value_b": r.value_b, "doc_b": r.doc_b, "page_b": r.page_b,
                "diff_a_minus_b": round(diff, 2), "rel_diff_pct": round(100 * rel, 2),
                "flag": "differs" if rel > REL_TOLERANCE else "agrees", "note": comp.note,
                # user decision 2026-10-15: BoT/TPDC is the main gas series, EWURA a check
                "preferred_source": PREFERRED.get(comp.b.dataset, PREFERRED.get(comp.a.dataset, "")),
            })
    out = pd.DataFrame(rows)
    out.to_csv(processed / "qa_cross_source.csv", index=False)
    log.info("cross-source: %d comparisons, %d differ by more than %.1f%%", len(out),
             int((out["flag"] == "differs").sum()) if len(out) else 0, 100 * REL_TOLERANCE)
    return out
