"""Mark which report edition supplies the default value for each observation.

The same figure is often republished, sometimes revised, in later editions.
All rows are kept; ``latest_edition == 1`` marks the row from the most recent
edition (user decision, 2026-10-15). A document's recency is the latest period
it reports on; publication date breaks ties. Listing dates alone are
unreliable on the EWURA site, so they are only used as the tie-break.
"""

from __future__ import annotations

import pandas as pd

OBS_KEYS: tuple[str, ...] = ("source", "variable", "category", "fy", "year", "month",
                             "x_seller", "x_market", "x_port", "x_price_basis")


def mark_latest_edition(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 0/1 ``latest_edition`` column (narrative rows are always 1)."""
    if df.empty:
        return df.assign(latest_edition=pd.Series(dtype="int8"))
    out = df.copy()
    coverage = out.groupby("url")["year"].max().rename("_doc_latest_year")
    out = out.join(coverage, on="url")
    out["_pub"] = out["pub_date"].fillna("")
    keys = [k for k in OBS_KEYS if k in out.columns]
    filled = out[keys].astype("string").fillna("")
    out["_obs"] = filled.agg("|".join, axis=1)
    order = out.sort_values(["_doc_latest_year", "_pub", "url"], na_position="first")
    best_url = order.groupby("_obs")["url"].last()
    out["latest_edition"] = (out["url"] == out["_obs"].map(best_url)).astype("int8")
    if "variable" in out.columns:
        out.loc[out["variable"] == "explanatory_text", "latest_edition"] = 1
    return out.drop(columns=["_doc_latest_year", "_pub", "_obs"])
