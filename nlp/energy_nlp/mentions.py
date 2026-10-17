"""How an article mentions natural gas: discussed, or only listed.

Rule-based and INDICATIVE until validated against the hand-coded ``gas_mention``
labels. Version 2 follows codebook v4 (rules T1, T1b, T1c, T4, T4a). It was
built from the codebook and from articles outside the labelled sample, never
from the labels.

A gas mention is a *list item* when a list joiner (comma, "and", "or") is next
to it and another item of a known kind is nearby. The item kind decides the
list type:
  resource list    "minerals, natural gas and wildlife"                  (T1)
  energy-mix list  "hydropower, natural gas and solar"                   (T1b)
  project list     "projects in energy, natural gas and roads"           (T1c)
Project lists are coded ``energy_mix_list`` as in the codebook, but kept apart
as ``project_list`` by ``article_mention(detail=True)``, like the coder's
``proj_list`` notes. "Oil and natural gas" followed by a sector word ("sector",
"industry", "companies", ...) is only the sector's name and does not count as a
mention (T4a); "oil and natural gas exploration" still does.
"""

from __future__ import annotations

import re

RULE_VERSION = 2
GAS = re.compile(r"\bnatural gas\b|\bliquefied natural gas\b|\bcompressed natural gas\b|\blng\b|\bcng\b")
RESOURCE_ITEMS = {
    "mineral", "minerals", "gold", "graphite", "nickel", "rare", "coal", "uranium", "helium",
    "iron", "gemstones", "tanzanite", "diamonds", "wildlife", "forests", "land", "lakes",
    "coastline", "parks", "copper", "lithium", "cobalt", "fisheries", "beaches",
}
ENERGY_ITEMS = {
    "hydropower", "hydro", "hydroelectric", "solar", "wind", "geothermal", "renewable",
    "renewables", "nuclear", "biomass", "electricity", "lpg", "charcoal", "firewood", "biogas",
}
# Sectors, projects, facilities and investment areas listed alongside gas (T1c).
PROJECT_ITEMS = {
    "mining", "agriculture", "tourism", "manufacturing", "fishing", "infrastructure", "roads", "road",
    "railway", "railways", "sgr", "ports", "port", "airports", "airport", "bridges", "jnhpp", "eacop",
    "refinery", "projects", "transport", "telecommunications", "ict", "health", "education",
    "construction", "logistics", "livestock", "industry", "industries", "energy", "petroleum",
}
# A word right after "oil and natural gas" that makes it a sector name (T4a).
SECTOR_WORDS = {
    "sector", "sectors", "sub", "subsector", "industry", "industries", "companies", "company", "firms",
    "investors", "players", "stakeholders", "value", "conference", "summit", "exhibition", "expo", "week",
}
JOINERS = {",", "and", "or", "&", ";"}
WINDOW = 6          # tokens either side searched for other list items
GAP = 2             # words allowed between a joiner and the gas term ("and abundant natural gas")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
TOKEN = re.compile(r"[a-z]+|[,;&]")
CODEBOOK_KIND = {"substantive": "substantive", "resource_list": "resource_list",
                 "energy_mix_list": "energy_mix_list", "project_list": "energy_mix_list"}


def sentence_kind(sentence: str) -> str | None:
    """'substantive', 'resource_list', 'energy_mix_list', 'project_list', or None
    if the sentence has no gas mention (sector names do not count)."""
    s = sentence.lower()
    if not GAS.search(s):
        return None
    tokens = TOKEN.findall(s)
    kinds = []
    i = 0
    while i < len(tokens):
        j = _gas_end(tokens, i)
        if j is None:
            i += 1
            continue
        if tokens[max(0, i - 2):i] in (["oil", "and"], ["petroleum", "and"]) and \
                j < len(tokens) and tokens[j] in SECTOR_WORDS:
            i = j                                           # sector name only (T4a)
            continue
        left = tokens[max(0, i - GAP - 1):i]
        right = tokens[j:j + 1]
        joined = bool(JOINERS & set(left)) or bool(JOINERS & set(right))
        near = set(tokens[max(0, i - WINDOW):i]) | set(tokens[j:j + WINDOW])
        if joined and near & RESOURCE_ITEMS:
            kinds.append("resource_list")
        elif joined and near & ENERGY_ITEMS:
            kinds.append("energy_mix_list")
        elif joined and near & PROJECT_ITEMS:
            kinds.append("project_list")
        else:
            kinds.append("substantive")
        i = j
    for k in ("substantive", "resource_list", "energy_mix_list", "project_list"):
        if k in kinds:
            return k
    return None


def classify_sentence(sentence: str) -> str | None:
    """Codebook ``gas_mention`` code for one sentence (project lists -> energy_mix_list)."""
    k = sentence_kind(sentence)
    return CODEBOOK_KIND[k] if k else None


def _gas_end(tokens: list[str], i: int) -> int | None:
    """End index (exclusive) if a gas term starts at ``i``."""
    t = tokens[i]
    if t in ("lng", "cng"):
        return i + 1
    if t == "natural" and i + 1 < len(tokens) and tokens[i + 1] == "gas":
        return i + 2
    if t in ("liquefied", "compressed") and tokens[i + 1:i + 3] == ["natural", "gas"]:
        return i + 3
    return None


def article_mention(text: str, detail: bool = False) -> str:
    """Article-level code with the codebook's precedence: substantive if any sentence
    discusses gas, else resource_list, else energy_mix_list; 'none' if gas is not
    mentioned. With ``detail``, project lists are returned as 'project_list'."""
    kinds = {k for k in map(sentence_kind, SENT_SPLIT.split(text)) if k}
    for k in ("substantive", "resource_list", "energy_mix_list", "project_list"):
        if k in kinds:
            return k if detail else CODEBOOK_KIND[k]
    return "none"
