"""How an article mentions natural gas: discussed, or only listed.

Rule-based and INDICATIVE until validated against the hand-coded ``gas_mention``
labels (codebook rules T1/T1b/T4). A gas mention is a *list item* when a list
joiner (comma, "and", "or") is next to it and another item of a known kind is
nearby; the item kind decides between a resource list ("minerals, natural gas
and wildlife") and an energy-mix list ("hydropower, natural gas and solar").
"Oil and gas" is a sector name, not a list, so oil/petroleum are not list items.
"""

from __future__ import annotations

import re

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
JOINERS = {",", "and", "or", "&", ";"}
WINDOW = 6          # tokens either side searched for other list items
GAP = 2             # words allowed between a joiner and the gas term ("and abundant natural gas")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
TOKEN = re.compile(r"[a-z]+|[,;&]")


def classify_sentence(sentence: str) -> str | None:
    """'resource_list', 'energy_mix_list', 'substantive', or None if gas is not mentioned."""
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
        left = tokens[max(0, i - GAP - 1):i]
        right = tokens[j:j + 1 + 1]
        joined = bool(JOINERS & set(left)) or bool(JOINERS & set(right[:1]))
        near = set(tokens[max(0, i - WINDOW):i]) | set(tokens[j:j + WINDOW])
        if joined and near & RESOURCE_ITEMS:
            kinds.append("resource_list")
        elif joined and near & ENERGY_ITEMS:
            kinds.append("energy_mix_list")
        else:
            kinds.append("substantive")
        i = j
    if "substantive" in kinds:
        return "substantive"
    return "resource_list" if "resource_list" in kinds else "energy_mix_list"


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


def article_mention(text: str) -> str:
    """Article-level code, mirroring the codebook's precedence:
    substantive if any sentence discusses gas, else resource_list if any resource
    list, else energy_mix_list; 'none' if gas is not mentioned."""
    kinds = {k for k in map(classify_sentence, SENT_SPLIT.split(text)) if k}
    for k in ("substantive", "resource_list", "energy_mix_list"):
        if k in kinds:
            return k
    return "none"
