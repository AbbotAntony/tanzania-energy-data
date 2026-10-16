"""Tests for the rule-based gas-mention classifier."""

from __future__ import annotations

import pytest

from energy_nlp import mentions as mn


@pytest.mark.parametrize("sentence, expected", [
    # resource lists (endowment)
    ("Tanzania has graphite deposits, nickel reserves, rare earth minerals and abundant natural gas.",
     "resource_list"),
    ("It is endowed with vast mineral deposits, natural gas reserves, wildlife and forests.",
     "resource_list"),
    # energy-mix lists
    ("Despite gains in hydropower and natural gas, the country still imports fuel.", "energy_mix_list"),
    ("The government is investing in hydropower, natural gas and renewable energy.", "energy_mix_list"),
    ("Households will switch to LPG, natural gas and electricity for cooking.", "energy_mix_list"),
    # substantive
    ("Natural gas production at Madimba more than doubled in 2023.", "substantive"),
    ("More gas will be available for electricity generation, industrial use and households, "
     "after drilling raised natural gas production.", "substantive"),
    ("The LNG project host government agreement was signed in Dodoma.", "substantive"),
    ("CNG is 40 per cent cheaper than petrol, motorists said.", "substantive"),
    # "oil and gas" is a sector name, not a list
    ("Oil and natural gas exploration continues in the Eyasi-Wembere basin.", "substantive"),
    # no gas
    ("Hydropower and solar will power the railway.", None),
])
def test_classify_sentence(sentence, expected):
    assert mn.classify_sentence(sentence) == expected


def test_article_precedence():
    listed = "Tanzania has gold, coal and natural gas. It also has hydropower and natural gas."
    assert mn.article_mention(listed) == "resource_list"
    assert mn.article_mention(listed + " Natural gas prices rose sharply this year.") == "substantive"
    assert mn.article_mention("We have hydropower, solar and natural gas.") == "energy_mix_list"
    assert mn.article_mention("Roads and bridges were built.") == "none"


def test_liquefied_natural_gas_is_one_mention():
    assert mn.classify_sentence("Liquefied natural gas exports could start by 2030.") == "substantive"
