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
    # "oil and gas" is a sector name, not a list; exploration still says something about gas (T4)
    ("Oil and natural gas exploration continues in the Eyasi-Wembere basin.", "substantive"),
    # T4a: sector name only -> no gas mention
    ("Investors in the oil and natural gas sector met in Dar es Salaam.", None),
    ("The petroleum and natural gas industry employs thousands.", None),
    # T1c: project / sector lists are coded energy_mix_list
    ("The government is investing in roads, railways and natural gas projects.", "energy_mix_list"),
    ("Flagship projects such as the SGR, JNHPP and the LNG plant will transform the economy.",
     "energy_mix_list"),
    ("Opportunities exist in mining, agriculture and natural gas.", "energy_mix_list"),
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


def test_project_lists_kept_apart_with_detail():
    text = "Flagship projects such as the SGR, JNHPP and the LNG plant will transform the economy."
    assert mn.sentence_kind(text) == "project_list"
    assert mn.article_mention(text) == "energy_mix_list"
    assert mn.article_mention(text, detail=True) == "project_list"
    # a resource list outranks a project list in the same article
    both = text + " Tanzania has gold, coal and natural gas."
    assert mn.article_mention(both, detail=True) == "resource_list"


def test_sector_name_only_article_is_none():
    assert mn.article_mention("Investors in the oil and natural gas sector met. Roads were built.") == "none"
