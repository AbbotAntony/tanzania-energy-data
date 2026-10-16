"""Tests for the labelling sample, blind sheet and label validation."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_nlp import labels as lb


def _corpus():
    rows = []
    pid = 1
    for year, n_gas, n_other in [(2022, 25, 12), (2023, 40, 30), (2024, 22, 9)]:
        for gas, n in [(True, n_gas), (False, n_other)]:
            for _ in range(n):
                rows.append((pid, pd.Timestamp(f"{year}-06-01"), "natural_gas;tpdc" if gas else "jnhpp",
                             f"https://x/{pid}", f"title {pid}", f"text {pid}"))
                pid += 1
    return pd.DataFrame(rows, columns=["post_id", "pub_date", "topics", "url", "title", "text"])


def test_draw_sample_strata_and_weights():
    c = _corpus()
    s = lb.draw_sample(c, seed=42)
    counts = s.groupby("stratum").size().to_dict()
    assert counts == {"2022_gas": 20, "2022_other": 10, "2023_gas": 20, "2023_other": 10,
                      "2024_gas": 20, "2024_other": 9}            # 2024_other has only 9
    # weights re-create the population size of every stratum
    assert s.groupby("stratum")["weight"].sum().round(6).to_dict() == \
        {"2022_gas": 25, "2022_other": 12, "2023_gas": 40, "2023_other": 30, "2024_gas": 22, "2024_other": 9}
    assert s["post_id"].is_unique
    assert s["sample_id"].tolist() == [f"S{i:03d}" for i in range(1, len(s) + 1)]


def test_draw_sample_reproducible_and_shuffled():
    c = _corpus()
    a, b = lb.draw_sample(c, seed=42), lb.draw_sample(c, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert not a["stratum"].is_monotonic_increasing          # order does not reveal strata
    assert set(lb.draw_sample(c, seed=7)["post_id"]) != set(a["post_id"])


def test_blind_sheet_hides_identifying_columns():
    sheet = lb.blind_sheet(lb.draw_sample(_corpus()))
    assert list(sheet.columns) == ["sample_id", "title", "text", "gas_mention", "gas_tone", "frame",
                                   "unsure", "notes"]
    assert (sheet[["gas_mention", "gas_tone", "frame"]] == "").all().all()


def test_draw_recode_subset_and_reproducible():
    design = lb.design_table(lb.draw_sample(_corpus()))
    r = lb.draw_recode(design, n=30, seed=43)
    assert len(set(r)) == 30 and set(r) <= set(design["sample_id"])
    assert r == lb.draw_recode(design, n=30, seed=43)


def test_write_sheet_refuses_to_overwrite(tmp_path):
    sheet = lb.blind_sheet(lb.draw_sample(_corpus()))
    path = tmp_path / "sheet.csv"
    lb.write_sheet(sheet, path)
    with pytest.raises(FileExistsError):
        lb.write_sheet(sheet, path)
    lb.write_sheet(sheet, path, force=True)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1252"])
def test_read_sheet_normalises_excel_input(tmp_path, encoding):
    df = pd.DataFrame({
        "sample_id": ["S001", "S002"], "title": ["Gas’s rise", "b"], "text": ["t", "u"],
        "Gas_Mention": ["Substantive ", "resource list"], "gas_tone": ["POSITIVE", "not_about_gas"],
        "frame": ["economic_opportunity", "resource_wealth"], "unsure": ["x", ""], "notes": ["", " hm "],
    })
    path = tmp_path / "done.csv"
    df.to_csv(path, index=False, encoding=encoding)
    out = lb.read_sheet(path)
    assert out["gas_mention"].tolist() == ["substantive", "resource_list"]
    assert out["gas_tone"].tolist() == ["positive", "not_about_gas"]
    assert out["unsure"].tolist() == [True, False]
    assert out["notes"].tolist() == ["", "hm"]
    assert "title" not in out.columns                            # no text in the label file


def _labels(rows):
    return pd.DataFrame(rows, columns=["sample_id", "gas_mention", "gas_tone", "frame"]).assign(
        unsure=False, notes="")


def test_validate_clean_labels_pass():
    ok = _labels([("S1", "substantive", "positive", "economic_opportunity"),
                  ("S2", "resource_list", "not_about_gas", "resource_wealth"),
                  ("S3", "energy_mix_list", "not_about_gas", "supply_reliability"),
                  ("S4", "none", "not_about_gas", "other")])
    assert lb.validate(ok).empty


def test_validate_flags_each_rule():
    bad = _labels([("S1", "", "positive", "economic_opportunity"),        # missing
                   ("S2", "substantive", "great", "economic_opportunity"),  # invalid code
                   ("S3", "energy_mix_list", "positive", "prices_costs"),   # T1b
                   ("S4", "substantive", "not_about_gas", "prices_costs"),  # T4
                   ("S5", "resource_list", "not_about_gas", "economic_opportunity")])  # T1
    p = lb.validate(bad)
    assert p.groupby("sample_id")["variable"].apply(list).to_dict() == {
        "S1": ["gas_mention"], "S2": ["gas_tone"], "S3": ["gas_tone"], "S4": ["gas_tone"], "S5": ["frame"]}
