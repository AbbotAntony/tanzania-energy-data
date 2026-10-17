"""Task 3: hand-labelling workflow (codebook: nlp/labels/CODEBOOK.md).

From the repo root:
  nlp/.venv/Scripts/python nlp/scripts/03a_labels.py sheet           # draw sample, write blind sheet
  nlp/.venv/Scripts/python nlp/scripts/03a_labels.py import          # check + save round-1 labels
  nlp/.venv/Scripts/python nlp/scripts/03a_labels.py recode-sheet    # >= 14 days later: 30-article sheet
  nlp/.venv/Scripts/python nlp/scripts/03a_labels.py import --round 2

Sheets are Excel workbooks (dropdowns with the valid codes, a Codebook tab) in
data/nlp/, which is never committed. The label
files in nlp/labels/ hold sample ids and codes only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

import pandas as pd  # noqa: E402

from energy_nlp import DATA_DIR, NLP_DIR, SEED  # noqa: E402
from energy_nlp import labels as lb  # noqa: E402
from energy_nlp import prep  # noqa: E402

LABEL_DIR = NLP_DIR / "labels"
DESIGN = LABEL_DIR / "sample_design.csv"
SHEETS = {1: DATA_DIR / "labelling_sheet.xlsx", 2: DATA_DIR / "recode_sheet.xlsx"}
OUTPUT = {1: LABEL_DIR / "labels_round1.csv", 2: LABEL_DIR / "labels_round2.csv"}
MIN_DAYS_BETWEEN = 14


def cmd_sheet(force: bool) -> None:
    corpus = prep.load_corpus(DATA_DIR / "corpus.csv")
    sample = lb.draw_sample(corpus, seed=SEED)
    design = lb.design_table(sample)
    if DESIGN.exists():
        # The committed design is the record of what was sampled; the sheet must match it.
        old = pd.read_csv(DESIGN)
        if old["post_id"].tolist() != design["post_id"].tolist() or                 old["sample_id"].tolist() != design["sample_id"].tolist():
            sys.exit(f"the sample drawn now differs from {DESIGN} - corpus changed? Not writing a sheet.")
    try:
        lb.write_xlsx(lb.blind_sheet(sample), SHEETS[1], force=force)
    except FileExistsError as e:
        sys.exit(str(e))
    design.to_csv(DESIGN, index=False)
    print(f"{len(sample)} articles -> {SHEETS[1]}")
    print(sample.groupby("stratum").agg(N_h=("N_h", "first"), n_h=("n_h", "first"),
                                        weight=("weight", "first")).to_string())


def cmd_import(round_: int) -> None:
    labels = lb.read_sheet(SHEETS[round_])
    problems = lb.validate(labels)
    if not problems.empty:
        print(f"{len(problems)} problem(s) - fix them in {SHEETS[round_]} and run import again:")
        print(problems.to_string(index=False))
        sys.exit(1)
    design = pd.read_csv(DESIGN)
    out = labels.merge(design[["sample_id", "post_id"]], on="sample_id", how="left")
    if out["post_id"].isna().any():
        sys.exit(f"unknown sample_ids: {out.loc[out['post_id'].isna(), 'sample_id'].tolist()}")
    out = out[["sample_id", "post_id", "gas_mention", "gas_tone", "frame", "unsure"]]
    out.assign(coded_on=date.today().isoformat()).to_csv(OUTPUT[round_], index=False)
    print(f"{len(out)} labels saved -> {OUTPUT[round_]}")
    for c in ("gas_mention", "gas_tone", "frame"):
        print("\n", out[c].value_counts().to_string())
    print(f"\nmarked unsure: {int(out['unsure'].sum())}")


def cmd_recode_sheet(force: bool) -> None:
    if not OUTPUT[1].exists():
        sys.exit("import round 1 first")
    coded = date.fromisoformat(pd.read_csv(OUTPUT[1])["coded_on"].iloc[0])
    waited = (date.today() - coded).days
    if waited < MIN_DAYS_BETWEEN and not force:
        sys.exit(f"round 1 was imported {waited} day(s) ago; wait until {MIN_DAYS_BETWEEN} days have passed")
    design = pd.read_csv(DESIGN)
    ids = lb.draw_recode(design, n=30, seed=SEED + 1)
    corpus = prep.load_corpus(DATA_DIR / "corpus.csv")
    sample = design.merge(corpus[["post_id", "title", "text"]], on="post_id")
    sheet = lb.blind_sheet(sample.set_index("sample_id").loc[ids].reset_index())
    try:
        lb.write_xlsx(sheet, SHEETS[2], force=force)
    except FileExistsError as e:
        sys.exit(str(e))
    print(f"{len(sheet)} articles -> {SHEETS[2]}  (do not look at your round-1 codes first)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--force", action="store_true", help="replace an existing sheet (loses its labels)")
    i = sub.add_parser("import")
    i.add_argument("--round", type=int, choices=(1, 2), default=1)
    r = sub.add_parser("recode-sheet")
    r.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "sheet":
        cmd_sheet(args.force)
    elif args.cmd == "import":
        cmd_import(args.round)
    else:
        cmd_recode_sheet(args.force)


if __name__ == "__main__":
    main()
