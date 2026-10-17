"""Task 3: hand-labelling workflow (codebook: nlp/labels/CODEBOOK.md).

From the repo root:
  nlp/.venv/Scripts/python nlp/scripts/03a_labels.py sheet           # draw sample, write blind sheet
  nlp/.venv/Scripts/python nlp/scripts/03a_labels.py import          # check + save round-1 labels
  nlp/.venv/Scripts/python nlp/scripts/03a_labels.py recode-sheet    # >= 14 days later: 30-article sheet
  nlp/.venv/Scripts/python nlp/scripts/03a_labels.py import --round 2
  nlp/.venv/Scripts/python nlp/scripts/03a_labels.py replace S072 --reason duplicate_of:S002

Sheets are Excel workbooks (dropdowns with the valid codes, a Codebook tab) in
data/nlp/, which is never committed. The label
files in nlp/labels/ hold sample ids and codes only.
"""

from __future__ import annotations

import argparse
import sys
import shutil
from datetime import date, datetime
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
        old = lb.ensure_status(pd.read_csv(DESIGN))
        old = old[old["replaces"] == ""]                    # the original draw, before replacements
        if old["post_id"].tolist() != design["post_id"].tolist() or \
                old["sample_id"].tolist() != design["sample_id"].tolist():
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
    design = lb.ensure_status(pd.read_csv(DESIGN))
    excluded = design.loc[design["status"] != "active", "sample_id"]
    dropped = labels["sample_id"].isin(excluded)
    if dropped.any():
        print(f"ignoring excluded rows: {', '.join(labels.loc[dropped, 'sample_id'])}")
    labels = labels[~dropped]
    if round_ == 1:
        missing = sorted(set(design.loc[design["status"] == "active", "sample_id"]) - set(labels["sample_id"]))
        if missing:
            sys.exit(f"active sample rows missing from the sheet: {missing}")
    problems = lb.validate(labels)
    if not problems.empty:
        print(f"{len(problems)} problem(s) - fix them in {SHEETS[round_]} and run import again:")
        print(problems.to_string(index=False))
        sys.exit(1)
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


def cmd_replace(sample_id: str, reason: str) -> None:
    """Exclude a sample row and add a blind replacement from the same stratum."""
    design = pd.read_csv(DESIGN)
    corpus = prep.load_corpus(DATA_DIR / "corpus.csv")
    try:
        new_design = lb.draw_replacement(design, corpus, sample_id, reason)
    except ValueError as e:
        sys.exit(str(e))
    new = new_design.iloc[-1]
    article = corpus.set_index("post_id").loc[new["post_id"]]
    backup = SHEETS[1].with_name(f"{SHEETS[1].stem}.backup-{datetime.now():%Y%m%d-%H%M%S}{SHEETS[1].suffix}")
    shutil.copy2(SHEETS[1], backup)
    lb.append_to_xlsx(SHEETS[1], pd.DataFrame([{"sample_id": new["sample_id"], "title": article["title"],
                                                "text": article["text"]}]))
    check = lb.read_sheet(SHEETS[1])
    if check["sample_id"].iloc[-1] != new["sample_id"] or \
            pd.read_excel(SHEETS[1], sheet_name="Labels")["text"].iloc[-1] != article["text"]:
        shutil.copy2(backup, SHEETS[1])
        sys.exit("workbook check failed - restored the backup, nothing changed")
    new_design.to_csv(DESIGN, index=False)
    print(f"{sample_id} excluded ({reason}); replacement {new['sample_id']} added to {SHEETS[1].name} "
          f"(backup: {backup.name})")


def cmd_reweight() -> None:
    """Recompute N_h, n_h and weights for the current (cleaned) corpus."""
    design = pd.read_csv(DESIGN)
    corpus = prep.load_corpus(DATA_DIR / "corpus.csv")
    new = lb.recompute_weights(design, corpus)
    new.to_csv(DESIGN, index=False)
    act = new[new["status"] == "active"]
    print(act.groupby("stratum").agg(N_h=("N_h", "first"), n_h=("n_h", "first"),
                                     weight=("weight", "first")).to_string())
    print(f"weights sum to {act['weight'].sum():.0f} = corpus size {len(corpus)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--force", action="store_true", help="replace an existing sheet (loses its labels)")
    i = sub.add_parser("import")
    i.add_argument("--round", type=int, choices=(1, 2), default=1)
    r = sub.add_parser("recode-sheet")
    r.add_argument("--force", action="store_true")
    x = sub.add_parser("replace", help="exclude a sample row and add a blind replacement")
    x.add_argument("sample_id")
    x.add_argument("--reason", required=True)
    sub.add_parser("reweight", help="recompute N_h and weights for the current corpus")
    args = ap.parse_args()
    if args.cmd == "sheet":
        cmd_sheet(args.force)
    elif args.cmd == "import":
        cmd_import(args.round)
    elif args.cmd == "replace":
        cmd_replace(args.sample_id, args.reason)
    elif args.cmd == "reweight":
        cmd_reweight()
    else:
        cmd_recode_sheet(args.force)


if __name__ == "__main__":
    main()
