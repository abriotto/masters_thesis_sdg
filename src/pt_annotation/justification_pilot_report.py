"""
Turn the pilot annotations into the tables that decide whether the prompt is
ready to run on all 2,287 justifications.

The scheme is defined by src/prompts/justification_annotation.txt, which is
the authority. JUSTIFICATION_CODEBOOK.md is an earlier draft and out of date;
the categories, the use vocabulary and the rule_mentioned flag here all follow
the prompt.

Three questions, in order of what would sink the study:

  1. Did the model follow the output contract? Altered sentence text,
     paraphrased evidence_span, invented sentence ids, or vocabulary drift
     all break downstream alignment. Reported as compliance rates, per flag
     type, so a systematic failure is visible as a rate rather than as one
     scary example.

  2. Is `Other` behaving as the prompt intends? The prompt states that Other
     is "a coverage category, NOT a default category". A high Other rate
     means either the category set has a genuine gap worth naming, or the
     model is using Other as a dumping ground. Every Other instance is
     dumped with its other_description so the two can be told apart.

  3. Is any category too rare to judge? A category with almost no instances
     in the pilot tells you nothing about whether the prompt handles it --
     which matters most for `Social`, the construct the research question
     turns on and the riskiest boundary in the prompt (Social vs Testimony,
     fact vs judgement). If it is near-empty here, a random full-corpus run
     will not tell you whether that boundary works either.

Also writes the review sheet: the model's own annotations laid out one row
per decision, with blank verdict columns. This is a check of the model's
output, not independent coding, so what it yields is an error rate against
your adjudication -- not a chance-corrected agreement coefficient.

Everything for this pilot lives under results/justification_annotation/pilot_v1/:
the sample, the raw annotations, the review sheet, and the summary tables in
tables/. One run, one folder.

Usage:
    python src/pt_annotation/justification_pilot_report.py
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PILOT_DIR = REPO_ROOT / "results" / "justification_annotation" / "pilot_v1"
DEFAULT_ANNOTATIONS_PATH = DEFAULT_PILOT_DIR / "pilot_annotations.jsonl"
DEFAULT_SAMPLE_PATH = DEFAULT_PILOT_DIR / "pilot_sample.jsonl"
DEFAULT_REVIEW_SHEET_PATH = DEFAULT_PILOT_DIR / "pilot_review_sheet.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_PILOT_DIR / "tables"

CATEGORIES = [
    "Deduction", "Consistency", "Payoff",
    "Testimony", "Social", "Behavioral", "Other",
]
USES = ["used", "discounted", "mentioned"]

# Not a rule from the prompt -- the prompt gives no numeric threshold. This is
# a review trigger: above it, read every Other instance before the full run
# and decide whether the category set has a gap.
OTHER_THRESHOLD = 0.15

# Below this many instances, the pilot has not exercised the category enough
# to say whether the prompt handles it.
RARE_CATEGORY_THRESHOLD = 5


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def flag_type(flag):
    """Collapse a validation flag to its kind, dropping the specifics, so
    rates can be counted."""
    flag = re.sub(r"^sentence \d+: ", "", flag)
    for pattern, name in [
        (r"^evidence_span is not a substring", "evidence_span not verbatim"),
        (r"^evidence_span matches only after whitespace", "evidence_span whitespace-only match"),
        (r"^missing evidence_span", "evidence_span missing"),
        (r"^text was altered", "sentence text altered"),
        (r"^invalid category", "invalid category"),
        (r"^invalid use", "invalid use"),
        (r"^rule_mentioned is", "rule_mentioned not boolean"),
        (r"^Other without other_description", "Other without description"),
        (r"^other_description set on non-Other", "description on non-Other"),
        (r"^duplicate annotation", "duplicate annotation"),
        (r"^missing sentences", "missing sentences"),
        (r"^sentences not in the input", "invented sentence id"),
        (r"^sentence_id not in the input", "invented sentence id"),
        (r"^duplicate sentence_id", "duplicate sentence id"),
        (r"^vote not preserved", "vote not preserved"),
    ]:
        if re.match(pattern, flag):
            return name
    return flag[:60]


def flatten(annotation_records, sample_by_id):
    """One row per (sentence, annotation); sentences with no annotation get a
    single row with category None, so the unlabelled rate is measurable."""
    sentence_rows = []
    annotation_rows = []

    for record in annotation_records:
        metadata = record["metadata"]
        annotation = record.get("annotation")
        if annotation is None:
            continue

        justification_id = metadata["justification_id"]
        item = sample_by_id.get(justification_id, {})

        for sentence in annotation.get("sentences", []):
            categories = [
                a.get("category")
                for a in sentence.get("annotations", [])
                if isinstance(a, dict)
            ]

            sentence_rows.append({
                "justification_id": justification_id,
                "model_under_annotation": metadata.get("model_under_annotation"),
                "is_correct": metadata.get("is_correct"),
                "sentence_id": sentence.get("sentence_id"),
                "text": sentence.get("text"),
                "rule_mentioned": sentence.get("rule_mentioned"),
                "n_annotations": len(categories),
                "is_labelled": len(categories) > 0,
                "categories": "|".join(sorted(set(c for c in categories if c))),
            })

            for annotation_item in sentence.get("annotations", []):
                if not isinstance(annotation_item, dict):
                    continue
                annotation_rows.append({
                    "justification_id": justification_id,
                    "model_under_annotation": metadata.get("model_under_annotation"),
                    "is_correct": metadata.get("is_correct"),
                    "vote": item.get("vote"),
                    "sentence_id": sentence.get("sentence_id"),
                    "text": sentence.get("text"),
                    "rule_mentioned": sentence.get("rule_mentioned"),
                    "category": annotation_item.get("category"),
                    "use": annotation_item.get("use"),
                    "evidence_span": annotation_item.get("evidence_span"),
                    "other_description": annotation_item.get("other_description"),
                })

    return pd.DataFrame(sentence_rows), pd.DataFrame(annotation_rows)


def write_review_sheet(sentence_frame, annotation_frame, path):
    """The model's annotations laid out for checking, one row per decision.

    Sentences the model left unlabelled get a row too, with category
    "(none)". They are the easiest failure to miss by eye and the one the
    prompt most invites -- it instructs the model not to force an annotation
    onto every sentence, so silent under-labelling looks exactly like correct
    restraint until you look.

    Rows are ordered by justification and sentence_id, not shuffled: you are
    checking output, and a sentence is hard to judge without the ones around
    it. verdict is left blank for you to fill with ok / wrong / missed.
    """
    unlabelled = sentence_frame[~sentence_frame["is_labelled"]].copy()
    unlabelled["category"] = "(none)"
    for column in ["use", "evidence_span", "other_description", "vote"]:
        unlabelled[column] = None

    columns = [
        "justification_id", "sentence_id", "text", "rule_mentioned",
        "category", "use", "evidence_span", "other_description",
    ]

    sheet = pd.concat(
        [annotation_frame[columns], unlabelled[columns]],
        ignore_index=True,
    )

    sheet = sheet.sort_values(
        ["justification_id", "sentence_id", "category"]
    ).reset_index(drop=True)

    sheet["verdict"] = ""
    sheet["should_be"] = ""
    sheet["note"] = ""

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False, encoding="utf-8")
    return len(sheet)


def main():
    parser = argparse.ArgumentParser(description="Summarise the justification-annotation pilot.")
    parser.add_argument("--annotations-path", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-sheet-path", type=Path, default=DEFAULT_REVIEW_SHEET_PATH)
    args = parser.parse_args()

    records = read_jsonl(args.annotations_path)

    # Dry-run stubs never contacted the API. Counting them would inflate the
    # compliance rate with calls that were never made.
    n_dry_run = sum(r["metadata"].get("dry_run", False) for r in records)
    records = [r for r in records if not r["metadata"].get("dry_run")]
    if n_dry_run:
        print(f"Ignoring {n_dry_run} dry-run record(s).\n")

    if not records:
        print("No live annotations in the file -- run the annotator first.")
        return

    sample_by_id = {item["justification_id"]: item for item in read_jsonl(args.sample_path)}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. contract compliance ----------
    n_total = len(records)
    n_errored = sum("error" in r["metadata"] for r in records)
    n_returned = n_total - n_errored
    n_clean = sum(
        not r["metadata"].get("validation_flags") and "error" not in r["metadata"]
        for r in records
    )

    flag_counter = Counter()
    justifications_with_flag = Counter()
    for record in records:
        flags = record["metadata"].get("validation_flags", [])
        kinds = [flag_type(f) for f in flags]
        flag_counter.update(kinds)
        justifications_with_flag.update(set(kinds))

    compliance = pd.DataFrame(
        [
            {
                "flag": name,
                "n_occurrences": count,
                "n_justifications": justifications_with_flag[name],
                "pct_justifications": 100 * justifications_with_flag[name] / max(n_returned, 1),
            }
            for name, count in flag_counter.most_common()
        ]
    )
    compliance.to_csv(args.output_dir / "01_contract_compliance.csv", index=False)

    print("=" * 72)
    print("1. CONTRACT COMPLIANCE")
    print("=" * 72)
    print(f"justifications sent      : {n_total}")
    print(f"returned an annotation   : {n_returned}")
    print(f"API/parse errors         : {n_errored}")
    print(f"fully clean (no flags)   : {n_clean} "
          f"({100 * n_clean / max(n_total, 1):.1f}%)")
    if len(compliance):
        print()
        print(compliance.to_string(index=False))
    print()

    sentence_frame, annotation_frame = flatten(records, sample_by_id)

    if sentence_frame.empty:
        print("No annotations returned -- nothing further to report.")
        return

    sentence_frame.to_csv(args.output_dir / "02_sentences.csv", index=False)
    annotation_frame.to_csv(args.output_dir / "03_annotations.csv", index=False)

    # ---------- 2. label distribution ----------
    n_sentences = len(sentence_frame)
    n_labelled = int(sentence_frame["is_labelled"].sum())

    print("=" * 72)
    print("2. LABEL DISTRIBUTION")
    print("=" * 72)
    print(f"sentences                : {n_sentences}")
    print(f"labelled                 : {n_labelled} ({100 * n_labelled / n_sentences:.1f}%)")
    print(f"unlabelled               : {n_sentences - n_labelled} "
          f"({100 * (n_sentences - n_labelled) / n_sentences:.1f}%)")
    print(f"rule_mentioned = true    : {int(sentence_frame['rule_mentioned'].sum())} "
          f"({100 * sentence_frame['rule_mentioned'].mean():.1f}%)")
    print(f"multi-label sentences    : {int((sentence_frame['n_annotations'] > 1).sum())} "
          f"({100 * (sentence_frame['n_annotations'] > 1).mean():.1f}%)")
    print()

    category_counts = (
        annotation_frame["category"]
        .value_counts()
        .reindex(CATEGORIES)
        .fillna(0)
        .astype(int)
    )
    category_table = pd.DataFrame({
        "category": CATEGORIES,
        "n_annotations": category_counts.values,
        "pct_of_annotations": 100 * category_counts.values / max(len(annotation_frame), 1),
        "n_sentences": [
            int(sentence_frame["categories"].str.split("|").apply(lambda cs: c in cs).sum())
            for c in CATEGORIES
        ],
    })
    category_table["pct_of_labelled_sentences"] = (
        100 * category_table["n_sentences"] / max(n_labelled, 1)
    )
    category_table["too_rare_for_agreement"] = (
        category_table["n_annotations"] < RARE_CATEGORY_THRESHOLD
    )
    category_table.to_csv(args.output_dir / "04_category_distribution.csv", index=False)
    print(category_table.to_string(index=False))
    print()

    use_table = (
        annotation_frame
        .groupby(["category", "use"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=CATEGORIES, columns=USES, fill_value=0)
    )
    use_table.to_csv(args.output_dir / "05_use_by_category.csv")
    print("use x category:")
    print(use_table.to_string())
    print()

    by_model = (
        sentence_frame
        .groupby("model_under_annotation")
        .agg(
            n_sentences=("sentence_id", "size"),
            pct_labelled=("is_labelled", lambda s: 100 * s.mean()),
            pct_rule_mentioned=("rule_mentioned", lambda s: 100 * s.mean()),
        )
    )
    by_model.to_csv(args.output_dir / "06_by_model.csv")
    print("by model under annotation:")
    print(by_model.to_string())
    print()

    # ---------- 3. prompt-level decisions ----------
    other_count = int(category_counts.get("Other", 0))
    other_share = other_count / max(len(annotation_frame), 1)

    print("=" * 72)
    print("3. PROMPT DECISIONS")
    print("=" * 72)
    print(f"Other: {other_count} annotations = {100 * other_share:.1f}% of all annotations "
          f"(review trigger {100 * OTHER_THRESHOLD:.0f}%)")
    if other_share > OTHER_THRESHOLD:
        print("  -> ABOVE the review trigger. The prompt calls Other a coverage category,")
        print("     not a default. Read every instance below: either the category set has")
        print("     a real gap worth naming, or Other is being used as a dumping ground.")
    else:
        print("  -> below the review trigger; Other is behaving as a coverage category.")

    other_rows = annotation_frame[annotation_frame["category"].eq("Other")]
    if len(other_rows):
        other_rows.to_csv(args.output_dir / "07_other_instances.csv", index=False)
        print()
        print("  Other descriptions:")
        for description, count in Counter(
            other_rows["other_description"].fillna("(none)")
        ).most_common():
            print(f"    {count:3d}  {description}")

    rare = category_table[category_table["too_rare_for_agreement"]]
    print()
    if len(rare):
        print(f"Categories with < {RARE_CATEGORY_THRESHOLD} instances "
              "(the pilot did not exercise these):")
        for row in rare.itertuples(index=False):
            print(f"    {row.category}: {row.n_annotations}")
        print("  -> the pilot says nothing about whether the prompt handles them. If Social")
        print("     is on this list, target more candidates before trusting the full run.")
    else:
        print(f"Every category has >= {RARE_CATEGORY_THRESHOLD} instances.")

    # ---------- 4. review sheet ----------
    n_review_rows = write_review_sheet(
        sentence_frame, annotation_frame, args.review_sheet_path
    )

    print()
    print(f"Tables written to {args.output_dir}")
    print(f"Review sheet    : {args.review_sheet_path} ({n_review_rows} rows)")


if __name__ == "__main__":
    main()
