"""Evaluate the completed 50-item manual validation sheet.

    python src/justification_analysis/validation/evaluate_manual_validation.py \
        [--csv <completed sheet>] [--out <report dir>]

Run this AFTER the manual columns are filled in. It reports raw counts only:
no confidence intervals, no weighting, no corpus-level precision estimate. The
sample was purposively balanced for inspection, so it does not support a
population estimate and none is produced.

The two not-accepted strata are a missed-relation / coverage diagnostic over
candidates DiMLex identified independently. They are NOT recall: the
justifications are not exhaustively gold-annotated, so relations outside the
DiMLex inventory, and relations carried by no lexical marker, are invisible
to this design. No corpus-level recall figure is derived here and none should
be quoted from these numbers.

The A/B/C section assembles evidence. It deliberately does NOT pick an option:
there is no threshold rule, because the choice depends on judgements about
which failures matter for the research question, which is a human call.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root(start=None, repo_name="masters_thesis_sdg"):
    current = (start or Path(__file__)).resolve()
    while current.name != repo_name:
        if current.parent == current:
            raise FileNotFoundError(f"repo root {repo_name!r} not found")
        current = current.parent
    return current


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# BASE-ONLY utility. This documents a CLOSED base-development strand, so it is
# not rerun for other stages and must never quietly read base paths from a
# non-base configuration.
# ---------------------------------------------------------------------------
from src.justification_analysis.pipeline.config import (  # noqa: E402
    default_config, require_base_stage)

_CONFIG = default_config(repo_root=REPO_ROOT)
require_base_stage(_CONFIG, "evaluate_manual_validation", "It evaluates the completed 50-case base validation sheet, which is a one-off record and not regenerated per stage.")

ARTIFACTS = _CONFIG.discourse_dir
AMBIGUOUS_FORMS = {"as", "for", "and", "or", "then", "since", "while"}

TRUE_VALUES = {"y", "yes", "true", "1", "t"}
FALSE_VALUES = {"n", "no", "false", "0", "f"}


def parse_bool(value):
    """Accept y/n, yes/no, true/false, 1/0. Blank stays missing."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text == "":
        return np.nan
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return np.nan


def load_completed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "validation_id", "failure_type", "marker", "discopy_top_level",
        "discopy_confidence", "dimlex_category", "sentence_text",
        "manual_is_connective", "manual_top_level_category",
        "manual_valid_relation_missed_by_discopy", "notes",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"validation sheet is missing columns: {sorted(missing)}")

    frame["manual_is_connective_bool"] = frame["manual_is_connective"].map(parse_bool)
    frame["manual_missed_bool"] = (
        frame["manual_valid_relation_missed_by_discopy"].map(parse_bool)
    )
    frame["manual_top_level_category"] = (
        frame["manual_top_level_category"].astype(str).str.strip().replace({"nan": ""})
    )
    frame["form"] = frame["marker"].astype(str).str.lower().str.strip()
    frame["is_ambiguous_form"] = frame["form"].isin(AMBIGUOUS_FORMS)
    frame["confidence_band"] = pd.cut(
        frame["discopy_confidence"], [-0.01, 0.5, 0.9, 1.01],
        labels=["low(<0.5)", "mid(0.5-0.9)", "high(>=0.9)"],
    )
    return frame


def report_accepted(frame: pd.DataFrame, lines: list):
    accepted = frame[frame["failure_type"] == "accepted"].copy()
    n = len(accepted)
    lines.append("=" * 84)
    lines.append("ACCEPTED discopy RELATIONS")
    lines.append("=" * 84)

    unlabelled = int(accepted["manual_is_connective_bool"].isna().sum())
    if unlabelled:
        lines.append(f"  !! {unlabelled}/{n} rows have no manual_is_connective label")

    labelled = accepted[accepted["manual_is_connective_bool"].notna()]
    n_conn = int(labelled["manual_is_connective_bool"].sum())
    lines.append(
        f"\nConnective identification: {n_conn}/{len(labelled)} judged actual "
        "discourse connectives"
    )

    valid = labelled[labelled["manual_is_connective_bool"] == True]  # noqa: E712
    scored = valid[valid["manual_top_level_category"] != ""]
    correct = scored["manual_top_level_category"].str.lower().eq(
        scored["discopy_top_level"].astype(str).str.lower()
    )
    lines.append(
        f"Top-level sense: {int(correct.sum())}/{len(scored)} correct "
        "(among those judged connectives, with a manual category given)"
    )

    lines.append("\n-- by predicted top-level category --")
    for category, group in valid.groupby("discopy_top_level"):
        sub = group[group["manual_top_level_category"] != ""]
        ok = sub["manual_top_level_category"].str.lower().eq(
            sub["discopy_top_level"].astype(str).str.lower()).sum()
        lines.append(f"   {category:<12} sense correct {int(ok)}/{len(sub)}")

    lines.append("\n-- by form ambiguity --")
    for flag, group in labelled.groupby("is_ambiguous_form"):
        label = "ambiguous" if flag else "non-ambiguous"
        lines.append(
            f"   {label:<14} connective {int(group['manual_is_connective_bool'].sum())}"
            f"/{len(group)}"
        )

    lines.append("\n-- by confidence band --")
    for band, group in labelled.groupby("confidence_band", observed=True):
        lines.append(
            f"   {str(band):<14} connective {int(group['manual_is_connective_bool'].sum())}"
            f"/{len(group)}"
        )

    lines.append("\n-- individual errors --")
    errors = []
    for row in labelled.itertuples(index=False):
        problems = []
        if row.manual_is_connective_bool is False:
            problems.append("NOT a connective")
        elif (row.manual_top_level_category
              and row.manual_top_level_category.lower()
              != str(row.discopy_top_level).lower()):
            problems.append(
                f"sense {row.discopy_top_level} -> should be "
                f"{row.manual_top_level_category}"
            )
        if problems:
            errors.append((row.validation_id, row.marker, "; ".join(problems),
                           row.discopy_confidence, row.sentence_text))
    if not errors:
        lines.append("   (none)")
    for vid, marker, problem, conf, sentence in errors:
        lines.append(f"   [#{vid}] {marker!r} p={conf:.2f} - {problem}")
        lines.append(f"        {str(sentence)[:150]}")
    return errors


def report_missed(frame: pd.DataFrame, failure_type: str, title: str, lines: list):
    subset = frame[frame["failure_type"] == failure_type].copy()
    lines.append("\n" + "=" * 84)
    lines.append(title)
    lines.append("=" * 84)
    lines.append(
        "Missed-relation / coverage diagnostic over DiMLex-identified "
        "candidates. NOT recall; no corpus-level figure may be derived."
    )
    labelled = subset[subset["manual_missed_bool"].notna()]
    unlabelled = len(subset) - len(labelled)
    if unlabelled:
        lines.append(f"  !! {unlabelled}/{len(subset)} rows unlabelled")
    n_valid = int(labelled["manual_missed_bool"].sum())
    lines.append(
        f"\n{n_valid}/{len(labelled)} manually judged valid discourse relations"
    )
    lines.append("\n-- individual cases judged valid --")
    valid = labelled[labelled["manual_missed_bool"] == True]  # noqa: E712
    if not len(valid):
        lines.append("   (none)")
    for row in valid.itertuples(index=False):
        lines.append(
            f"   [#{row.validation_id}] {row.marker!r} "
            f"DiMLex={row.dimlex_category} "
            f"manual={row.manual_top_level_category or '-'}"
        )
        lines.append(f"        {str(row.sentence_text)[:150]}")
    if len(labelled):
        lines.append("\n-- forms responsible --")
        counts = valid.groupby("form").size().sort_values(ascending=False)
        lines.append("   " + (counts.to_string().replace("\n", "\n   ")
                              if len(counts) else "(none)"))
    return valid


def report_evidence(frame, accepted_errors, rejected_valid, notenum_valid, lines):
    lines.append("\n" + "=" * 84)
    lines.append("EVIDENCE FOR OPTIONS A / B / C  (no automatic choice)")
    lines.append("=" * 84)
    lines.append(
        "This section assembles evidence only. No threshold decides the "
        "option: which failures matter depends on the research question, "
        "which is a human judgement made after reading the cases below."
    )

    accepted = frame[frame["failure_type"] == "accepted"]
    labelled = accepted[accepted["manual_is_connective_bool"].notna()]
    n_not_conn = int((labelled["manual_is_connective_bool"] == False).sum())  # noqa: E712

    lines.append("\nA. discopy as the main contextual analysis")
    lines.append(f"   accepted cases judged NOT connectives : {n_not_conn}/{len(labelled)}")
    lines.append(f"   accepted cases with a sense error     : "
                 f"{sum(1 for e in accepted_errors if 'sense' in e[2])}")
    lines.append("   -> weaker if either count is high relative to the sample")

    lines.append("\nB. DiMLex candidate inventory + contextual disambiguation")
    lines.append(f"   never-enumerated judged valid relations: {len(notenum_valid)}/10")
    lines.append(f"   NoSense rejections judged valid        : {len(rejected_valid)}/10")
    lines.append(
        "   -> B addresses ONLY the never-enumerated column. NoSense errors "
        "are contextual-classification failures that supplying DiMLex "
        "candidates would not fix, because the same classifier fires on the "
        "same span."
    )

    lines.append("\nC. retain the lexical DiMLex analysis")
    lines.append(
        "   -> supported only if discopy is unreliable overall, i.e. if the "
        "accepted-case error counts above are high."
    )

    lines.append("\n-- conditional (if / then / if ... then) cases in the sample --")
    conditional = frame[frame["form"].isin({"if", "then", "if then"})]
    if not len(conditional):
        lines.append("   (none sampled)")
    for row in conditional.itertuples(index=False):
        lines.append(
            f"   [#{row.validation_id}] {row.marker!r} type={row.failure_type} "
            f"pred={row.discopy_top_level} manual={row.manual_top_level_category or '-'}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ARTIFACTS / "manual_validation_sample_50.csv"))
    parser.add_argument("--out", default=str(ARTIFACTS))
    args = parser.parse_args()

    frame = load_completed(Path(args.csv))
    lines = []
    lines.append(f"Manual validation report - source: {Path(args.csv).name}")
    lines.append(f"rows: {len(frame)}  |  "
                 f"{frame['failure_type'].value_counts().to_dict()}")
    lines.append("Raw counts only. No confidence intervals, no corpus-level "
                 "precision estimate, no recall.\n")

    accepted_errors = report_accepted(frame, lines)
    rejected_valid = report_missed(
        frame, "rejected_nosense",
        "NoSense REJECTIONS  (contextual-classification evidence)", lines)
    notenum_valid = report_missed(
        frame, "not_enumerated",
        "NEVER-ENUMERATED CANDIDATES  (candidate-coverage evidence)", lines)
    report_evidence(frame, accepted_errors, rejected_valid, notenum_valid, lines)

    text = "\n".join(lines)
    print(text)
    out_path = Path(args.out) / "manual_validation_report.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
