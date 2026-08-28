"""Characterise the DiMLex occurrences discopy never enumerated as candidates.

Answers one question: of the ~1,660 out-of-inventory occurrences, how many are
plausibly genuine discourse connectives? That is what decides whether a
DiMLex-candidates -> discopy-sense hybrid would buy anything.

No LLM is used. Classification is deterministic and syntactic, based on the
position of the form in its sentence and the material immediately around it.
Every rule is a heuristic and is reported as such - the output is a triage that
tells you where the mass sits, not a gold annotation.

The three buckets:

  CLAUSE_INITIAL_PLAUSIBLE  the form opens the sentence or a clause and is
                            followed later by a comma, the shape a subordinator
                            or discourse adverbial takes ("Given X, Y").
  NON_CONNECTIVE_LIKELY     the form is used in a way that cannot be a
                            clause-level connective: a verb ("was given"), a
                            plain preposition inside an NP, and so on.
  UNCLEAR                   neither test fires; needs eyes.
"""
from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

# Forms worth separating carefully: each is in DiMLex but not in discopy's
# candidate inventory, and each has a non-connective reading that dominates in
# ordinary prose.
FOCUS_FORMS = ("with", "given", "despite", "particularly", "eventually",
               "given that", "without", "upon")

# `given` after any of these is a passive/verbal use, never a connective.
VERBAL_GIVEN = re.compile(
    r"\b(?:was|were|is|are|been|being|be|has|have|had|having)\s+given\b", re.I)

# A determiner/possessive right after the form means it heads a noun phrase.
NP_HEAD = re.compile(
    r"^(?:the|a|an|his|her|its|their|this|that|these|those|my|our|your|"
    r"[A-Z][a-z]+(?:'s)?)\b")


def classify_occurrence(marker: str, sentence: str, start: int, end: int) -> str:
    """Deterministic triage of one out-of-inventory occurrence."""
    if not isinstance(sentence, str) or not sentence:
        return "UNCLEAR"

    before = sentence[:start]
    after = sentence[end:].lstrip()
    stripped_before = before.strip()
    form = marker.lower()

    # Verbal `given` is decisive.
    if form == "given" and VERBAL_GIVEN.search(sentence[max(0, start - 30):end]):
        return "NON_CONNECTIVE_LIKELY"

    # Clause-initial: nothing before it, or only a clause boundary.
    clause_initial = (
        stripped_before == ""
        or bool(re.search(r"[,;:]\s*$", before))
        or bool(re.search(r"\b(?:and|but|or|which|that|because|while)\s*$",
                          before, re.I))
    )

    # A subordinator introduces material that is closed by a comma before the
    # main clause. Without a following comma the form is almost always a
    # preposition heading an NP.
    followed_by_comma = "," in sentence[end:]

    if form in ("given that",):
        # Explicit complementiser - a clause follows by construction.
        return ("CLAUSE_INITIAL_PLAUSIBLE" if clause_initial or followed_by_comma
                else "UNCLEAR")

    if form in ("given", "despite", "upon", "without", "with"):
        if clause_initial and followed_by_comma:
            return "CLAUSE_INITIAL_PLAUSIBLE"
        if NP_HEAD.match(after) and not clause_initial:
            return "NON_CONNECTIVE_LIKELY"
        if not clause_initial:
            return "NON_CONNECTIVE_LIKELY"
        return "UNCLEAR"

    if form == "particularly":
        # Calibrated against the manual validation: both sampled `particularly`
        # cases were clause-initial by the test above, and both were judged NOT
        # discourse relations. In this corpus it is a focus adverb scoping over
        # a constituent ("particularly when X"), not a clause-level connective,
        # which is also why PDTB does not list it. Treated as non-connective.
        return "NON_CONNECTIVE_LIKELY"

    if form == "eventually":
        # Temporal adverbial: discourse-level only at a clause boundary.
        if clause_initial:
            return "CLAUSE_INITIAL_PLAUSIBLE"
        return "NON_CONNECTIVE_LIKELY"

    return "UNCLEAR"


def analyse_coverage_gap(dimlex_aligned: pd.DataFrame) -> pd.DataFrame:
    """Per-form triage of every NOT_A_CANDIDATE occurrence."""
    from src.justification_analysis.comparison.discourse_comparison import (
        COVERAGE_STATUSES,
    )

    gap = dimlex_aligned.loc[
        dimlex_aligned["alignment_status"].isin(COVERAGE_STATUSES)
    ].copy()

    starts, ends = [], []
    for row in gap.itertuples(index=False):
        sentence = row.sentence_text if isinstance(row.sentence_text, str) else ""
        # Offsets in the table are justification-relative; re-locate inside the
        # sentence so the syntactic tests see the right neighbourhood.
        match = re.search(rf"(?<!\w){re.escape(str(row.marker))}(?!\w)",
                          sentence, re.I)
        starts.append(match.start() if match else -1)
        ends.append(match.end() if match else -1)
    gap["sent_start"] = starts
    gap["sent_end"] = ends

    gap["triage"] = [
        classify_occurrence(r.marker, r.sentence_text, r.sent_start, r.sent_end)
        if r.sent_start >= 0 else "UNCLEAR"
        for r in gap.itertuples(index=False)
    ]
    return gap


def coverage_gap_summary(gap: pd.DataFrame) -> pd.DataFrame:
    table = (
        gap.groupby(["marker", "category"])["triage"]
        .value_counts().unstack(fill_value=0)
    )
    for column in ("CLAUSE_INITIAL_PLAUSIBLE", "NON_CONNECTIVE_LIKELY", "UNCLEAR"):
        if column not in table.columns:
            table[column] = 0
    table["total"] = table.sum(axis=1)
    table["pct_plausible"] = (
        100 * table["CLAUSE_INITIAL_PLAUSIBLE"] / table["total"]
    ).round(1)
    return table.sort_values("total", ascending=False)
