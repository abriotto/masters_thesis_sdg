"""Compare the DiMLex lexical analysis with the discopy explicit-relation analysis.

Pure pandas: runs in the project's `sdglogs` environment. Consumes the CSV that
`run_discopy_on_justifications.py` produces in `discopy-env`; nothing here
imports TensorFlow.

The two analyses stay separate. Nothing in this module feeds discopy output
back into the DiMLex results or uses DiMLex to filter discopy output - the
point is to measure how far apart they are, not to merge them.

The central design decision is the alignment taxonomy. A DiMLex occurrence that
discopy does not report as a connective can fail for two very different
reasons, and collapsing them would make a hybrid look better justified than the
evidence supports:

  * discopy never enumerated the span as a connective candidate. Its candidate
    generator only proposes forms from a fixed 101-form lexicon, so anything
    outside that lexicon is invisible to the sense classifier. Supplying DiMLex
    candidates WOULD fix these. This is candidate-coverage evidence.

  * discopy enumerated the span and its classifier assigned NoSense. The
    candidate was seen and rejected on contextual grounds. Supplying DiMLex
    candidates would NOT change this - the same classifier would fire on the
    same span and reject it again. This is contextual-classification evidence,
    and it argues about parser quality, not inventory.

Only the first category counts toward a DiMLex-candidates -> discopy-sense
hybrid.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PDTB_TOP_LEVEL = ("Comparison", "Contingency", "Expansion", "Temporal")

# --- DiMLex occurrence alignment statuses ---------------------------------
ALIGNED_CONNECTIVE = "ALIGNED_CONNECTIVE"
CANDIDATE_REJECTED_NOSENSE = "CANDIDATE_REJECTED_NOSENSE"
NOT_A_CANDIDATE_FORM_OUTSIDE_INVENTORY = "NOT_A_CANDIDATE_FORM_OUTSIDE_INVENTORY"
NOT_A_CANDIDATE_FORM_IN_INVENTORY = "NOT_A_CANDIDATE_FORM_IN_INVENTORY"

# Statuses that are evidence about candidate COVERAGE (a hybrid would fix).
COVERAGE_STATUSES = (
    NOT_A_CANDIDATE_FORM_OUTSIDE_INVENTORY,
    NOT_A_CANDIDATE_FORM_IN_INVENTORY,
)
# Status that is evidence about contextual CLASSIFICATION (a hybrid would not).
CLASSIFICATION_STATUSES = (CANDIDATE_REJECTED_NOSENSE,)

# --- discopy occurrence alignment statuses --------------------------------
DISCOPY_ALIGNED = "ALIGNED"
DISCOPY_ONLY_FORM_OUTSIDE_DIMLEX = "DISCOPY_ONLY_FORM_OUTSIDE_DIMLEX"
DISCOPY_ONLY_DIMLEX_NO_MATCH_HERE = "DISCOPY_ONLY_DIMLEX_NO_MATCH_HERE"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def parse_char_spans(spans: str) -> List[Tuple[int, int]]:
    """`'12-14;30-34'` -> `[(12, 14), (30, 34)]`."""
    if not isinstance(spans, str) or not spans:
        return []
    out = []
    for chunk in spans.split(";"):
        start, _, end = chunk.partition("-")
        out.append((int(start), int(end)))
    return out


def normalise_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and decorate an already-loaded candidate table.

    Split out of `load_discopy_candidates` so that callers which obtain the
    table through the manifest freshness gate get the identical validation
    without reading the file a second time by path.
    """
    frame = frame.copy()
    assert (frame["relation_type"] == "Explicit").all(), \
        "the discopy table contains non-explicit relations"
    assert frame["occurrence_id"].is_unique, "duplicate discopy occurrence ids"
    frame["span_list"] = frame["char_spans"].map(parse_char_spans)
    accepted = frame.loc[frame["is_connective"]]
    bad = set(accepted["top_level"].dropna()) - set(PDTB_TOP_LEVEL)
    assert not bad, f"unexpected top-level labels: {sorted(bad)}"
    return frame


def load_discopy_candidates(path) -> pd.DataFrame:
    """Load the candidate table straight from a path.

    NOTE: this bypasses the manifest freshness gate and therefore cannot tell
    whether the file belongs to the corpus being analysed. Analyses must go
    through `pipeline.manifest.load_verified_candidates`; this remains for
    diagnostic and migration code that deliberately inspects a file as a file.
    """
    return normalise_candidates(pd.read_csv(path, encoding="utf-8-sig"))


def normalise_surface(form: str) -> str:
    """Normalise a marker/connective form for inventory comparison.

    DiMLex writes some discontinuous entries with a slash (`not only/but`)
    while discopy writes them with a space (`if then`), so slashes become
    spaces and whitespace is collapsed.
    """
    if not isinstance(form, str):
        return ""
    return " ".join(form.replace("/", " ").lower().split())


# ---------------------------------------------------------------------------
# Span alignment
# ---------------------------------------------------------------------------

def _spans_overlap(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> bool:
    return any(a0 < b1 and a1 > b0 for a0, a1 in a for b0, b1 in b)


def _span_key(spans: Sequence[Tuple[int, int]]) -> Tuple:
    return tuple(sorted(tuple(s) for s in spans))


def align_dimlex_discopy_occurrences(
    dimlex_occurrences: pd.DataFrame,
    discopy_candidates: pd.DataFrame,
    discopy_inventory: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align the two occurrence tables by character offsets.

    Alignment is by span overlap inside the same justification, never by
    surface string alone: two occurrences of `as` in one sentence are distinct
    events and only offsets can tell them apart.

    `dimlex_occurrences` must carry `justification_id`, `marker`, `category`,
    `start`, `end` and, for discontinuous matches, `span_list`.

    Returns `(dimlex_aligned, discopy_aligned)`, each the input table plus
    alignment columns.
    """
    inventory = {normalise_surface(f) for f in discopy_inventory}

    discopy_by_just: Dict[int, List[dict]] = {}
    for row in discopy_candidates.to_dict("records"):
        discopy_by_just.setdefault(row["justification_id"], []).append(row)

    dimlex_rows = []
    matched_discopy_ids = set()

    for occ in dimlex_occurrences.to_dict("records"):
        spans = occ.get("span_list") or [(occ["start"], occ["end"])]
        candidates = discopy_by_just.get(occ["justification_id"], [])

        best = None
        for cand in candidates:
            if not _spans_overlap(spans, cand["span_list"]):
                continue
            # Prefer an exact span match, then an accepted candidate, then
            # the largest overlap, so a bare `if` never outranks `if ... then`
            # when both overlap the same DiMLex span.
            exact = _span_key(spans) == _span_key(cand["span_list"])
            score = (exact, bool(cand["is_connective"]), len(cand["span_list"]))
            if best is None or score > best[0]:
                best = (score, cand)

        if best is None:
            form = normalise_surface(occ["marker"])
            status = (
                NOT_A_CANDIDATE_FORM_IN_INVENTORY
                if form in inventory
                else NOT_A_CANDIDATE_FORM_OUTSIDE_INVENTORY
            )
            occ.update(
                alignment_status=status,
                discopy_occurrence_id=np.nan,
                discopy_surface=None,
                discopy_raw_sense=None,
                discopy_top_level=None,
                discopy_confidence=np.nan,
                category_changed=np.nan,
            )
        else:
            cand = best[1]
            matched_discopy_ids.add(cand["occurrence_id"])
            if cand["is_connective"]:
                status = ALIGNED_CONNECTIVE
                changed = cand["top_level"] != occ["category"]
            else:
                status = CANDIDATE_REJECTED_NOSENSE
                changed = np.nan
            occ.update(
                alignment_status=status,
                discopy_occurrence_id=cand["occurrence_id"],
                discopy_surface=cand["connective_surface"],
                discopy_raw_sense=cand["raw_sense"],
                discopy_top_level=cand["top_level"],
                discopy_confidence=cand["confidence"],
                category_changed=changed,
            )
        dimlex_rows.append(occ)

    dimlex_aligned = pd.DataFrame(dimlex_rows)
    dimlex_aligned["is_coverage_evidence"] = (
        dimlex_aligned["alignment_status"].isin(COVERAGE_STATUSES)
    )
    dimlex_aligned["is_classification_evidence"] = (
        dimlex_aligned["alignment_status"].isin(CLASSIFICATION_STATUSES)
    )

    # --- the discopy side -------------------------------------------------
    dimlex_forms = {normalise_surface(m) for m in dimlex_occurrences["marker"]}
    discopy_aligned = discopy_candidates.copy()
    discopy_aligned["aligned_to_dimlex"] = (
        discopy_aligned["occurrence_id"].isin(matched_discopy_ids)
    )

    def discopy_status(row):
        if row["aligned_to_dimlex"]:
            return DISCOPY_ALIGNED
        form = normalise_surface(row["candidate_surface"])
        return (
            DISCOPY_ONLY_DIMLEX_NO_MATCH_HERE
            if form in dimlex_forms
            else DISCOPY_ONLY_FORM_OUTSIDE_DIMLEX
        )

    discopy_aligned["alignment_status"] = discopy_aligned.apply(discopy_status, axis=1)
    return dimlex_aligned, discopy_aligned


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compare_connective_inventories(
    dimlex_aligned: pd.DataFrame,
    discopy_aligned: pd.DataFrame,
) -> pd.DataFrame:
    """Per surface form: DiMLex counts, discopy counts, and why they differ.

    The two "missing" columns are kept apart on purpose:
      `n_coverage_gap`   - never enumerated as a discopy candidate;
      `n_rejected_nosense` - enumerated and rejected on contextual grounds.
    """
    dim = dimlex_aligned.copy()
    dim["form"] = dim["marker"].map(normalise_surface)

    per_form = dim.groupby("form").agg(
        n_dimlex=("form", "size"),
        n_aligned=("alignment_status", lambda s: int((s == ALIGNED_CONNECTIVE).sum())),
        n_rejected_nosense=(
            "alignment_status",
            lambda s: int((s == CANDIDATE_REJECTED_NOSENSE).sum()),
        ),
        n_coverage_gap=(
            "alignment_status", lambda s: int(s.isin(COVERAGE_STATUSES).sum()),
        ),
        n_outside_inventory=(
            "alignment_status",
            lambda s: int((s == NOT_A_CANDIDATE_FORM_OUTSIDE_INVENTORY).sum()),
        ),
        n_in_inventory_not_enumerated=(
            "alignment_status",
            lambda s: int((s == NOT_A_CANDIDATE_FORM_IN_INVENTORY).sum()),
        ),
        n_category_changed=("category_changed", lambda s: int(s.fillna(False).sum())),
        dimlex_category=("category", lambda s: s.mode().iat[0] if len(s) else None),
    )

    dis = discopy_aligned.copy()
    dis["form"] = dis["candidate_surface"].map(normalise_surface)
    accepted = dis.loc[dis["is_connective"]]
    per_form_discopy = accepted.groupby("form").agg(
        n_discopy_accepted=("form", "size"),
        n_discopy_only=(
            "alignment_status", lambda s: int((s != DISCOPY_ALIGNED).sum()),
        ),
    )
    per_candidate = dis.groupby("form").agg(n_discopy_candidates=("form", "size"))

    out = (
        per_form.join(per_form_discopy, how="outer")
        .join(per_candidate, how="outer")
        .fillna(0)
    )
    int_cols = [c for c in out.columns if c.startswith("n_")]
    out[int_cols] = out[int_cols].astype(int)
    out["pct_dimlex_retained"] = np.where(
        out["n_dimlex"] > 0, 100 * out["n_aligned"] / out["n_dimlex"], np.nan
    )
    return out.sort_values("n_dimlex", ascending=False)


def sense_change_crosstab(dimlex_aligned: pd.DataFrame) -> pd.DataFrame:
    """DiMLex fixed majority category x discopy contextual top level.

    Restricted to occurrences both systems call a connective, since a category
    comparison is only meaningful where both assigned one.
    """
    both = dimlex_aligned.loc[
        dimlex_aligned["alignment_status"] == ALIGNED_CONNECTIVE
    ]
    return pd.crosstab(
        both["category"].rename("dimlex_majority"),
        both["discopy_top_level"].rename("discopy_contextual"),
        margins=True,
        margins_name="TOTAL",
    )


def coverage_vs_classification_summary(dimlex_aligned: pd.DataFrame) -> pd.DataFrame:
    """Headline split of DiMLex occurrences discopy did not accept."""
    counts = dimlex_aligned["alignment_status"].value_counts()
    total = int(counts.sum())
    rows = []
    for status in (
        ALIGNED_CONNECTIVE,
        CANDIDATE_REJECTED_NOSENSE,
        NOT_A_CANDIDATE_FORM_OUTSIDE_INVENTORY,
        NOT_A_CANDIDATE_FORM_IN_INVENTORY,
    ):
        n = int(counts.get(status, 0))
        if status == ALIGNED_CONNECTIVE:
            evidence = "-"
        elif status in CLASSIFICATION_STATUSES:
            evidence = "contextual classification (hybrid would NOT fix)"
        else:
            evidence = "candidate coverage (hybrid WOULD fix)"
        rows.append(
            {
                "alignment_status": status,
                "n": n,
                "pct_of_dimlex": 100 * n / total if total else np.nan,
                "evidence_about": evidence,
            }
        )
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Manual validation sample
# ---------------------------------------------------------------------------

# Deliberately a targeted quality check, NOT a benchmark of parser performance.
# There is no weighting, no population-level precision estimate and no
# confidence interval: the sample is purposively balanced for inspection, so
# raw counts are the only thing it supports. Results are reported as plain
# fractions ("27/30 accepted relations judged correct").
#
# The 20 not-accepted items are a missed-relation / coverage diagnostic over
# candidates DiMLex identified independently. They are NOT a recall estimate
# and no full-corpus recall figure may be derived from them: the justifications
# are not exhaustively gold-annotated, so relations outside the DiMLex
# inventory - and relations carried by no lexical marker at all - are invisible
# to this design.

AMBIGUOUS_FORMS = ("as", "for", "and", "or", "then", "since", "while")

VALIDATION_COLUMNS = [
    "validation_id", "failure_type",
    "model", "game_id", "run_label", "decoding_group",
    "justification_id", "sentence_id", "sentence_text",
    "marker", "char_spans", "discopy_raw_sense", "discopy_top_level",
    "discopy_confidence", "dimlex_category",
    "manual_is_connective", "manual_top_level_category",
    "manual_valid_relation_missed_by_discopy", "notes",
]


def _spread_sample(frame, n, seed, priority_cols):
    """Take `n` rows, spreading across `priority_cols` before filling at random."""
    if len(frame) <= n:
        return frame.copy()
    rng = np.random.RandomState(seed)
    picked = []
    remaining = frame.copy()
    # One pass per priority column value, so every group is represented before
    # any group gets a second slot.
    for col in priority_cols:
        for value in remaining[col].dropna().unique():
            if len(picked) >= n:
                break
            pool = remaining.loc[remaining[col] == value]
            pool = pool.loc[~pool.index.isin([p for p in picked])]
            if len(pool):
                picked.append(pool.sample(1, random_state=rng).index[0])
    leftover = remaining.loc[~remaining.index.isin(picked)]
    still_needed = max(0, n - len(picked))
    if still_needed and len(leftover):
        picked += list(
            leftover.sample(min(still_needed, len(leftover)),
                            random_state=rng).index
        )
    return frame.loc[picked[:n]].copy()


def create_validation_sample(
    dimlex_aligned: pd.DataFrame,
    discopy_aligned: pd.DataFrame,
    seed: int = 20260825,
    n_accepted: int = 30,
    n_rejected_nosense: int = 10,
    n_not_enumerated: int = 10,
) -> pd.DataFrame:
    """Build the 50-item manual validation sheet.

    30 accepted discopy explicit relations (is the span really a connective,
    and is the top-level category right) plus 20 DiMLex candidates discopy did
    not accept, split evenly between the two failure modes so classification
    and coverage stay separately scoreable.
    """
    accepted = discopy_aligned.loc[discopy_aligned["is_connective"]].copy()
    accepted["form"] = accepted["candidate_surface"].map(normalise_surface)
    accepted["is_ambiguous_form"] = accepted["form"].isin(AMBIGUOUS_FORMS)
    # Low/high confidence both need representation.
    accepted["confidence_band"] = np.where(
        accepted["confidence"] < 0.50, "low",
        np.where(accepted["confidence"] < 0.90, "mid", "high"),
    )
    accepted_sample = _spread_sample(
        accepted, n_accepted, seed,
        ["top_level", "confidence_band", "is_ambiguous_form", "form"],
    )
    accepted_sample["failure_type"] = "accepted"
    accepted_sample["marker"] = accepted_sample["connective_surface"]
    accepted_sample["dimlex_category"] = None
    # The discopy table names these without the prefix the sheet uses.
    accepted_sample["discopy_raw_sense"] = accepted_sample["raw_sense"]
    accepted_sample["discopy_top_level"] = accepted_sample["top_level"]
    accepted_sample["discopy_confidence"] = accepted_sample["confidence"]

    rejected = dimlex_aligned.loc[
        dimlex_aligned["alignment_status"] == CANDIDATE_REJECTED_NOSENSE
    ].copy()
    not_enum = dimlex_aligned.loc[
        dimlex_aligned["alignment_status"].isin(COVERAGE_STATUSES)
    ].copy()

    for frame in (rejected, not_enum):
        frame["form"] = frame["marker"].map(normalise_surface)
        frame["is_ambiguous_form"] = frame["form"].isin(AMBIGUOUS_FORMS)

    rejected_sample = _spread_sample(
        rejected, n_rejected_nosense, seed + 1,
        ["category", "is_ambiguous_form", "form"],
    )
    rejected_sample["failure_type"] = "rejected_nosense"

    not_enum_sample = _spread_sample(
        not_enum, n_not_enumerated, seed + 2,
        ["category", "is_ambiguous_form", "form"],
    )
    not_enum_sample["failure_type"] = "not_enumerated"

    for frame in (rejected_sample, not_enum_sample):
        frame["dimlex_category"] = frame["category"]
        frame["discopy_raw_sense"] = frame.get("discopy_raw_sense")
        frame["discopy_top_level"] = frame.get("discopy_top_level")
        frame["discopy_confidence"] = frame.get("discopy_confidence")

    sample = pd.concat(
        [accepted_sample, rejected_sample, not_enum_sample], ignore_index=True
    )
    for column in ("manual_is_connective", "manual_top_level_category",
                   "manual_valid_relation_missed_by_discopy", "notes"):
        sample[column] = ""
    sample.insert(0, "validation_id", np.arange(1, len(sample) + 1))

    for column in VALIDATION_COLUMNS:
        if column not in sample.columns:
            sample[column] = None

    sample = sample[VALIDATION_COLUMNS]

    assert sample["validation_id"].is_unique
    assert set(sample["failure_type"]) <= {
        "accepted", "rejected_nosense", "not_enumerated"
    }
    return sample
