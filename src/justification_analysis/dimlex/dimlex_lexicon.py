"""Shared DiMLex-Eng resource parsing and lexical matching.

Extracted VERBATIM from `1_dimlex_justification_analysis.ipynb` so that the
notebook and the discopy comparison call one implementation. Notebook 2 no
longer requires notebook 1 to have been executed by hand in order to obtain a
valid occurrence table.

The matching rules are FROZEN. `MAX_COMPONENT_GAP_TOKENS = 15` was chosen
during pipeline development against the BASE corpus, whose justification
sentences average 21.1 word tokens. It is a fixed methodological choice,
applied identically to base and fine-tuned outputs, and must NOT be
recalibrated per stage. Sentence-length statistics remain a descriptive
diagnostic; they no longer feed back into the threshold.

DiMLex is a lexical COVERAGE DIAGNOSTIC for the native-discopy pipeline. It is
not a substantive discourse result and is not part of the RQ2 findings.

Segmentation comes from `src.utils.sentences`, shared with the parser and the
annotator, so a discontinuous connective can never be assembled across a
boundary the rest of the project treats as a boundary.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.sentences import count_sentences, split_sentences

# The token pattern used for word counts, marker matching and the
# component-gap measure. Identical to the one the discourse density uses; NOT
# the segmenter's internal has-a-word-character predicate, which shares the
# name in src.utils.sentences and does a different job.
WORD_PATTERN = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", flags=re.UNICODE)


# Two different sizes are in circulation for DiMLex-Eng and the notebook
# documents both rather than silently picking one:
#
#   PAPER_REPORTED_DIMLEX_ENTRIES - the count quoted in the DiMLex-Eng paper;
#   RELEASED_DIMLEX_ENTRY_COUNT   - what the public release actually ships,
#                                   checked against the upstream repository.
#
# The two do not agree, and the released lexicon is the only one that can be
# matched against, so it is the authoritative number for every count below.
# The assert further down pins the released size, so a truncated or swapped
# copy of the XML fails loudly instead of quietly shrinking the lexicon.
PAPER_REPORTED_DIMLEX_ENTRIES = 149


RELEASED_DIMLEX_ENTRY_COUNT = 142


VALID_TOP_LEVEL_CATEGORIES = {
    "Comparison",
    "Contingency",
    "Expansion",
    "Temporal",
}


VALID_CATEGORIES = {
    "Contingency",
    "Comparison",
    "Expansion",
    "Temporal",
}


# marker -> list of ordered component tuples, for the discontinuous matcher.
DISCONTINUOUS_COMPONENTS = {}


MANUAL_OVERRIDES = {
}


# ------------------------------------------------------------
# Conservative pairing rule for discontinuous connectives
# ------------------------------------------------------------
#
# A discontinuous connective ("if ... then", "either ... or") is accepted only
# when all of the following hold:
#
#   1. every component occurs inside ONE deterministically segmented
#      sentence, so a match can never straddle a sentence boundary;
#   2. the components occur in the order DiMLex documents them;
#   3. the component spans do not overlap each other;
#   4. at most MAX_COMPONENT_GAP_TOKENS word tokens separate one component
#      from the next.
#
# Rule 4 is what keeps two unrelated uses of common words ("if", "or",
# "then") from being paired at opposite ends of a long sentence. The value is
# set from the corpus: justification sentences here have a mean of 21.1 and a
# median of 21.0 word tokens, so a 15-token ceiling keeps both components
# inside a single clause pair rather than letting one connective span a whole
# sentence, while still covering the long antecedents that "if X, then Y"
# actually produces in this corpus. The diagnostic section prints the full
# sensitivity of the counts to this value, and every accepted match reports
# its own gap so wide pairings can be inspected by hand.
MAX_COMPONENT_GAP_TOKENS = 15


# Gaps at or above this share of the ceiling are flagged for manual review in
# the diagnostic table. They are still counted; the flag is advisory.
WIDE_GAP_REVIEW_THRESHOLD = 10


def extract_component_variants(entry):
    """Read the discontinuous structure of one DiMLex entry out of the XML.

    DiMLex-Eng encodes discontinuous connectives in two different ways, and
    both are parsed here rather than hard-coded marker by marker:

    1. `<orth type="discont">` with one `<part>` per component, given in the
       order the components have to appear (`If ... then`);
    2. a single `<part>` whose text joins the components with a slash
       (`not only/but`). A handful of English entries are discontinuous in
       fact but carry `type="cont"` and use this notation.

    Returns one component tuple per orthographic variant, lower-cased,
    de-duplicated, document order preserved. An empty list means the entry is
    an ordinary contiguous connective.
    """
    variants = []

    for orthography in entry.findall("./orths/orth"):
        parts = [
            (part.text or "").strip()
            for part in orthography.findall("part")
        ]
        parts = [part for part in parts if part]

        if (
            orthography.get("type") == "discont"
            and len(parts) >= 2
        ):
            variants.append(
                tuple(part.lower() for part in parts)
            )
            continue

        for part in parts:
            if "/" not in part:
                continue

            components = [
                component.strip().lower()
                for component in part.split("/")
                if component.strip()
            ]

            if len(components) >= 2:
                variants.append(tuple(components))

    unique_variants = []

    for variant in variants:
        if variant not in unique_variants:
            unique_variants.append(variant)

    return unique_variants


def format_components(component_variants):
    """Readable `if ... then` rendering of the first component variant."""
    if not component_variants:
        return ""

    return " ... ".join(component_variants[0])


def make_phrase_pattern(marker):
    """Case-insensitive, word-bounded pattern for one contiguous phrase.

    Also used for the individual components of a discontinuous connective, so
    a component only matches a whole word or whole phrase.
    """
    tokens = marker.split()

    body = r"\s+".join(
        re.escape(token)
        for token in tokens
    )

    return re.compile(
        rf"(?<!\w){body}(?!\w)",
        flags=re.IGNORECASE,
    )


def sentence_spans(text):
    """Character spans of the deterministic sentences of `text`.

    split_sentences() returns the sentences themselves, stripped; they are
    verbatim substrings of the justification, so each one is located by
    scanning forward from the end of the previous match. Segmentation is not
    re-implemented here - this only recovers the offsets of the units the
    shared segmenter already produced.
    """
    spans = []
    cursor = 0

    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)

        if start < 0:
            start = text.find(sentence)

        if start < 0:
            continue

        spans.append((start, start + len(sentence)))
        cursor = start + len(sentence)

    if not spans and text.strip():
        # No terminal punctuation at all: the whole justification is one unit.
        spans = [(0, len(text))]

    return spans


def find_discontinuous_matches(
    text,
    marker,
    component_variants,
    max_gap=MAX_COMPONENT_GAP_TOKENS,
):
    """Every accepted occurrence of one discontinuous connective in `text`.

    Within each sentence the scan is leftmost and greedy: the first component
    is taken at its earliest position, each following component at its
    earliest position strictly after the previous one, and the scan then
    resumes after the last component so a sentence can hold more than one
    instance of the same construction. Taking the earliest admissible
    position for every component means the tightest available pairing is the
    one that gets tested against the gap ceiling.

    Returns one record per accepted construction. The record spans the whole
    construction, but `component_spans` keeps the individual component spans:
    only those are reserved against the contiguous matcher, which is what
    leaves an unrelated marker sitting in the gap free to be counted.
    """
    compiled_variants = [
        [make_phrase_pattern(component) for component in variant]
        for variant in component_variants
    ]

    matches = []

    for sentence_start, sentence_end in sentence_spans(text):
        sentence = text[sentence_start:sentence_end]

        for variant_index, patterns in enumerate(compiled_variants):
            cursor = 0

            while True:
                local_spans = []
                accepted = True

                position = cursor

                for component_index, pattern in enumerate(patterns):
                    component_match = pattern.search(sentence, position)

                    if component_match is None:
                        accepted = False
                        break

                    if component_index > 0:
                        gap_tokens = len(
                            WORD_PATTERN.findall(
                                sentence[
                                    local_spans[-1][1]
                                    : component_match.start()
                                ]
                            )
                        )

                        if gap_tokens > max_gap:
                            accepted = False
                            break

                    local_spans.append(
                        (component_match.start(), component_match.end())
                    )
                    position = component_match.end()

                if not accepted:
                    break

                total_gap = len(
                    WORD_PATTERN.findall(
                        sentence[local_spans[0][1] : local_spans[-1][0]]
                    )
                )

                matches.append(
                    {
                        "marker": marker,
                        "matched_text": " ... ".join(
                            sentence[start:end]
                            for start, end in local_spans
                        ),
                        "start": sentence_start + local_spans[0][0],
                        "end": sentence_start + local_spans[-1][1],
                        "component_spans": [
                            (sentence_start + start, sentence_start + end)
                            for start, end in local_spans
                        ],
                        "component_variant": " ... ".join(
                            component_variants[variant_index]
                        ),
                        "gap_tokens": total_gap,
                        "sentence": sentence,
                        "match_type": "discontinuous",
                    }
                )

                cursor = local_spans[-1][1]

    return matches


def compile_contiguous_lexicon(lexicon_df):
    """Attach compiled patterns and sort so longer phrases are tried first."""
    compiled_df = lexicon_df.copy()

    compiled_df["pattern"] = compiled_df["marker"].map(make_phrase_pattern)
    compiled_df["marker_length"] = compiled_df["marker"].str.len()

    # Match longer phrases before shorter phrases.
    return (
        compiled_df
        .sort_values(
            ["marker_length", "marker"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def find_markers(text, contiguous_lexicon, discontinuous_lexicon):
    """All DiMLex marker occurrences in one justification.

    Resolution order:

    1. valid discontinuous constructions are detected first, and each counts
       as exactly ONE marker occurrence;
    2. the component spans of an accepted construction are reserved, so the
       same words cannot also be counted as standalone contiguous markers;
    3. contiguous candidates are then resolved with the existing
       longest-span-first, no-overlap rule, seeded with those reservations.

    Only the component spans are reserved, never the gap between them: an
    unrelated marker inside the gap stays eligible, and unrelated occurrences
    of the component words elsewhere in the sentence stay eligible too.
    """
    accepted = []
    occupied_spans = []

    # ---- 1. discontinuous constructions -------------------------------
    discontinuous_candidates = []

    for row in discontinuous_lexicon.itertuples(index=False):
        for match in find_discontinuous_matches(
            text,
            row.marker,
            row.component_variants,
        ):
            match["category"] = row.assigned_category
            discontinuous_candidates.append(match)

    # Earliest first, and among equal starts the tightest construction first,
    # so a compact pairing is never displaced by a looser one.
    discontinuous_candidates.sort(
        key=lambda item: (
            item["start"],
            item["end"] - item["start"],
            item["marker"],
        )
    )

    for candidate in discontinuous_candidates:
        overlaps = any(
            component_start < existing_end
            and component_end > existing_start
            for component_start, component_end in candidate["component_spans"]
            for existing_start, existing_end in occupied_spans
        )

        if overlaps:
            continue

        accepted.append(candidate)
        occupied_spans.extend(candidate["component_spans"])

    # ---- 2. contiguous candidates -------------------------------------
    contiguous_candidates = []

    for row in contiguous_lexicon.itertuples(index=False):
        for match in row.pattern.finditer(text):
            contiguous_candidates.append(
                {
                    "marker": row.marker,
                    "category": row.assigned_category,
                    "matched_text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "component_spans": [(match.start(), match.end())],
                    "component_variant": "",
                    "gap_tokens": 0,
                    "sentence": "",
                    "match_type": "contiguous",
                }
            )

    # Prevent overlapping phrases from being counted twice.
    contiguous_candidates.sort(
        key=lambda item: (
            -(item["end"] - item["start"]),
            item["start"],
        )
    )

    for candidate in contiguous_candidates:
        overlaps = any(
            candidate["start"] < existing_end
            and candidate["end"] > existing_start
            for existing_start, existing_end in occupied_spans
        )

        if overlaps:
            continue

        accepted.append(candidate)
        occupied_spans.append(
            (candidate["start"], candidate["end"])
        )

    return sorted(
        accepted,
        key=lambda item: item["start"],
    )


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------
#
# The marker assignment table is a STABLE RESOURCE artifact: it is derived from
# the DiMLex XML alone (142 released entries) and does not depend on any
# corpus, so it is committed and loaded rather than rebuilt on every run.
# Notebook 1 remains the place that regenerates and documents it from the XML;
# everything else consumes it here. That is what lets notebook 2 obtain a valid
# occurrence table without notebook 1 having been executed by hand.

DIMLEX_DIR = Path(__file__).resolve().parent
MARKER_ASSIGNMENTS_PATH = DIMLEX_DIR / "dimlex_marker_assignments.csv"
DIMLEX_XML_PATH = DIMLEX_DIR / "en_dimlex.xml"


def load_marker_assignments(path: Path = None) -> pd.DataFrame:
    """The committed DiMLex marker assignment table."""
    path = Path(path or MARKER_ASSIGNMENTS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"DiMLex marker assignments not found: {path}\n"
            f"Regenerate them by running notebook 1 "
            f"(1_dimlex_justification_analysis.ipynb)."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    assert len(frame) == RELEASED_DIMLEX_ENTRY_COUNT, (
        f"expected {RELEASED_DIMLEX_ENTRY_COUNT} released DiMLex entries, "
        f"found {len(frame)} - the lexicon changed")
    return frame


def build_matching_lexicons(assignments: pd.DataFrame = None):
    """Split the assignment table into the two matcher inputs.

    Contiguous entries go to the phrase matcher; discontinuous entries go to
    the component matcher with their ordered component variants restored from
    the JSON column the notebook exported.
    """
    import json as _json

    assignments = (assignments if assignments is not None
                   else load_marker_assignments())

    contiguous = assignments.loc[
        assignments["use_for_contiguous_matching"],
        ["marker", "assigned_category"]].copy()

    discontinuous = assignments.loc[
        assignments["use_for_discontinuous_matching"],
        ["marker", "assigned_category", "components"]].copy()
    discontinuous["component_variants"] = [
        [tuple(variant) for variant in _json.loads(payload)]
        for payload in assignments.loc[
            assignments["use_for_discontinuous_matching"],
            "component_variants_json"]
    ]

    return compile_contiguous_lexicon(contiguous), discontinuous


# ---------------------------------------------------------------------------
# Matching against a corpus
# ---------------------------------------------------------------------------

def match_corpus(corpus: pd.DataFrame, contiguous_lexicon: pd.DataFrame = None,
                 discontinuous_lexicon: pd.DataFrame = None):
    """Apply the frozen matcher to every justification of a corpus.

    Identical accounting to the notebook: a contiguous phrase is one
    occurrence; an accepted discontinuous construction is ONE occurrence of its
    DiMLex entry, and its component spans are withheld from the contiguous
    matcher so the same words are not counted twice.

    Returns (metrics, occurrences), one row per justification and one row per
    occurrence.
    """
    if contiguous_lexicon is None or discontinuous_lexicon is None:
        contiguous_lexicon, discontinuous_lexicon = build_matching_lexicons()

    metric_rows = []
    occurrence_rows = []

    for row in corpus.itertuples(index=False):
        text = row.justification
        marker_matches = find_markers(text, contiguous_lexicon,
                                      discontinuous_lexicon)

        n_words = len(WORD_PATTERN.findall(text))
        n_sentences = count_sentences(text)
        n_markers = len(marker_matches)

        metric_rows.append({
            "justification_id": row.justification_id,
            "model": row.model,
            "game_id": row.game_id,
            "run_label": row.run_label,
            "run_number": row.run_number,
            "decoding_group": row.decoding_group,
            "n_words": n_words,
            "n_sentences": n_sentences,
            "words_per_sentence": (n_words / n_sentences
                                   if n_sentences > 0 else np.nan),
            "n_markers": n_markers,
            "markers_per_100_words": (100 * n_markers / n_words
                                      if n_words > 0 else np.nan),
            "has_marker": int(n_markers > 0),
        })

        for marker_match in marker_matches:
            occurrence_rows.append({
                "justification_id": row.justification_id,
                "model": row.model,
                "game_id": row.game_id,
                "run_label": row.run_label,
                "run_number": row.run_number,
                "decoding_group": row.decoding_group,
                **marker_match,
            })

    metrics = pd.DataFrame(metric_rows)
    occurrences = pd.DataFrame(occurrence_rows)

    # Every accepted discontinuous match keeps its components ordered and
    # non-overlapping; this is the notebook's own assertion, kept.
    for check_row in occurrences.loc[
            occurrences["match_type"].eq("discontinuous")].itertuples(index=False):
        spans = check_row.component_spans
        assert all(earlier[1] <= later[0]
                   for earlier, later in zip(spans, spans[1:])), \
            f"component order violated for {check_row.marker!r}"

    return metrics, occurrences


def attach_sentence_ids(occurrences: pd.DataFrame,
                        corpus: pd.DataFrame) -> pd.DataFrame:
    """Add 1-indexed sentence_id and sentence_text to occurrence rows.

    An occurrence always falls inside a single sentence: contiguous matches are
    one span, and the discontinuous matcher already requires every component to
    sit in one sentence. A row that cannot be placed keeps a null sentence_id
    rather than being dropped, so the export never loses an occurrence silently.
    """
    text_by_id = dict(zip(corpus["justification_id"], corpus["justification"]))
    spans_by_id = {justification_id: sentence_spans(text)
                   for justification_id, text in text_by_id.items()}

    sentence_ids = []
    sentence_texts = []
    for row in occurrences.itertuples(index=False):
        text = text_by_id[row.justification_id]
        located = (None, None)
        for index, (start, end) in enumerate(
                spans_by_id[row.justification_id], start=1):
            if start <= row.start and row.end <= end:
                located = (index, text[start:end])
                break
        sentence_ids.append(located[0])
        sentence_texts.append(located[1])

    out = occurrences.copy()
    out["sentence_id"] = sentence_ids
    out["sentence_text"] = sentence_texts
    return out
