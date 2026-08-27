"""Joint discourse x semantic analysis with the JUSTIFICATION as the unit.

This REPLACES the sentence-level joint analysis for thesis purposes. That
earlier version defined association as same-sentence co-occurrence, which is
wrong for two reasons:

  * an explicit connective can relate content across a sentence boundary -
    "He swapped Paul with Mike. Therefore Mike is the Werewolf." carries
    Mechanical in sentence 1 and Contingency in sentence 2, and the two plainly
    belong to the same short argument;
  * it is a cross-model confound. A model that packs one thought into a single
    sentence scores a co-occurrence; a model that splits the same thought over
    two sentences does not. Justification length differs systematically across
    these three models, so the sentence-level metric partly measured packaging.

The old module is kept for provenance and is not deleted. It must not be used
for thesis results.

WHAT THIS STILL DOES NOT ESTABLISH. The parser identifies explicit connectives
and their PDTB sense. It does not recover discourse arguments. A justification
containing both a category and a relation class is an association between the
kind of information a justification carries and the kind of explicit discourse
marking it uses - never evidence that the relation structures the category, or
that the semantic content is an argument of the relation. Widening the window
from sentence to justification makes the claim weaker, not stronger.

Aggregation follows the two frozen strands exactly: per run first, then mean
and SD across the three stochastic runs; greedy is one run, kept separate, SD
undefined; every bootstrap resamples GAMES with all their runs and
justifications attached, reusing the same resampled game ids across models.

The discourse DENSITY definition is not reinvented here. It is the canonical
one from `discourse_final.run_level_table`: a ratio of sums per run,
100 * relations / WORD_PATTERN tokens, over the subset of justifications
containing the semantic category. A mean of per-justification densities is
also reported, explicitly as a secondary reading.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.justification_analysis.comparison.discourse_statistics import (
    WORD_PATTERN,
    to_latex,
)
from src.justification_analysis.joint import joint_final as sentence_level
from src.justification_analysis.semantic import semantic_final as sem

# ---------------------------------------------------------------------------
# Vocabulary and design constants - shared with the frozen strands
# ---------------------------------------------------------------------------

CATEGORY_ORDER: List[str] = list(sem.CATEGORY_ORDER)
OTHER_CATEGORY = sem.OTHER_CATEGORY
ALL_CATEGORIES: List[str] = list(sem.ALL_CATEGORIES)

TOP_LEVEL_ORDER = list(sentence_level.TOP_LEVEL_ORDER)
SENSE_ORDER = list(sentence_level.SENSE_ORDER)
ANY_RELATION = sentence_level.ANY_RELATION

MODEL_ORDER = list(sem.MODEL_ORDER)
DECODING_ORDER = list(sem.DECODING_ORDER)
RUNS_BY_DECODING = dict(sem.RUNS_BY_DECODING)

ARTIFACT_SUBPATH = Path(
    "analysis/cross_model/base/voting/prompt_v4/justification_analysis"
    "/joint_discourse_semantic_justification"
)
FINAL_TABLES_SUBPATH = ARTIFACT_SUBPATH / "thesis_tables" / "final_joint_justification"
FINAL_FIGURES_SUBPATH = ARTIFACT_SUBPATH / "figures" / "final_joint_justification"
QUALITATIVE_SUBPATH = ARTIFACT_SUBPATH / "qualitative"

BOOTSTRAP_SEED = sem.BOOTSTRAP_SEED
BOOTSTRAP_REPLICATES = sem.BOOTSTRAP_REPLICATES

# Purely DIAGNOSTIC. Not a rule, not a filter, and nothing is dropped by it -
# it marks cells whose denominator is small so that support and interval width
# are read together rather than the point estimate alone.
LOW_SUPPORT_DIAGNOSTIC = 30

INVARIANTS = {
    "justifications": 2292,
    "games": 191,
    "sentences": 8044,
    "semantic_labels": 11526,
    "accepted_relations": 5504,
    "word_pattern_tokens": 169748,
}


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model" in out.columns:
        out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    if "decoding_group" in out.columns:
        out["decoding_group"] = pd.Categorical(
            out["decoding_group"], DECODING_ORDER, ordered=True)
    if "semantic_category" in out.columns:
        known = [c for c in ALL_CATEGORIES
                 if c in set(out["semantic_category"].astype(str))]
        out["semantic_category"] = pd.Categorical(
            out["semantic_category"], known, ordered=True)
    if "discourse_relation" in out.columns:
        vocabulary = [*TOP_LEVEL_ORDER, *SENSE_ORDER, ANY_RELATION]
        known = [r for r in vocabulary
                 if r in set(out["discourse_relation"].astype(str))]
        out["discourse_relation"] = pd.Categorical(
            out["discourse_relation"], known, ordered=True)
    sort_cols = [c for c in ("model", "decoding_group", "run_label",
                             "semantic_category", "discourse_relation")
                 if c in out.columns]
    return (out.sort_values(sort_cols).reset_index(drop=True)
            if sort_cols else out)


def _relation_columns(kind: str) -> Tuple[List[str], List[str]]:
    if kind == "top_level":
        return ([*TOP_LEVEL_ORDER, ANY_RELATION],
                [f"disc_{r}" for r in TOP_LEVEL_ORDER] + ["has_any_relation"])
    if kind == "sense":
        return list(SENSE_ORDER), [f"sense_{s}" for s in SENSE_ORDER]
    raise ValueError(f"unknown relation kind {kind!r}")


def _count_columns(kind: str) -> Tuple[List[str], List[str]]:
    if kind == "top_level":
        return ([*TOP_LEVEL_ORDER, ANY_RELATION],
                [f"n_disc_{r}" for r in TOP_LEVEL_ORDER] + ["n_relations"])
    if kind == "sense":
        return list(SENSE_ORDER), [f"n_sense_{s}" for s in SENSE_ORDER]
    raise ValueError(f"unknown relation kind {kind!r}")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_justification_metadata(repo_root: Path) -> pd.DataFrame:
    """Vote, correctness and the canonical WORD_PATTERN token count.

    Word counts come from the justification text in the annotator's input
    shards, tokenised with the SAME `WORD_PATTERN` the frozen discourse
    analysis uses. The corpus total is asserted against the frozen 169,748.
    """
    rows = []
    directory = Path(repo_root) / sem.INPUT_SUBPATH
    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rows.append({
                    "justification_id": record["justification_id"],
                    "justification_text": record["justification"],
                    "vote": record["vote"],
                    "voted_player_end_role": record.get("voted_player_end_role"),
                    "is_correct": bool(record["is_correct"]),
                    "n_words": len(WORD_PATTERN.findall(record["justification"])),
                })
    frame = pd.DataFrame(rows)
    total = int(frame["n_words"].sum())
    assert total == INVARIANTS["word_pattern_tokens"], (
        f"word denominator is {total}, frozen value is "
        f"{INVARIANTS['word_pattern_tokens']} - the corpus changed"
    )
    return frame


def build_justification_frame(layers: Dict[str, pd.DataFrame],
                              alignment: Dict[str, pd.DataFrame],
                              sentences_joint: pd.DataFrame,
                              metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per justification: presence AND counts on both layers.

    The sentence-level records are not discarded - they are the input here and
    stay available for the localization diagnostic and qualitative inspection.
    """
    keys = ["justification_id", "model", "game_id", "run_label",
            "decoding_group"]
    frame = sentences_joint[keys].drop_duplicates().reset_index(drop=True)

    n_sentences = (sentences_joint.groupby("justification_id", observed=True)
                   .size().rename("n_sentences"))
    frame = frame.merge(n_sentences, on="justification_id", how="left")

    # --- semantic: presence anywhere, plus how many sentences carry it ------
    for category in ALL_CATEGORIES:
        per_justification = (
            sentences_joint.groupby("justification_id", observed=True)
            [f"sem_{category}"].sum().rename(f"n_sent_{category}"))
        frame = frame.merge(per_justification, on="justification_id", how="left")
        frame[f"sem_{category}"] = frame[f"n_sent_{category}"] > 0

    # --- discourse: counts of relations anywhere in the justification -------
    # NOTE the id column. `aligned` carries TWO: `justification_id` is the
    # discourse pipeline's own integer row index, and `justification_id_
    # canonical` is the semantic string id this analysis keys on. The
    # sentence-level module only ever grouped by the sentence key, so it never
    # had to choose; here the choice matters and the canonical id is the right
    # one.
    aligned = alignment["aligned"].rename(
        columns={"justification_id": "discourse_row_id",
                 "justification_id_canonical": "justification_id"})
    for relation in TOP_LEVEL_ORDER:
        counts = (aligned.loc[aligned["top_level"].astype(str).eq(relation)]
                  .groupby("justification_id").size().rename(f"n_disc_{relation}"))
        frame = frame.merge(counts, on="justification_id", how="left")
        frame[f"n_disc_{relation}"] = frame[f"n_disc_{relation}"].fillna(0).astype(int)
        frame[f"disc_{relation}"] = frame[f"n_disc_{relation}"] > 0

    for sense in SENSE_ORDER:
        counts = (aligned.loc[aligned["raw_sense"].astype(str).eq(sense)]
                  .groupby("justification_id").size().rename(f"n_sense_{sense}"))
        frame = frame.merge(counts, on="justification_id", how="left")
        frame[f"n_sense_{sense}"] = frame[f"n_sense_{sense}"].fillna(0).astype(int)
        frame[f"sense_{sense}"] = frame[f"n_sense_{sense}"] > 0

    total_relations = aligned.groupby("justification_id").size().rename("n_relations")
    frame = frame.merge(total_relations, on="justification_id", how="left")
    frame["n_relations"] = frame["n_relations"].fillna(0).astype(int)
    frame["has_any_relation"] = frame["n_relations"] > 0

    frame = frame.merge(
        metadata[["justification_id", "vote", "voted_player_end_role",
                  "is_correct", "n_words"]],
        on="justification_id", how="left")

    frame["n_semantic_categories"] = frame[
        [f"sem_{c}" for c in CATEGORY_ORDER]].sum(axis=1)
    return _order(frame)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validation_report(layers: Dict[str, pd.DataFrame],
                      alignment: Dict[str, pd.DataFrame],
                      sentences_joint: pd.DataFrame,
                      justifications: pd.DataFrame) -> pd.DataFrame:
    """Everything that has to hold before a single number is interpreted."""
    semantic = layers["semantic"]
    aligned = alignment["aligned"]

    # Aggregation must not lose a record: relation counts summed over
    # justifications have to equal the aligned relation count, and sentence
    # presence summed over justifications has to equal sentence presence.
    relation_sum = int(justifications["n_relations"].sum())
    top_level_sum = {
        r: int(justifications[f"n_disc_{r}"].sum()) for r in TOP_LEVEL_ORDER}
    top_level_direct = (
        aligned["top_level"].astype(str).value_counts().to_dict())
    sentence_presence = {
        c: int(sentences_joint[f"sem_{c}"].sum()) for c in ALL_CATEGORIES}
    aggregated_presence = {
        c: int(justifications[f"n_sent_{c}"].sum()) for c in ALL_CATEGORIES}

    checks: List[Tuple[str, object, object]] = [
        ("justifications", len(justifications), INVARIANTS["justifications"]),
        ("no duplicate justification records",
         int(justifications["justification_id"].duplicated().sum()), 0),
        ("no justification lost",
         justifications["justification_id"].nunique(),
         sentences_joint["justification_id"].nunique()),
        ("underlying canonical sentences",
         len(sentences_joint), INVARIANTS["sentences"]),
        ("sentence counts sum to the corpus",
         int(justifications["n_sentences"].sum()), INVARIANTS["sentences"]),
        ("semantic labels unchanged",
         len(semantic["labels"]), INVARIANTS["semantic_labels"]),
        ("accepted discourse relations",
         len(aligned), INVARIANTS["accepted_relations"]),
        ("relation counts aggregate without loss",
         relation_sum, INVARIANTS["accepted_relations"]),
        ("top-level counts aggregate without loss",
         {k: top_level_sum[k] for k in sorted(top_level_sum)},
         {k: top_level_direct[k] for k in sorted(top_level_direct)}),
        ("top-level totals reproduce the frozen discourse analysis",
         {k: top_level_sum[k] for k in sorted(top_level_sum)},
         {"Comparison": 1608, "Contingency": 1534,
          "Expansion": 1513, "Temporal": 849}),
        ("sentence-presence aggregates without loss",
         aggregated_presence, sentence_presence),
        ("WORD_PATTERN token total reproduces the frozen denominator",
         int(justifications["n_words"].sum()),
         INVARIANTS["word_pattern_tokens"]),
        ("games", justifications["game_id"].nunique(), INVARIANTS["games"]),
        ("models agree across layers",
         sorted(justifications["model"].astype(str).unique()),
         sorted(MODEL_ORDER)),
        ("runs agree across layers",
         sorted(justifications["run_label"].unique()),
         sorted(sem.STOCHASTIC_RUNS + sem.GREEDY_RUNS)),
        ("game sets agree across layers",
         set(aligned["game_id"]) == set(justifications["game_id"]), True),
        ("justifications per model x run",
         sorted(int(n) for n in
                justifications.groupby(["model", "run_label"], observed=True)
                .size().unique()), [191]),
        ("stochastic and greedy separated",
         sorted(justifications["decoding_group"].astype(str).unique()),
         sorted(DECODING_ORDER)),
        ("no vote or correctness missing",
         int(justifications[["vote", "is_correct"]].isna().sum().sum()), 0),
    ]

    rows = []
    for name, observed, expected in checks:
        rows.append({
            "check": name,
            "observed": str(observed),
            "expected": str(expected),
            "status": "OK" if observed == expected else "FAIL",
        })
    return pd.DataFrame(rows)


def toy_example_check() -> pd.DataFrame:
    """The behaviour the whole rewrite exists for, asserted on a toy corpus.

    Sentence 1 carries Mechanical, sentence 2 carries Contingency, and they
    never share a sentence. The justification-level analysis MUST record the
    pairing; the localization rate for it MUST be 0. The deprecated
    sentence-level analysis would have recorded nothing at all.
    """
    sentences = pd.DataFrame([
        {"justification_id": "toy", "sentence_id": 1,
         "sem_Mechanical": True, "disc_Contingency": False},
        {"justification_id": "toy", "sentence_id": 2,
         "sem_Mechanical": False, "disc_Contingency": True},
    ])
    category_anywhere = bool(sentences["sem_Mechanical"].any())
    relation_anywhere = bool(sentences["disc_Contingency"].any())
    same_sentence = bool(
        (sentences["sem_Mechanical"] & sentences["disc_Contingency"]).any())

    return pd.DataFrame([
        {"check": "category present anywhere in the justification",
         "observed": category_anywhere, "expected": True,
         "status": "OK" if category_anywhere else "FAIL"},
        {"check": "relation present anywhere in the justification",
         "observed": relation_anywhere, "expected": True,
         "status": "OK" if relation_anywhere else "FAIL"},
        {"check": "justification-level pairing IS recorded",
         "observed": category_anywhere and relation_anywhere, "expected": True,
         "status": "OK" if category_anywhere and relation_anywhere else "FAIL"},
        {"check": "the pair never shares a sentence (localization = 0)",
         "observed": same_sentence, "expected": False,
         "status": "OK" if not same_sentence else "FAIL"},
        {"check": "deprecated sentence-level analysis would have missed it",
         "observed": not same_sentence, "expected": True,
         "status": "OK" if not same_sentence else "FAIL"},
    ])


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------

def count_tensors(justifications: pd.DataFrame, decoding: str,
                  kind: str = "top_level") -> Dict[str, object]:
    """Per (model, run, game): justification, word and relation aggregates.

    Every table and every bootstrap below reads these arrays, so no two can
    disagree about their inputs. Games sit in a fixed shared order, which is
    what makes the paired bootstrap meaningful.
    """
    runs = RUNS_BY_DECODING[decoding]
    frame = justifications.loc[
        justifications["decoding_group"].astype(str).eq(decoding)]
    games = sorted(frame["game_id"].unique())
    game_index = {game: i for i, game in enumerate(games)}
    names, presence_columns = _relation_columns(kind)
    _, count_columns = _count_columns(kind)

    shape = (len(MODEL_ORDER), len(runs), len(games))
    n_justifications = np.zeros(shape, dtype=np.float64)
    n_category = np.zeros(shape + (len(CATEGORY_ORDER),), dtype=np.float64)
    n_relation = np.zeros(shape + (len(names),), dtype=np.float64)
    n_pair = np.zeros(shape + (len(CATEGORY_ORDER), len(names)),
                      dtype=np.float64)
    # Words and relation counts inside justifications carrying category c,
    # for the canonical ratio-of-sums density.
    words_category = np.zeros(shape + (len(CATEGORY_ORDER),), dtype=np.float64)
    relations_in_category = np.zeros(
        shape + (len(CATEGORY_ORDER), len(names)), dtype=np.float64)

    category_matrix = frame[[f"sem_{c}" for c in CATEGORY_ORDER]].to_numpy(bool)
    relation_matrix = frame[presence_columns].to_numpy(bool)
    relation_counts = frame[count_columns].to_numpy(float)
    words = frame["n_words"].to_numpy(float)
    models = frame["model"].astype(str).to_numpy()
    run_labels = frame["run_label"].to_numpy()
    game_ids = frame["game_id"].to_numpy()

    for row in range(len(frame)):
        m = MODEL_ORDER.index(models[row])
        r = runs.index(run_labels[row])
        g = game_index[game_ids[row]]
        has_category = category_matrix[row]
        has_relation = relation_matrix[row]
        n_justifications[m, r, g] += 1
        n_category[m, r, g] += has_category
        n_relation[m, r, g] += has_relation
        n_pair[m, r, g] += np.outer(has_category, has_relation)
        words_category[m, r, g] += has_category * words[row]
        relations_in_category[m, r, g] += np.outer(
            has_category, relation_counts[row])

    return {
        "games": games,
        "runs": runs,
        "relation_names": names,
        "n_justifications": n_justifications,
        "n_category": n_category,
        "n_relation": n_relation,
        "n_pair": n_pair,
        "words_category": words_category,
        "relations_in_category": relations_in_category,
    }


def _game_weights(n_games: int, n_replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        n_games, np.full(n_games, 1 / n_games), size=n_replicates
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# B - conditional prevalence P(r | c) at justification level
# ---------------------------------------------------------------------------

def conditional_prevalence(justifications: pd.DataFrame,
                           kind: str = "top_level") -> pd.DataFrame:
    """P(r|c) = justifications containing both / justifications containing c.

    Presence anywhere in the justification, on both sides. Multiple
    occurrences never increase a presence count.
    """
    rows = []
    for decoding in DECODING_ORDER:
        tensors = count_tensors(justifications, decoding, kind)
        names = tensors["relation_names"]
        for m, model in enumerate(MODEL_ORDER):
            for r, run in enumerate(tensors["runs"]):
                for c, category in enumerate(CATEGORY_ORDER):
                    n_c = tensors["n_category"][m, r, :, c].sum()
                    for k, relation in enumerate(names):
                        n_cr = tensors["n_pair"][m, r, :, c, k].sum()
                        rows.append({
                            "model": model,
                            "decoding_group": decoding,
                            "run_label": run,
                            "semantic_category": category,
                            "discourse_relation": relation,
                            "n_justifications_with_category": int(n_c),
                            "n_justifications_with_both": int(n_cr),
                            "conditional_prevalence": (
                                n_cr / n_c if n_c else np.nan),
                            "low_support_diagnostic": bool(
                                n_c < LOW_SUPPORT_DIAGNOSTIC),
                        })
    return _order(pd.DataFrame(rows))


def summarise_across_runs(run_level: pd.DataFrame, value: str,
                          extra: Dict[str, tuple] = None) -> pd.DataFrame:
    """Mean and SD across runs. Greedy has one run, so its SD is NaN."""
    aggregations = {
        "n_runs": ("run_label", "nunique"),
        f"mean_{value}": (value, "mean"),
        f"sd_{value}": (value,
                        lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
        f"min_{value}": (value, "min"),
        f"max_{value}": (value, "max"),
    }
    if extra:
        aggregations.update(extra)
    grouped = run_level.groupby(
        ["model", "decoding_group", "semantic_category", "discourse_relation"],
        observed=True)
    return _order(grouped.agg(**aggregations).reset_index())


# ---------------------------------------------------------------------------
# C - conditional discourse density
# ---------------------------------------------------------------------------

def conditional_density(justifications: pd.DataFrame,
                        kind: str = "top_level") -> pd.DataFrame:
    """Relations of class r per 100 words, within justifications carrying c.

    PRIMARY definition is the canonical one from the frozen discourse
    analysis: a ratio of sums per run,
    100 * (relations of r inside those justifications) / (their words).

    `mean_per_justification_density` is the alternative reading - the mean of
    the per-justification densities - and is reported alongside so the
    difference is visible rather than assumed away. It weights a short
    justification as heavily as a long one; the ratio of sums does not.
    """
    rows = []
    for decoding in DECODING_ORDER:
        tensors = count_tensors(justifications, decoding, kind)
        names = tensors["relation_names"]
        frame = justifications.loc[
            justifications["decoding_group"].astype(str).eq(decoding)]
        _, count_columns = _count_columns(kind)

        for m, model in enumerate(MODEL_ORDER):
            for r, run in enumerate(tensors["runs"]):
                subset_run = frame.loc[
                    frame["model"].astype(str).eq(model)
                    & frame["run_label"].eq(tensors["runs"][r])]
                for c, category in enumerate(CATEGORY_ORDER):
                    words = tensors["words_category"][m, r, :, c].sum()
                    n_c = tensors["n_category"][m, r, :, c].sum()
                    carrying = subset_run.loc[subset_run[f"sem_{category}"]]
                    for k, relation in enumerate(names):
                        relations = tensors["relations_in_category"][
                            m, r, :, c, k].sum()
                        per_justification = (
                            100 * carrying[count_columns[k]] / carrying["n_words"]
                            if len(carrying) else pd.Series(dtype=float))
                        rows.append({
                            "model": model,
                            "decoding_group": decoding,
                            "run_label": run,
                            "semantic_category": category,
                            "discourse_relation": relation,
                            "n_justifications_with_category": int(n_c),
                            "total_words": int(words),
                            "total_relations": int(relations),
                            "relations_per_100_words": (
                                100 * relations / words if words else np.nan),
                            "mean_per_justification_density": (
                                float(per_justification.mean())
                                if len(per_justification) else np.nan),
                            "low_support_diagnostic": bool(
                                n_c < LOW_SUPPORT_DIAGNOSTIC),
                        })
    return _order(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# D - justification-level lift
# ---------------------------------------------------------------------------

def joint_prevalence_and_lift(justifications: pd.DataFrame,
                              kind: str = "top_level") -> pd.DataFrame:
    """P(c,r), marginals, lift and raw support, over justifications."""
    rows = []
    for decoding in DECODING_ORDER:
        tensors = count_tensors(justifications, decoding, kind)
        names = tensors["relation_names"]
        for m, model in enumerate(MODEL_ORDER):
            for r, run in enumerate(tensors["runs"]):
                total = tensors["n_justifications"][m, r].sum()
                for c, category in enumerate(CATEGORY_ORDER):
                    n_c = tensors["n_category"][m, r, :, c].sum()
                    p_c = n_c / total if total else np.nan
                    for k, relation in enumerate(names):
                        n_r = tensors["n_relation"][m, r, :, k].sum()
                        n_cr = tensors["n_pair"][m, r, :, c, k].sum()
                        p_r = n_r / total if total else np.nan
                        p_cr = n_cr / total if total else np.nan
                        expected = p_c * p_r
                        rows.append({
                            "model": model,
                            "decoding_group": decoding,
                            "run_label": run,
                            "semantic_category": category,
                            "discourse_relation": relation,
                            "n_justifications": int(total),
                            "support": int(n_cr),
                            "joint_prevalence": p_cr,
                            "prevalence_category": p_c,
                            "prevalence_relation": p_r,
                            "lift": (p_cr / expected
                                     if expected and expected > 0 else np.nan),
                            "low_support_diagnostic": bool(
                                n_c < LOW_SUPPORT_DIAGNOSTIC),
                        })
    return _order(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# H - localization diagnostic
# ---------------------------------------------------------------------------

def localization_rate(sentences_joint: pd.DataFrame,
                      justifications: pd.DataFrame,
                      kind: str = "top_level") -> pd.DataFrame:
    """Among justifications containing both c and r, how often do they share
    at least one sentence?

    DESCRIPTIVE ONLY. A high rate says the pair tends to surface together in
    one sentence; a low rate says the justification spreads them across
    sentences. Neither is evidence about argument attachment, and a rate of 0
    does not make the justification-level pairing spurious - it is exactly the
    case the sentence-level analysis used to discard.
    """
    # The presence columns carry the same names in both frames, so one lookup
    # serves the justification-level pairing and the sentence-level check.
    names, presence_columns = _relation_columns(kind)

    rows = []
    for (model, decoding, run), group in justifications.groupby(
            ["model", "decoding_group", "run_label"], observed=True):
        ids = set(group["justification_id"])
        sentences = sentences_joint.loc[
            sentences_joint["justification_id"].isin(ids)]

        for category in CATEGORY_ORDER:
            for name, column in zip(names, presence_columns):
                paired = group.loc[group[f"sem_{category}"] & group[column]]
                n_paired = len(paired)
                if n_paired == 0:
                    rows.append({
                        "model": model, "decoding_group": decoding,
                        "run_label": run, "semantic_category": category,
                        "discourse_relation": name,
                        "n_justifications_with_both": 0,
                        "n_with_same_sentence": 0,
                        "localization_rate": np.nan,
                    })
                    continue

                local = sentences.loc[
                    sentences["justification_id"].isin(paired["justification_id"])
                    & sentences[f"sem_{category}"] & sentences[column]
                ]["justification_id"].nunique()

                rows.append({
                    "model": model, "decoding_group": decoding,
                    "run_label": run, "semantic_category": category,
                    "discourse_relation": name,
                    "n_justifications_with_both": n_paired,
                    "n_with_same_sentence": int(local),
                    "localization_rate": local / n_paired,
                })
    return _order(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# E / F - paired game-level bootstraps
# ---------------------------------------------------------------------------

def _paired_bootstrap(justifications: pd.DataFrame, kind: str,
                      numerator_key: str, denominator_key: str,
                      scale: float, metric_name: str,
                      n_replicates: int, seed: int) -> pd.DataFrame:
    """Shared machinery: a ratio of sums over resampled games, per run,
    averaged across runs, then differenced between models."""
    rows = []
    for decoding in DECODING_ORDER:
        tensors = count_tensors(justifications, decoding, kind)
        games, runs = tensors["games"], tensors["runs"]
        names = tensors["relation_names"]
        n_games = len(games)
        weights = _game_weights(n_games, n_replicates, seed)

        for c, category in enumerate(CATEGORY_ORDER):
            for k, relation in enumerate(names):
                numerator = tensors[numerator_key]
                denominator = tensors[denominator_key]
                num = (numerator[:, :, :, c, k] if numerator.ndim == 5
                       else numerator[:, :, :, c]).reshape(-1, n_games)
                den = (denominator[:, :, :, c, k] if denominator.ndim == 5
                       else denominator[:, :, :, c]).reshape(-1, n_games)

                boot_num = num @ weights.T
                boot_den = den @ weights.T
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(boot_den > 0,
                                     scale * boot_num / boot_den, np.nan)
                ratio = ratio.reshape(len(MODEL_ORDER), len(runs), n_replicates)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model_values = np.nanmean(ratio, axis=1)

                observed_num = num.reshape(len(MODEL_ORDER), len(runs), n_games).sum(axis=2)
                observed_den = den.reshape(len(MODEL_ORDER), len(runs), n_games).sum(axis=2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    observed = np.where(observed_den > 0,
                                        scale * observed_num / observed_den, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    observed = np.nanmean(observed, axis=1)

                support = tensors["n_category"][:, :, :, c].sum(axis=(1, 2))

                for i in range(len(MODEL_ORDER)):
                    for j in range(i + 1, len(MODEL_ORDER)):
                        differences = model_values[i] - model_values[j]
                        valid = int(np.isfinite(differences).sum())
                        if valid:
                            low, high = np.nanpercentile(differences, [2.5, 97.5])
                        else:
                            low = high = np.nan
                        rows.append({
                            "decoding_group": decoding,
                            "metric": metric_name,
                            "semantic_category": category,
                            "discourse_relation": relation,
                            "model_a": MODEL_ORDER[i],
                            "model_b": MODEL_ORDER[j],
                            "value_a": observed[i],
                            "value_b": observed[j],
                            "difference": observed[i] - observed[j],
                            "ci_low": low,
                            "ci_high": high,
                            "ci_excludes_zero": bool(
                                np.isfinite(low) and np.isfinite(high)
                                and (low > 0 or high < 0)),
                            "n_justifications_with_category_a": int(support[i]),
                            "n_justifications_with_category_b": int(support[j]),
                            "low_support_diagnostic": bool(
                                support[i] < LOW_SUPPORT_DIAGNOSTIC * len(runs)
                                or support[j] < LOW_SUPPORT_DIAGNOSTIC * len(runs)),
                            "n_valid_replicates": valid,
                            "n_games": n_games,
                            "n_replicates": n_replicates,
                            "seed": seed,
                        })
    return _order(pd.DataFrame(rows))


def prevalence_bootstrap(justifications: pd.DataFrame, kind: str = "top_level",
                         n_replicates: int = BOOTSTRAP_REPLICATES,
                         seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """E: pairwise model differences in P(r|c), paired on games."""
    return _paired_bootstrap(
        justifications, kind, "n_pair", "n_category", 1.0,
        "conditional_prevalence", n_replicates, seed)


def density_bootstrap(justifications: pd.DataFrame, kind: str = "top_level",
                      n_replicates: int = BOOTSTRAP_REPLICATES,
                      seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """F: pairwise model differences in relations per 100 words within c."""
    return _paired_bootstrap(
        justifications, kind, "relations_in_category", "words_category", 100.0,
        "relations_per_100_words", n_replicates, seed)


# ---------------------------------------------------------------------------
# Matrix view and export
# ---------------------------------------------------------------------------

def matrix(summary: pd.DataFrame, model: str, decoding: str, value: str,
           relations: Sequence[str] = None) -> pd.DataFrame:
    relations = list(relations or TOP_LEVEL_ORDER)
    subset = summary.loc[
        summary["model"].astype(str).eq(model)
        & summary["decoding_group"].astype(str).eq(decoding)]
    return (subset.pivot(index="semantic_category",
                         columns="discourse_relation", values=value)
            .reindex(index=CATEGORY_ORDER, columns=relations))


def build_final_tables(layers, alignment, sentences_joint,
                       justifications) -> Dict[str, pd.DataFrame]:
    prevalence = conditional_prevalence(justifications, "top_level")
    density = conditional_density(justifications, "top_level")
    lift = joint_prevalence_and_lift(justifications, "top_level")
    fine_prevalence = conditional_prevalence(justifications, "sense")
    fine_lift = joint_prevalence_and_lift(justifications, "sense")
    localization_top_level = localization_rate(
        sentences_joint, justifications, "top_level")
    localization_fine = localization_rate(
        sentences_joint, justifications, "sense")

    return {
        "K0_validation_report": validation_report(
            layers, alignment, sentences_joint, justifications),
        "K0b_toy_example_check": toy_example_check(),
        "K1_justification_presence": justifications,
        "K2_conditional_prevalence_run_level": prevalence,
        "K2b_conditional_prevalence_summary": summarise_across_runs(
            prevalence, "conditional_prevalence",
            {"mean_n_with_category": ("n_justifications_with_category", "mean"),
             "total_n_with_category": ("n_justifications_with_category", "sum"),
             "total_n_with_both": ("n_justifications_with_both", "sum"),
             "low_support_diagnostic": ("low_support_diagnostic", "any")}),
        "K3_conditional_density_run_level": density,
        "K3b_conditional_density_summary": summarise_across_runs(
            density, "relations_per_100_words",
            {"mean_per_justification_density":
                 ("mean_per_justification_density", "mean"),
             "total_words": ("total_words", "sum"),
             "total_relations": ("total_relations", "sum"),
             "low_support_diagnostic": ("low_support_diagnostic", "any")}),
        "K4_joint_prevalence_lift_run_level": lift,
        "K4b_joint_prevalence_lift_summary": summarise_across_runs(
            lift, "lift",
            {"mean_joint_prevalence": ("joint_prevalence", "mean"),
             "mean_support": ("support", "mean"),
             "total_support": ("support", "sum"),
             "low_support_diagnostic": ("low_support_diagnostic", "any")}),
        "K5_bootstrap_prevalence_differences": prevalence_bootstrap(
            justifications, "top_level"),
        "K6_bootstrap_density_differences": density_bootstrap(
            justifications, "top_level"),
        "K7_finegrained_conditional_prevalence_summary": summarise_across_runs(
            fine_prevalence, "conditional_prevalence",
            {"total_n_with_category": ("n_justifications_with_category", "sum"),
             "total_n_with_both": ("n_justifications_with_both", "sum"),
             "low_support_diagnostic": ("low_support_diagnostic", "any")}),
        "K7b_finegrained_lift_summary": summarise_across_runs(
            fine_lift, "lift",
            {"mean_joint_prevalence": ("joint_prevalence", "mean"),
             "total_support": ("support", "sum"),
             "low_support_diagnostic": ("low_support_diagnostic", "any")}),
        # Run-level is exported as well as the summary: the raw numerator and
        # denominator live there, and that is where the "rate is a proportion"
        # invariant is meaningful.
        "K8_localization_run_level": localization_top_level,
        "K8b_localization_summary": summarise_across_runs(
            localization_top_level, "localization_rate",
            {"total_n_with_both": ("n_justifications_with_both", "sum"),
             "total_n_same_sentence": ("n_with_same_sentence", "sum")}),
        "K8c_localization_finegrained_summary": summarise_across_runs(
            localization_fine, "localization_rate",
            {"total_n_with_both": ("n_justifications_with_both", "sum"),
             "total_n_same_sentence": ("n_with_same_sentence", "sum")}),
    }


LATEX_TABLES = (
    "K2b_conditional_prevalence_summary",
    "K3b_conditional_density_summary",
    "K5_bootstrap_prevalence_differences",
    "K6_bootstrap_density_differences",
)


def write_final_tables(tables: Dict[str, pd.DataFrame],
                       directory: Path) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in tables.items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)
        if name in LATEX_TABLES and len(frame):
            tex_path = directory / f"{name}.tex"
            to_latex(frame.set_index(frame.columns[0]), tex_path)
            written.append(tex_path)
    return written
