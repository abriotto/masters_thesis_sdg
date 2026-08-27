"""Joint discourse x semantic analysis (RQ2), over two frozen layers.

Two inputs, both finished and neither touched here:

  * the native-discopy explicit-relation table, filtered to
    `is_connective == True` (5,504 accepted relations);
  * the frozen DeepSeek semantic annotations (8,044 sentences, 11,526 labels).

The question is narrow and deliberately so:

    Which explicit discourse relations tend to occur in sentences containing
    each semantic category?

WHAT THIS IS NOT. The parser identifies explicit connectives and their PDTB
sense. It does NOT recover discourse arguments, and this analysis never asks
it to. So a co-occurrence here is a SENTENCE-LEVEL ASSOCIATION between the
form of a sentence and the kind of information it carries - never evidence
that a connective attaches to a semantic proposition, that a relation "is
used to express" a category, or that a category is an argument of a relation.
Every name in this module is chosen to keep that distinction visible.

The unit is the CANONICAL SENTENCE, defined by the semantic layer's fixed
segmentation (`src/utils/sentences.split_sentences`, shared by both
pipelines). Presence, not counts: a sentence with two Contingency relations
counts once for any (category, Contingency) pair.

Alignment is exact, not fuzzy. Both layers carry
(model, game_id, run_label, sentence_id) and the discourse table carries the
sentence text it scored, so every accepted relation is matched to its
canonical sentence by key and then VERIFIED by byte-exact text comparison.
Anything that fails is reported, never silently matched.

Aggregation follows the two frozen strands exactly:

  * each run computed independently, then mean and SD across the three
    stochastic runs;
  * greedy is one deterministic run, always separate, SD undefined (NaN);
  * the bootstrap resamples GAMES with all their runs and sentences attached,
    reusing the same resampled game ids across models.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.justification_analysis.comparison.discourse_statistics import to_latex
from src.justification_analysis.semantic import semantic_final as sem

# ---------------------------------------------------------------------------
# Vocabulary and design constants
# ---------------------------------------------------------------------------

CATEGORY_ORDER: List[str] = list(sem.CATEGORY_ORDER)          # 7 substantive
OTHER_CATEGORY = sem.OTHER_CATEGORY
ALL_CATEGORIES: List[str] = list(sem.ALL_CATEGORIES)

TOP_LEVEL_ORDER = ["Comparison", "Contingency", "Expansion", "Temporal"]
ANY_RELATION = "Any explicit"

# The nine senses the parser actually produced on this corpus, grouped by
# top-level class. Taken from the frozen discourse handoff, and asserted
# against the data rather than trusted.
SENSE_ORDER = [
    "Comparison.Contrast", "Comparison.Concession",
    "Contingency.Cause", "Contingency.Condition",
    "Expansion.Conjunction", "Expansion.Alternative", "Expansion.Restatement",
    "Temporal.Asynchronous", "Temporal.Synchrony",
]

MODEL_ORDER = list(sem.MODEL_ORDER)
DECODING_ORDER = list(sem.DECODING_ORDER)
RUNS_BY_DECODING = dict(sem.RUNS_BY_DECODING)
RUN_KEYS = ["model", "decoding_group", "run_label"]
SENTENCE_KEY = ["model", "game_id", "run_label", "sentence_id"]

DISCOURSE_SUBPATH = Path(
    "analysis/cross_model/base/voting/prompt_v4/justification_analysis"
    "/discourse_parser/discopy_explicit_candidates.csv"
)

ARTIFACT_SUBPATH = Path(
    "analysis/cross_model/base/voting/prompt_v4/justification_analysis"
    "/joint_discourse_semantic"
)
FINAL_TABLES_SUBPATH = ARTIFACT_SUBPATH / "thesis_tables" / "final_joint"
FINAL_FIGURES_SUBPATH = ARTIFACT_SUBPATH / "figures" / "final_joint"

BOOTSTRAP_SEED = sem.BOOTSTRAP_SEED                # 20260826
BOOTSTRAP_REPLICATES = sem.BOOTSTRAP_REPLICATES    # 10,000

# A per-run denominator below this makes a conditional prevalence a statement
# about a handful of sentences. Cells are FLAGGED, never dropped - see
# `conditional_prevalence`.
THIN_SUPPORT = 30

# Frozen on both sides. A mismatch means one of the two layers changed and
# every number here is stale.
INVARIANTS = {
    "justifications": 2292,
    "games": 191,
    "sentences": 8044,
    "semantic_labels": 11526,
    "accepted_relations": 5504,
    "candidates": 14209,
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


# ---------------------------------------------------------------------------
# Input and alignment
# ---------------------------------------------------------------------------

def load_layers(repo_root: Path) -> Dict[str, pd.DataFrame]:
    """The semantic frames and the ACCEPTED discourse relations.

    Only `is_connective == True` rows are returned as relations: the rejected
    NoSense candidates stay in the source table so the accept/reject behaviour
    is inspectable, but they are not relations and never enter this analysis.
    """
    repo_root = Path(repo_root)
    semantic = sem.load_annotations(repo_root)

    candidates = pd.read_csv(repo_root / DISCOURSE_SUBPATH)
    relations = candidates.loc[candidates["is_connective"]].copy()
    relations["model"] = relations["model"].astype(str)

    return {
        "semantic": semantic,
        "candidates": candidates,
        "relations": relations,
    }


def align_relations(layers: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Attach every accepted relation to exactly one canonical sentence.

    The join is on (model, game_id, run_label, sentence_id) - identifiers both
    pipelines already carry - and is then VERIFIED by comparing the sentence
    text the parser scored against the canonical sentence text, byte for byte.
    No fuzzy matching, no normalisation, no fallback: a relation that fails
    either step is returned in `unaligned` for explicit reporting.
    """
    semantic = layers["semantic"]
    relations = layers["relations"]

    sentences = semantic["sentences"].merge(
        semantic["justifications"][
            ["justification_id", "model", "game_id", "run_label",
             "decoding_group"]],
        on="justification_id", how="left",
    )
    sentences["model"] = sentences["model"].astype(str)
    sentences["decoding_group"] = sentences["decoding_group"].astype(str)

    merged = relations.merge(
        sentences[SENTENCE_KEY + ["justification_id", "text"]],
        on=SENTENCE_KEY, how="left", indicator=True,
        suffixes=("", "_canonical"),
    )

    no_sentence = merged.loc[merged["_merge"] != "both"].copy()
    no_sentence["alignment_failure"] = "no canonical sentence for this key"

    matched = merged.loc[merged["_merge"] == "both"].copy()
    text_differs = matched.loc[
        matched["sentence_text"].astype(str) != matched["text"].astype(str)
    ].copy()
    text_differs["alignment_failure"] = "sentence text does not match canonical"

    aligned = matched.loc[
        matched["sentence_text"].astype(str) == matched["text"].astype(str)
    ].drop(columns=["_merge"]).copy()

    unaligned = pd.concat([no_sentence, text_differs], ignore_index=True)
    if len(unaligned):
        unaligned = unaligned[[
            "occurrence_id", "model", "game_id", "run_label", "sentence_id",
            "connective_surface", "raw_sense", "alignment_failure",
        ]]

    return {"sentences": sentences, "aligned": aligned, "unaligned": unaligned}


def build_joint_sentences(layers: Dict[str, pd.DataFrame],
                          alignment: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per canonical sentence, with both layers as presence flags.

    Multi-label structure is preserved on both sides: a sentence may carry
    several semantic categories and several discourse relations, and each is
    an independent boolean. Sentences with no relation and/or no semantic
    label are kept - dropping them would silently change every denominator.
    """
    semantic = layers["semantic"]
    sentences = alignment["sentences"]
    aligned = alignment["aligned"]

    joint = sentences[
        ["justification_id", "sentence_id", "model", "game_id", "run_label",
         "decoding_group", "text", "n_labels"]
    ].copy()

    # --- semantic presence -------------------------------------------------
    labels = semantic["labels"]
    for category in ALL_CATEGORIES:
        present = set(
            map(tuple,
                labels.loc[labels["category"].astype(str).eq(category),
                           ["justification_id", "sentence_id"]].to_numpy())
        )
        joint[f"sem_{category}"] = [
            (j, s) in present
            for j, s in zip(joint["justification_id"], joint["sentence_id"])
        ]

    # --- discourse presence ------------------------------------------------
    keys = list(map(tuple, joint[SENTENCE_KEY].to_numpy()))

    for relation in TOP_LEVEL_ORDER:
        present = set(map(tuple, aligned.loc[
            aligned["top_level"].astype(str).eq(relation), SENTENCE_KEY
        ].to_numpy()))
        joint[f"disc_{relation}"] = [key in present for key in keys]

    for sense in SENSE_ORDER:
        present = set(map(tuple, aligned.loc[
            aligned["raw_sense"].astype(str).eq(sense), SENTENCE_KEY
        ].to_numpy()))
        joint[f"sense_{sense}"] = [key in present for key in keys]

    counts = (aligned.groupby(SENTENCE_KEY, observed=True).size()
              .rename("n_relations"))
    joint = joint.merge(counts, on=SENTENCE_KEY, how="left")
    joint["n_relations"] = joint["n_relations"].fillna(0).astype(int)
    joint["has_any_relation"] = joint["n_relations"] > 0

    joint["n_semantic_categories"] = joint[
        [f"sem_{c}" for c in CATEGORY_ORDER]].sum(axis=1)

    return _order(joint)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validation_report(layers: Dict[str, pd.DataFrame],
                      alignment: Dict[str, pd.DataFrame],
                      joint: pd.DataFrame) -> pd.DataFrame:
    """Every invariant the joint dataset has to satisfy, recomputed.

    Both source analyses are frozen, so anything that disagrees with them is a
    bug in THIS module, not a new finding. The caller is expected to assert on
    the result rather than read past a FAIL.
    """
    semantic = layers["semantic"]
    relations = layers["relations"]
    candidates = layers["candidates"]
    aligned = alignment["aligned"]
    unaligned = alignment["unaligned"]

    semantic_totals = {
        category: int(
            semantic["labels"]["category"].astype(str).eq(category).sum())
        for category in ALL_CATEGORIES
    }
    joint_totals = {
        category: int(joint[f"sem_{category}"].sum())
        for category in ALL_CATEGORIES
    }
    # Label count >= sentence-presence count: a sentence can carry the same
    # category twice only if the annotator emitted two assignments for it.
    presence_le_labels = all(
        joint_totals[c] <= semantic_totals[c] for c in ALL_CATEGORIES)

    discourse_top_level = (
        relations["top_level"].astype(str).value_counts().to_dict())
    aligned_top_level = (
        aligned["top_level"].astype(str).value_counts().to_dict())

    checks: List[Tuple[str, object, object]] = [
        ("justifications represented",
         joint["justification_id"].nunique(), INVARIANTS["justifications"]),
        ("canonical sentences", len(joint), INVARIANTS["sentences"]),
        ("games", joint["game_id"].nunique(), INVARIANTS["games"]),
        ("discourse candidates in source table",
         len(candidates), INVARIANTS["candidates"]),
        ("accepted discourse relations",
         len(relations), INVARIANTS["accepted_relations"]),
        ("accepted relations aligned to a canonical sentence",
         len(aligned), INVARIANTS["accepted_relations"]),
        ("relations that could NOT be aligned", len(unaligned), 0),
        ("every relation assigned to exactly one sentence",
         int(aligned.duplicated("occurrence_id").sum()), 0),
        ("no duplicate canonical sentences",
         int(joint.duplicated(["justification_id", "sentence_id"]).sum()), 0),
        ("no duplicate sentence keys",
         int(joint.duplicated(SENTENCE_KEY).sum()), 0),
        ("semantic labels unchanged",
         len(semantic["labels"]), INVARIANTS["semantic_labels"]),
        ("sentence-presence never exceeds label count",
         presence_le_labels, True),
        ("top-level totals reproduce the frozen discourse analysis",
         {k: v for k, v in sorted(discourse_top_level.items())},
         {k: v for k, v in sorted(aligned_top_level.items())}),
        ("top-level classes are the four PDTB classes",
         sorted(aligned["top_level"].astype(str).unique()),
         sorted(TOP_LEVEL_ORDER)),
        ("observed level-2 senses",
         sorted(aligned["raw_sense"].astype(str).unique()),
         sorted(SENSE_ORDER)),
        ("no NoSense candidate entered the analysis",
         bool(aligned["is_connective"].all()), True),
        ("relation_type is Explicit throughout",
         sorted(aligned["relation_type"].astype(str).unique()), ["Explicit"]),
        ("models match across layers",
         sorted(joint["model"].astype(str).unique()), sorted(MODEL_ORDER)),
        ("runs match across layers",
         sorted(joint["run_label"].unique()),
         sorted(sem.STOCHASTIC_RUNS + sem.GREEDY_RUNS)),
        ("game sets identical across layers",
         set(relations["game_id"]) == set(joint["game_id"]), True),
        ("stochastic and greedy separated",
         sorted(joint["decoding_group"].astype(str).unique()),
         sorted(DECODING_ORDER)),
        ("sentences per model x run",
         sorted(int(n) for n in
                joint.groupby(["model", "run_label"], observed=True)
                .size().unique()),
         sorted(int(n) for n in
                alignment["sentences"]
                .groupby(["model", "run_label"], observed=True)
                .size().unique())),
        ("justifications per model x run",
         sorted(int(n) for n in
                joint.groupby(["model", "run_label"], observed=True)
                ["justification_id"].nunique().unique()),
         [191]),
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


# ---------------------------------------------------------------------------
# Tensors - the shared substrate for the descriptive tables and the bootstrap
# ---------------------------------------------------------------------------

def _relation_columns(kind: str) -> Tuple[List[str], List[str]]:
    if kind == "top_level":
        names = [*TOP_LEVEL_ORDER, ANY_RELATION]
        columns = [f"disc_{r}" for r in TOP_LEVEL_ORDER] + ["has_any_relation"]
    elif kind == "sense":
        names = list(SENSE_ORDER)
        columns = [f"sense_{s}" for s in SENSE_ORDER]
    else:
        raise ValueError(f"unknown relation kind {kind!r}")
    return names, columns


def count_tensors(joint: pd.DataFrame, decoding: str,
                  kind: str = "top_level") -> Dict[str, object]:
    """Per (model, run, game): sentences, sentences with c, sentences with c+r.

    Everything below is a different reading of these arrays, so no two tables
    can disagree about their inputs. Games are in a fixed shared order, which
    is what makes the paired bootstrap meaningful.
    """
    runs = RUNS_BY_DECODING[decoding]
    frame = joint.loc[joint["decoding_group"].astype(str).eq(decoding)]
    games = sorted(frame["game_id"].unique())
    game_index = {game: i for i, game in enumerate(games)}
    names, columns = _relation_columns(kind)

    shape = (len(MODEL_ORDER), len(runs), len(games))
    n_sentences = np.zeros(shape, dtype=np.float64)
    n_category = np.zeros(shape + (len(CATEGORY_ORDER),), dtype=np.float64)
    n_relation = np.zeros(shape + (len(names),), dtype=np.float64)
    n_joint = np.zeros(shape + (len(CATEGORY_ORDER), len(names)),
                       dtype=np.float64)

    category_columns = [f"sem_{c}" for c in CATEGORY_ORDER]
    category_matrix = frame[category_columns].to_numpy(dtype=bool)
    relation_matrix = frame[columns].to_numpy(dtype=bool)
    models = frame["model"].astype(str).to_numpy()
    run_labels = frame["run_label"].to_numpy()
    game_ids = frame["game_id"].to_numpy()

    for row in range(len(frame)):
        m = MODEL_ORDER.index(models[row])
        r = runs.index(run_labels[row])
        g = game_index[game_ids[row]]
        has_category = category_matrix[row]
        has_relation = relation_matrix[row]
        n_sentences[m, r, g] += 1
        n_category[m, r, g] += has_category
        n_relation[m, r, g] += has_relation
        n_joint[m, r, g] += np.outer(has_category, has_relation)

    return {
        "games": games,
        "runs": runs,
        "relation_names": names,
        "n_sentences": n_sentences,
        "n_category": n_category,
        "n_relation": n_relation,
        "n_joint": n_joint,
    }


# ---------------------------------------------------------------------------
# A / B / D - conditional prevalence P(r | c)
# ---------------------------------------------------------------------------

def conditional_prevalence(joint: pd.DataFrame,
                           kind: str = "top_level") -> pd.DataFrame:
    """Per (model, decoding, run, category, relation): N(c), N(c,r), P(r|c).

    P(r|c) reads: among sentences containing semantic category c, the
    proportion that ALSO contain at least one explicit relation of type r.
    Presence, not relation counts.

    `support_is_thin` marks a denominator below THIN_SUPPORT. Such cells are
    kept - deleting them would hide which model-category combinations the
    corpus simply cannot speak to - but they should not be interpreted.
    """
    rows = []
    for decoding in DECODING_ORDER:
        tensors = count_tensors(joint, decoding, kind)
        names = tensors["relation_names"]
        for m, model in enumerate(MODEL_ORDER):
            for r, run in enumerate(tensors["runs"]):
                for c, category in enumerate(CATEGORY_ORDER):
                    n_c = tensors["n_category"][m, r, :, c].sum()
                    for k, relation in enumerate(names):
                        n_cr = tensors["n_joint"][m, r, :, c, k].sum()
                        rows.append({
                            "model": model,
                            "decoding_group": decoding,
                            "run_label": run,
                            "semantic_category": category,
                            "discourse_relation": relation,
                            "n_semantic_sentences": int(n_c),
                            "n_joint_sentences": int(n_cr),
                            "conditional_prevalence": (
                                n_cr / n_c if n_c else np.nan),
                            "support_is_thin": bool(n_c < THIN_SUPPORT),
                            "thin_support_threshold": THIN_SUPPORT,
                        })
    return _order(pd.DataFrame(rows))


def conditional_prevalence_summary(run_level: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of P(r|c) across runs, per model and decoding group.

    Greedy is one run: SD is NaN by construction, never 0, so it cannot be
    mistaken for a measured spread of zero.
    """
    grouped = run_level.groupby(
        ["model", "decoding_group", "semantic_category", "discourse_relation"],
        observed=True,
    )
    summary = grouped.agg(
        n_runs=("run_label", "nunique"),
        mean_conditional_prevalence=("conditional_prevalence", "mean"),
        sd_conditional_prevalence=(
            "conditional_prevalence",
            lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
        min_conditional_prevalence=("conditional_prevalence", "min"),
        max_conditional_prevalence=("conditional_prevalence", "max"),
        mean_n_semantic_sentences=("n_semantic_sentences", "mean"),
        total_n_semantic_sentences=("n_semantic_sentences", "sum"),
        total_n_joint_sentences=("n_joint_sentences", "sum"),
        support_is_thin=("support_is_thin", "any"),
    ).reset_index()
    return _order(summary)


def any_relation_given_category(run_level: pd.DataFrame) -> pd.DataFrame:
    """D: P(at least one explicit relation | semantic category).

    Answers whether some semantic evidence types simply occur in more
    explicitly-connected sentences than others, which has to be known before
    any per-class P(r|c) is read.
    """
    subset = run_level.loc[
        run_level["discourse_relation"].astype(str).eq(ANY_RELATION)]
    return conditional_prevalence_summary(subset)


# ---------------------------------------------------------------------------
# C - joint prevalence and lift
# ---------------------------------------------------------------------------

def joint_prevalence_and_lift(joint: pd.DataFrame,
                              kind: str = "top_level") -> pd.DataFrame:
    """Sentence-level P(c,r), P(c), P(r) and lift, per run.

    lift = P(c,r) / [P(c) P(r)], over canonical sentences. Above 1 means the
    pair co-occurs in the same sentence more often than sentence-level
    independence implies; below 1, less often. It is DESCRIPTIVE and is not
    bootstrapped - with a small N(c) the ratio swings on a couple of
    sentences, which is why `support` travels in the same row and is never
    dropped.
    """
    rows = []
    for decoding in DECODING_ORDER:
        tensors = count_tensors(joint, decoding, kind)
        names = tensors["relation_names"]
        for m, model in enumerate(MODEL_ORDER):
            for r, run in enumerate(tensors["runs"]):
                total = tensors["n_sentences"][m, r].sum()
                for c, category in enumerate(CATEGORY_ORDER):
                    n_c = tensors["n_category"][m, r, :, c].sum()
                    p_c = n_c / total if total else np.nan
                    for k, relation in enumerate(names):
                        n_r = tensors["n_relation"][m, r, :, k].sum()
                        n_cr = tensors["n_joint"][m, r, :, c, k].sum()
                        p_r = n_r / total if total else np.nan
                        p_cr = n_cr / total if total else np.nan
                        expected = p_c * p_r
                        rows.append({
                            "model": model,
                            "decoding_group": decoding,
                            "run_label": run,
                            "semantic_category": category,
                            "discourse_relation": relation,
                            "n_sentences": int(total),
                            "support": int(n_cr),
                            "joint_prevalence": p_cr,
                            "prevalence_category": p_c,
                            "prevalence_relation": p_r,
                            "lift": (p_cr / expected
                                     if expected and expected > 0 else np.nan),
                            "support_is_thin": bool(n_c < THIN_SUPPORT),
                        })
    return _order(pd.DataFrame(rows))


def lift_summary(run_level_lift: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of lift and joint prevalence across runs."""
    grouped = run_level_lift.groupby(
        ["model", "decoding_group", "semantic_category", "discourse_relation"],
        observed=True,
    )
    summary = grouped.agg(
        n_runs=("run_label", "nunique"),
        mean_joint_prevalence=("joint_prevalence", "mean"),
        sd_joint_prevalence=("joint_prevalence",
                             lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
        mean_lift=("lift", "mean"),
        sd_lift=("lift", lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
        mean_support=("support", "mean"),
        total_support=("support", "sum"),
        support_is_thin=("support_is_thin", "any"),
    ).reset_index()
    return _order(summary)


# ---------------------------------------------------------------------------
# E - paired game-level bootstrap of P(r|c)
# ---------------------------------------------------------------------------

def _game_weights(n_games: int, n_replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        n_games, np.full(n_games, 1 / n_games), size=n_replicates
    ).astype(np.float64)


def conditional_prevalence_bootstrap(
    joint: pd.DataFrame,
    kind: str = "top_level",
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """E: 95% percentile CIs for pairwise model differences in P(r|c).

    One replicate: resample the 191 game ids with replacement; use the SAME
    ids for every model; keep every run and every sentence of a sampled game;
    recompute P(r|c) per run as (sum of N(c,r)) / (sum of N(c)) over the
    resampled games; average the three run values into one model value;
    difference the models.

    A replicate whose resampled games contain no sentence with category c has
    no defined P(r|c); those replicates are dropped and counted in
    `n_valid_replicates` rather than silently treated as zero.

    Lift is deliberately NOT bootstrapped - it stays descriptive, matching the
    frozen semantic analysis.
    """
    rows = []
    for decoding in DECODING_ORDER:
        tensors = count_tensors(joint, decoding, kind)
        games = tensors["games"]
        runs = tensors["runs"]
        names = tensors["relation_names"]
        n_games = len(games)
        weights = _game_weights(n_games, n_replicates, seed)     # (B, G)

        n_category = tensors["n_category"]
        n_joint = tensors["n_joint"]

        for c, category in enumerate(CATEGORY_ORDER):
            denominator = n_category[:, :, :, c].reshape(-1, n_games)
            boot_denominator = denominator @ weights.T           # (M*R, B)

            for k, relation in enumerate(names):
                numerator = n_joint[:, :, :, c, k].reshape(-1, n_games)
                boot_numerator = numerator @ weights.T
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(boot_denominator > 0,
                                     boot_numerator / boot_denominator, np.nan)
                ratio = ratio.reshape(len(MODEL_ORDER), len(runs), n_replicates)
                # A replicate can miss category c in every run for a thin
                # category (E2B Mechanical averages 6 sentences per run), so
                # an all-NaN slice here is expected, not a defect. It is
                # counted in n_valid_replicates below.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model_values = np.nanmean(ratio, axis=1)      # (M, B)

                observed_numerator = n_joint[:, :, :, c, k].sum(axis=2)
                observed_denominator = n_category[:, :, :, c].sum(axis=2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    observed = np.where(observed_denominator > 0,
                                        observed_numerator / observed_denominator,
                                        np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    observed = np.nanmean(observed, axis=1)       # (M,)

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
                            "semantic_category": category,
                            "discourse_relation": relation,
                            "model_a": MODEL_ORDER[i],
                            "model_b": MODEL_ORDER[j],
                            "prevalence_a": observed[i],
                            "prevalence_b": observed[j],
                            "difference": observed[i] - observed[j],
                            "ci_low": low,
                            "ci_high": high,
                            "ci_excludes_zero": bool(
                                np.isfinite(low) and np.isfinite(high)
                                and (low > 0 or high < 0)),
                            "n_semantic_sentences_a":
                                int(observed_denominator[i].sum()),
                            "n_semantic_sentences_b":
                                int(observed_denominator[j].sum()),
                            "support_is_thin": bool(
                                observed_denominator[i].sum()
                                < THIN_SUPPORT * len(runs)
                                or observed_denominator[j].sum()
                                < THIN_SUPPORT * len(runs)),
                            "n_valid_replicates": valid,
                            "n_games": n_games,
                            "n_replicates": n_replicates,
                            "seed": seed,
                        })
    return _order(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Matrix views, for figures and inspection
# ---------------------------------------------------------------------------

def prevalence_matrix(summary: pd.DataFrame, model: str, decoding: str,
                      value: str = "mean_conditional_prevalence",
                      relations: Sequence[str] = None) -> pd.DataFrame:
    """Categories x relations, for one model and decoding group."""
    relations = list(relations or TOP_LEVEL_ORDER)
    subset = summary.loc[
        summary["model"].astype(str).eq(model)
        & summary["decoding_group"].astype(str).eq(decoding)
    ]
    matrix = subset.pivot(index="semantic_category",
                          columns="discourse_relation", values=value)
    return matrix.reindex(index=CATEGORY_ORDER, columns=relations)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_final_tables(layers: Dict[str, pd.DataFrame],
                       alignment: Dict[str, pd.DataFrame],
                       joint: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Every joint table. Greedy is not a separate file: it is a
    `decoding_group` value in each table, never averaged with stochastic."""
    top_run = conditional_prevalence(joint, "top_level")
    top_lift = joint_prevalence_and_lift(joint, "top_level")
    sense_run = conditional_prevalence(joint, "sense")
    sense_lift = joint_prevalence_and_lift(joint, "sense")

    return {
        "J0_validation_report": validation_report(layers, alignment, joint),
        "J0b_unaligned_relations": alignment["unaligned"],
        "J1_conditional_prevalence_run_level": top_run,
        "J2_conditional_prevalence_summary":
            conditional_prevalence_summary(top_run),
        "J3_joint_prevalence_lift_run_level": top_lift,
        "J3b_joint_prevalence_lift_summary": lift_summary(top_lift),
        "J4_any_relation_given_category": any_relation_given_category(top_run),
        "J5_bootstrap_model_differences":
            conditional_prevalence_bootstrap(joint, "top_level"),
        "J6_finegrained_conditional_prevalence_run_level": sense_run,
        "J6b_finegrained_conditional_prevalence_summary":
            conditional_prevalence_summary(sense_run),
        "J7_finegrained_joint_prevalence_lift_summary": lift_summary(sense_lift),
    }


LATEX_TABLES = (
    "J2_conditional_prevalence_summary",
    "J4_any_relation_given_category",
    "J5_bootstrap_model_differences",
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
