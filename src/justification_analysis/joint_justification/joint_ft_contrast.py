"""BASE vs fine-tuned JOINT discourse x semantic contrast (RQ3), matched games.

The separate RQ3 strands answered "did the discourse marking change?" and "did
the semantic content change?". This one answers a question neither can: did the
RELATIONSHIP between them change - is semantic content still organised through
the same explicit discourse relations after familiarisation?

## What is reused unchanged

Everything that defines a quantity. The parser output, the semantic taxonomy,
the accepted-relation rule (`is_connective`), the byte-exact sentence
alignment, the localization rule, the density definition, the lift definition
and the per-run-then-average aggregation all come from the frozen RQ2 modules:

* `joint_final.load_layers / align_relations / build_joint_sentences`;
* `justification_joint.load_justification_metadata /
   build_justification_frame / count_tensors`.

`count_tensors` in particular is called directly, so the ratio-of-sums density
(100 * relations inside justifications carrying c / their words) and the
justification-level lift are computed from the frozen arrays rather than from a
second reading of them.

## What is new, and why it had to be

The frozen bootstraps difference MODELS. RQ3 differences CONDITIONS within a
model. `_paired_bootstrap` hard-codes the model pairing and resamples the whole
three-model grid at once, so it cannot express "BASE vs FT for 31B on its own
188 games". `condition_contrast` below reuses the same resampling scheme - one
multinomial multiplicity vector over games, applied to both sides, per run then
averaged - and differences the two conditions instead.

The metrics it differences are recomputed from `count_tensors` output rather
than from the frozen table functions, because those aggregate over all three
models jointly. `test_matched_joint_equivalence` pins that recomputation
against `conditional_density`, `joint_prevalence_and_lift`,
`conditional_prevalence` and `localization_rate` on the full BASE stochastic
set, which is the one input where both paths are valid.

## Matched games

Imported from `discourse_ft_contrast`, not restated, so all three RQ3 strands
describe identical game sets by construction: 190 / 191 / 188 for E2B / E4B /
31B. A game is retained only if BOTH conditions have all three stochastic runs
and no empty justification in either.

## What this still does not establish

Unchanged from RQ2, and worth repeating because a BASE-FT contrast invites the
stronger reading: the parser identifies explicit connectives and their PDTB
sense, not discourse arguments. A justification carrying both a category and a
relation class is an association between the kind of information stated and the
kind of explicit marking used. A CHANGE in that association is a change in
stated justifications - never evidence that the relation structures the
category, and never evidence about internal reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.justification_analysis.comparison.discourse_ft_contrast import (
    excluded_games, matched_games,
)
from src.justification_analysis.joint import joint_final as jf
from src.justification_analysis.joint_justification import justification_joint as jj
from src.justification_analysis.pipeline.config import AnalysisConfig, default_config

CATEGORY_ORDER: List[str] = list(jj.CATEGORY_ORDER)
TOP_LEVEL_ORDER: List[str] = list(jj.TOP_LEVEL_ORDER)
SENSE_ORDER: List[str] = list(jj.SENSE_ORDER)
ANY_RELATION = jj.ANY_RELATION

BOOTSTRAP_SEED = jj.BOOTSTRAP_SEED
BOOTSTRAP_REPLICATES = jj.BOOTSTRAP_REPLICATES
LOW_SUPPORT_DIAGNOSTIC = jj.LOW_SUPPORT_DIAGNOSTIC

BASE_STAGE = "base"
FT_STAGE = "ft"
STOCHASTIC = "Stochastic"

# The associations RQ2 established. The contrast is anchored to these; it does
# not sweep every semantic x discourse pair and report whichever moved, which
# with 7 x 5 cells per model would be a multiple-comparisons exercise dressed
# up as a finding.
PRIMARY_PAIR = ("Mechanical", "Contingency")
ANCHORED_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("Mechanical", "Contingency"),
    ("ClaimComparison", "Temporal"),
    ("SocialJudgment", "Contingency"),
)
ANCHORED_SENSE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("Mechanical", "Contingency.Cause"),
    ("Mechanical", "Contingency.Condition"),
    ("ClaimComparison", "Temporal.Asynchronous"),
)


@dataclass
class JointCondition:
    """One condition's fully built joint layers."""
    stage: str
    config: AnalysisConfig
    layers: Dict[str, pd.DataFrame]
    alignment: Dict[str, pd.DataFrame]
    sentences_joint: pd.DataFrame
    justifications: pd.DataFrame

    @property
    def corpus(self) -> pd.DataFrame:
        """The stage's corpus, restricted to stochastic runs.

        Exposed under this name so `discourse_ft_contrast.matched_games` can be
        applied to a `JointCondition` unchanged - it reads only `.corpus` and
        `.stage`. Reusing it is the point: the matched sets must be the same
        objects the other two RQ3 strands used, not a third derivation.
        """
        corpus = self.layers["corpus"]
        return corpus.loc[corpus["decoding_group"].eq(STOCHASTIC)]


def load_condition(stage: str) -> JointCondition:
    """Build the frozen joint layers for one stage, refusing another stage's.

    `load_layers` goes through the manifest freshness gate, so a parser
    artifact that does not belong to this corpus stops the analysis here rather
    than producing a plausible contrast against the wrong relations.
    """
    config = default_config(stage=stage)
    layers = jf.load_layers(config=config)
    alignment = jf.align_relations(layers)
    sentences_joint = jf.build_joint_sentences(layers, alignment)
    metadata = jj.load_justification_metadata(config=config)
    justifications = jj.build_justification_frame(
        layers, alignment, sentences_joint, metadata)

    justifications = justifications.loc[
        justifications["decoding_group"].astype(str).eq(STOCHASTIC)].copy()

    return JointCondition(stage, config, layers, alignment,
                          sentences_joint, justifications)


# ---------------------------------------------------------------------------
# Alignment audit - requirement (1)
# ---------------------------------------------------------------------------

def id_alignment(condition: JointCondition) -> pd.DataFrame:
    """Do the discourse and semantic layers cover exactly the same ids?

    The two pipelines are joined on (model, game_id, run_label, sentence_id)
    and verified byte-exactly by `align_relations`, but that verifies the
    relations it could place. This asks the complementary question: is every
    justification present on both sides, and did any relation fail to align.
    """
    semantic_ids = set(condition.layers["semantic"]["justifications"]
                       .loc[lambda f: f["decoding_group"].astype(str)
                            .eq(STOCHASTIC), "justification_id"])
    joint_ids = set(condition.justifications["justification_id"])
    aligned = condition.alignment["aligned"]
    aligned_ids = set(aligned["justification_id_canonical"])
    unaligned = condition.alignment["unaligned"]

    # `aligned` spans every decoding, because alignment happens before the
    # stochastic filter. Counting it raw would compare a BASE total that
    # includes greedy against an FT total that has no greedy to include.
    stochastic_relations = int(
        aligned["justification_id_canonical"].isin(joint_ids).sum())

    rows = [
        ("semantic justifications (stochastic)", len(semantic_ids), ""),
        ("joint-frame justifications", len(joint_ids), ""),
        ("ids in semantic but not in the joint frame",
         len(semantic_ids - joint_ids), ""),
        ("ids in the joint frame but not in semantic",
         len(joint_ids - semantic_ids), ""),
        ("relations that failed byte-exact alignment", len(unaligned),
         "any value > 0 invalidates the contrast"),
        ("justifications carrying at least one aligned relation",
         len(aligned_ids & joint_ids), "descriptive"),
        ("aligned relations on stochastic justifications",
         stochastic_relations, "descriptive"),
        ("justifications with a relation that are NOT in the joint frame",
         len(aligned_ids - joint_ids),
         "greedy, excluded by design; 0 for a stage with no greedy pass"),
    ]
    return pd.DataFrame([
        {"stage": condition.stage, "check": name, "observed": value,
         "note": note}
        for name, value, note in rows
    ])


# ---------------------------------------------------------------------------
# Per (run, game) arrays for ONE model and ONE condition
# ---------------------------------------------------------------------------

def model_tensors(condition: JointCondition, model: str,
                  games: Sequence[str], kind: str = "top_level") -> Dict:
    """Frozen `count_tensors` output, narrowed to one model and its games.

    The frozen function builds a (model, run, game) grid from whatever frame it
    is handed; handing it one model's matched games gives that model's slice
    with the other two models' planes left at zero. Nothing about the counting,
    the word denominator or the ratio definitions is re-implemented here - only
    the subsetting, which is what a matched contrast needs and the frozen
    signature has no way to express.
    """
    frame = condition.justifications.loc[
        condition.justifications["model"].astype(str).eq(model)
        & condition.justifications["game_id"].isin(games)
    ]
    tensors = jj.count_tensors(frame, STOCHASTIC, kind)
    m = jj.MODEL_ORDER.index(model)

    assert list(tensors["games"]) == list(games), (
        f"{condition.stage}/{model}: game order differs between the tensor "
        f"and the matched set")

    runs = tensors["runs"]
    for r, run in enumerate(runs):
        n = tensors["n_justifications"][m, r].sum()
        assert n == len(games), (
            f"{condition.stage}/{model}/{run}: {int(n)} justifications for "
            f"{len(games)} matched games")

    # Unconditional per (run, game) totals, for the model-wide baseline
    # density. Same ratio-of-sums definition, denominator = every word.
    names, _ = jj._relation_columns(kind)
    _, count_columns = jj._count_columns(kind)
    index = {game: i for i, game in enumerate(games)}
    words_all = np.zeros((len(runs), len(games)))
    relations_all = np.zeros((len(runs), len(games), len(names)))
    # Column access is positional, not by attribute: sense names carry a dot
    # ("Contingency.Cause") and `itertuples` silently renames those to
    # positional placeholders, so an attribute lookup would fail or, worse,
    # read the wrong column.
    counts = frame[count_columns].to_numpy(float)
    words = frame["n_words"].to_numpy(float)
    run_labels = frame["run_label"].to_numpy()
    game_ids = frame["game_id"].to_numpy()
    for row in range(len(frame)):
        r = runs.index(run_labels[row])
        g = index[game_ids[row]]
        words_all[r, g] += words[row]
        relations_all[r, g] += counts[row]

    return {
        "model_index": m,
        "runs": list(runs),
        "games": list(games),
        "relation_names": list(names),
        "tensors": tensors,
        "words_all": words_all,
        "relations_all": relations_all,
    }


def localization_arrays(condition: JointCondition, model: str,
                        games: Sequence[str],
                        kind: str = "top_level") -> Dict[Tuple[str, str], np.ndarray]:
    """Per (run, game): justifications with both c and r, and how many of those
    carry the pair inside a single sentence.

    The rule is the frozen one from `localization_rate` - a justification
    counts as localized if at least one of its sentences carries the category
    and the relation together - applied per game so the rate can be
    bootstrapped rather than only reported.
    """
    names, presence_columns = jj._relation_columns(kind)
    frame = condition.justifications.loc[
        condition.justifications["model"].astype(str).eq(model)
        & condition.justifications["game_id"].isin(games)
    ]
    sentences = condition.sentences_joint.loc[
        condition.sentences_joint["justification_id"].isin(
            frame["justification_id"])]

    runs = sorted(frame["run_label"].unique())
    index = {game: i for i, game in enumerate(games)}
    shape = (len(runs), len(games))

    out: Dict[Tuple[str, str], np.ndarray] = {}
    for category in CATEGORY_ORDER:
        for name, column in zip(names, presence_columns):
            both = np.zeros(shape)
            local = np.zeros(shape)
            paired = frame.loc[frame[f"sem_{category}"] & frame[column]]
            localized = set(sentences.loc[
                sentences[f"sem_{category}"] & sentences[column],
                "justification_id"])
            for row in paired.itertuples(index=False):
                r = runs.index(row.run_label)
                g = index[row.game_id]
                both[r, g] += 1
                if row.justification_id in localized:
                    local[r, g] += 1
            out[(category, name)] = np.stack([local, both])
    return out


# ---------------------------------------------------------------------------
# Metrics, as (numerator, denominator) per (run, game) - so the point estimate
# and the bootstrap are literally the same arrays.
# ---------------------------------------------------------------------------

def _metric_arrays(model_data: Dict, localization: Dict,
                   category: str, relation: str) -> Dict[str, Tuple]:
    """Every anchored metric for one (category, relation) cell.

    Each entry is (numerator[R, G], denominator[R, G], scale) and is evaluated
    as scale * sum(numerator) / sum(denominator) per run, then averaged across
    runs - the frozen aggregation order.

    `lift` is the exception: it is a ratio of ratios, so it carries a callable
    instead and is evaluated on the same resampled sums.
    """
    m = model_data["model_index"]
    t = model_data["tensors"]
    c = CATEGORY_ORDER.index(category)
    k = model_data["relation_names"].index(relation)

    n_category = t["n_category"][m, :, :, c]
    n_relation = t["n_relation"][m, :, :, k]
    n_pair = t["n_pair"][m, :, :, c, k]
    n_just = t["n_justifications"][m]
    words_c = t["words_category"][m, :, :, c]
    relations_in_c = t["relations_in_category"][m, :, :, c, k]
    words_all = model_data["words_all"]
    relations_all = model_data["relations_all"][:, :, k]
    local, both = localization[(category, relation)]

    return {
        "category_prevalence": (n_category, n_just, 100.0),
        "conditional_prevalence": (n_pair, n_category, 100.0),
        "conditional_density": (relations_in_c, words_c, 100.0),
        "baseline_density": (relations_all, words_all, 100.0),
        "localization": (local, both, 100.0),
        "lift": (n_pair, n_category, n_relation, n_just),
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray,
           scale: float) -> float:
    """scale * sum(num) / sum(den) per run, averaged. NaN where unsupported."""
    num = numerator.sum(axis=1)
    den = denominator.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_run = np.where(den > 0, scale * num / den, np.nan)
    return float(np.nanmean(per_run)) if np.isfinite(per_run).any() else np.nan


def _ratio_boot(numerator: np.ndarray, denominator: np.ndarray, scale: float,
                weights: np.ndarray) -> np.ndarray:
    num = numerator @ weights.T
    den = denominator @ weights.T
    with np.errstate(divide="ignore", invalid="ignore"):
        per_run = np.where(den > 0, scale * num / den, np.nan)
    return np.nanmean(per_run, axis=0)


def _lift(n_pair, n_category, n_relation, n_just) -> float:
    """P(c,r) / (P(c) P(r)) per run, averaged - the frozen definition,
    rearranged to (n_cr * N) / (n_c * n_r) so it can take resampled sums."""
    num = n_pair.sum(axis=1) * n_just.sum(axis=1)
    den = n_category.sum(axis=1) * n_relation.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_run = np.where(den > 0, num / den, np.nan)
    return float(np.nanmean(per_run)) if np.isfinite(per_run).any() else np.nan


def _lift_boot(n_pair, n_category, n_relation, n_just, weights) -> np.ndarray:
    num = (n_pair @ weights.T) * (n_just @ weights.T)
    den = (n_category @ weights.T) * (n_relation @ weights.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_run = np.where(den > 0, num / den, np.nan)
    return np.nanmean(per_run, axis=0)


def point_estimates(condition: JointCondition, model: str,
                    games: Sequence[str], category: str, relation: str,
                    kind: str = "top_level") -> Dict[str, float]:
    """Every anchored metric for one cell, one condition. No resampling."""
    model_data = model_tensors(condition, model, games, kind)
    localization = localization_arrays(condition, model, games, kind)
    arrays = _metric_arrays(model_data, localization, category, relation)

    m, t = model_data["model_index"], model_data["tensors"]
    c = CATEGORY_ORDER.index(category)
    k = model_data["relation_names"].index(relation)
    _, both = localization[(category, relation)]

    values = {name: (_lift(*spec) if name == "lift" else _ratio(*spec))
              for name, spec in arrays.items()}
    values.update({
        "stage": condition.stage,
        "model": model,
        "semantic_category": category,
        "discourse_relation": relation,
        "n_games": len(games),
        # Support is per run; the diagnostic compares the per-run mean so it
        # means the same thing as the frozen LOW_SUPPORT_DIAGNOSTIC.
        "n_with_category_per_run": float(t["n_category"][m, :, :, c].sum(axis=1).mean()),
        "n_with_both_per_run": float(both.sum(axis=1).mean()),
        "low_support_diagnostic": bool(
            t["n_category"][m, :, :, c].sum(axis=1).mean() < LOW_SUPPORT_DIAGNOSTIC),
    })
    return values


def condition_contrast(base: JointCondition, ft: JointCondition, model: str,
                       games: Sequence[str], category: str, relation: str,
                       kind: str = "top_level",
                       n_replicates: int = BOOTSTRAP_REPLICATES,
                       seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """FT - BASE differences for every anchored metric, paired 95% CIs.

    One replicate resamples the matched games with replacement and applies the
    SAME resampled games to both conditions, so between-game variation cancels
    out of the difference. This is the frozen resampling scheme with the
    difference taken across conditions instead of across models.

    No p-values. `ci_excludes_zero` is a fact about the interval, not a
    decision rule.
    """
    base_data = model_tensors(base, model, games, kind)
    ft_data = model_tensors(ft, model, games, kind)
    base_local = localization_arrays(base, model, games, kind)
    ft_local = localization_arrays(ft, model, games, kind)

    base_arrays = _metric_arrays(base_data, base_local, category, relation)
    ft_arrays = _metric_arrays(ft_data, ft_local, category, relation)

    n_games = len(games)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_games, np.full(n_games, 1 / n_games), size=n_replicates
    ).astype(np.float64)

    base_support = base_arrays["conditional_prevalence"][1].sum(axis=1).mean()
    ft_support = ft_arrays["conditional_prevalence"][1].sum(axis=1).mean()

    rows = []
    for metric in ["category_prevalence", "conditional_prevalence",
                   "conditional_density", "baseline_density", "localization",
                   "lift"]:
        if metric == "lift":
            base_point = _lift(*base_arrays[metric])
            ft_point = _lift(*ft_arrays[metric])
            base_boot = _lift_boot(*base_arrays[metric], weights)
            ft_boot = _lift_boot(*ft_arrays[metric], weights)
        else:
            base_point = _ratio(*base_arrays[metric])
            ft_point = _ratio(*ft_arrays[metric])
            base_boot = _ratio_boot(*base_arrays[metric], weights)
            ft_boot = _ratio_boot(*ft_arrays[metric], weights)

        differences = ft_boot - base_boot
        valid = int(np.isfinite(differences).sum())
        if valid:
            low, high = np.nanpercentile(differences, [2.5, 97.5])
        else:
            low = high = np.nan

        rows.append({
            "model": model,
            "semantic_category": category,
            "discourse_relation": relation,
            "metric": metric,
            "n_games": n_games,
            "base": base_point,
            "ft": ft_point,
            "delta": ft_point - base_point,
            "ci_low": low,
            "ci_high": high,
            "ci_excludes_zero": bool(np.isfinite(low) and np.isfinite(high)
                                     and (low > 0 or high < 0)),
            "base_support_per_run": float(base_support),
            "ft_support_per_run": float(ft_support),
            "low_support_diagnostic": bool(
                min(base_support, ft_support) < LOW_SUPPORT_DIAGNOSTIC),
            "n_valid_replicates": valid,
            "n_replicates": n_replicates,
            "seed": seed,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sentence-level composition - which semantic categories carry a relation
# ---------------------------------------------------------------------------

def relation_composition(base: JointCondition, ft: JointCondition, model: str,
                         games: Sequence[str], relation: str,
                         kind: str = "top_level",
                         n_replicates: int = BOOTSTRAP_REPLICATES,
                         seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Of the SENTENCES carrying relation r, what share also carry category c?

    This is the RQ2 sentence-level reading that produced "81.6% of E2B's
    Contingency relations sit in Payoff sentences". It is a composition, so the
    shares do not sum to 100 - a sentence can carry several categories - and it
    is descriptive: it says where a relation surfaces, not what it attaches to.

    Denominator is sentences carrying the relation, per run, then averaged.
    """
    names, presence_columns = jj._relation_columns(kind)
    column = presence_columns[names.index(relation)]

    def arrays(condition: JointCondition):
        frame = condition.justifications.loc[
            condition.justifications["model"].astype(str).eq(model)
            & condition.justifications["game_id"].isin(games)]
        sentences = condition.sentences_joint.loc[
            condition.sentences_joint["justification_id"].isin(
                frame["justification_id"])
            & condition.sentences_joint[column]]
        sentences = sentences.merge(
            frame[["justification_id", "run_label", "game_id"]],
            on="justification_id", how="left", suffixes=("", "_j"))
        runs = sorted(frame["run_label"].unique())
        index = {game: i for i, game in enumerate(games)}
        shape = (len(runs), len(games))
        total = np.zeros(shape)
        by_category = {c: np.zeros(shape) for c in CATEGORY_ORDER}
        for row in sentences.itertuples(index=False):
            r = runs.index(row.run_label)
            g = index[row.game_id]
            total[r, g] += 1
            for category in CATEGORY_ORDER:
                if getattr(row, f"sem_{category}"):
                    by_category[category][r, g] += 1
        return total, by_category

    base_total, base_by = arrays(base)
    ft_total, ft_by = arrays(ft)

    n_games = len(games)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_games, np.full(n_games, 1 / n_games), size=n_replicates
    ).astype(np.float64)

    rows = []
    for category in CATEGORY_ORDER:
        base_point = _ratio(base_by[category], base_total, 100.0)
        ft_point = _ratio(ft_by[category], ft_total, 100.0)
        differences = (_ratio_boot(ft_by[category], ft_total, 100.0, weights)
                       - _ratio_boot(base_by[category], base_total, 100.0, weights))
        valid = int(np.isfinite(differences).sum())
        low, high = (np.nanpercentile(differences, [2.5, 97.5])
                     if valid else (np.nan, np.nan))
        rows.append({
            "model": model,
            "discourse_relation": relation,
            "semantic_category": category,
            "n_games": n_games,
            "base_pct_of_relation_sentences": base_point,
            "ft_pct_of_relation_sentences": ft_point,
            "delta_pp": ft_point - base_point,
            "ci_low": low,
            "ci_high": high,
            "ci_excludes_zero": bool(np.isfinite(low) and np.isfinite(high)
                                     and (low > 0 or high < 0)),
            "base_relation_sentences_per_run": float(base_total.sum(axis=1).mean()),
            "ft_relation_sentences_per_run": float(ft_total.sum(axis=1).mean()),
        })
    return pd.DataFrame(rows)
