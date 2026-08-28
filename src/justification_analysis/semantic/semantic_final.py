"""Canonical FINAL semantic results for the frozen justification annotations.

The input is the merged DeepSeek annotation run for the ACTIVE stage,
resolved through `AnalysisConfig.semantic_annotations_path`. For the base
stage that is the frozen annotation run (schema `frozen`, 2,292
justifications); another stage resolves to its own annotation run and raises
if that run does not exist. Nothing here re-annotates, re-prompts or
edits that file; the one documented repair below is applied in memory, to the
loaded frame, and is reported as an artifact.

The question this layer answers is WHAT TYPES OF INFORMATION a model invokes
in its stated justification. It does not speak to internal reasoning, to
faithfulness, or to causation, and no function here is named or shaped as if
it did.

Three aggregation rules hold everywhere and are not negotiable:

  * the PRIMARY unit is the JUSTIFICATION. A category counts once per
    justification no matter how many sentences carry it. The models were
    prompted for 3-5 sentences, so repetition within a justification is a
    length artefact, not a stronger semantic signal. Sentence-normalised
    density exists only as a sensitivity check;
  * descriptive tables compute each run independently, then report mean and SD
    across the three stochastic runs. Greedy is one run, kept separate, SD
    undefined (NaN). Stochastic and greedy are never pooled;
  * every bootstrap resamples GAMES, not justifications or runs. The three
    stochastic runs of a game are repeated realisations of the same transcript,
    not independent games, so they travel together in the resample. Model
    comparisons reuse the identical resampled game ids across models, which
    pairs the comparison and cancels game-level variation.

`Other` is carried through the corpus descriptives and the run-level prevalence
table so its rarity is visible, and is then excluded from co-occurrence, the
bootstrap comparisons and the correctness analyses. It is 0.3% of labels; it
would add noise, not information.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.justification_analysis.comparison.discourse_statistics import to_latex
from src.pt_annotation.justification_schema import get_schema

# ---------------------------------------------------------------------------
# Vocabulary and design constants
# ---------------------------------------------------------------------------

FROZEN_SCHEMA = get_schema("frozen")
OTHER_CATEGORY = "Other"

# Taxonomy order is taken from the frozen schema itself rather than restated,
# so the analysis cannot drift from the prompt that produced the labels.
ALL_CATEGORIES: List[str] = list(FROZEN_SCHEMA.categories)
CATEGORY_ORDER: List[str] = [c for c in ALL_CATEGORIES if c != OTHER_CATEGORY]

MODEL_KEYS = ["2B", "4B", "31B"]
MODEL_DISPLAY = {"2B": "Gemma 4 2B", "4B": "Gemma 4 4B", "31B": "Gemma 4 31B"}
MODEL_ORDER = [MODEL_DISPLAY[key] for key in MODEL_KEYS]

DECODING_ORDER = ["Stochastic", "Greedy"]
STOCHASTIC_RUNS = ["run_1", "run_2", "run_3"]
GREEDY_RUNS = ["greedy_t0"]
RUNS_BY_DECODING = {"Stochastic": STOCHASTIC_RUNS, "Greedy": GREEDY_RUNS}
RUN_KEYS = ["model", "decoding_group", "run_label"]

# Paths derive from the active configuration. The base stage resolves to the
# frozen annotation run and to the base artifact namespace, so
# base behaviour is byte-identical to before; any other stage resolves to its
# own annotation run and its own output namespace, and raises if that run does
# not exist rather than reading base annotations.
def _resolved_config(repo_root=None, config=None):
    from src.justification_analysis.pipeline import config as pipeline_config
    if config is not None:
        return config
    return pipeline_config.AnalysisConfig(
        repo_root=Path(repo_root) if repo_root
        else pipeline_config.find_repo_root())


def annotations_path(config=None) -> Path:
    return _resolved_config(config=config).semantic_annotations_path


def input_dir(config=None) -> Path:
    return _resolved_config(config=config).semantic_input_dir


def final_tables_dir(config=None) -> Path:
    return _resolved_config(config=config).semantic_dir / "thesis_tables" / "final_semantic"


def final_figures_dir(config=None) -> Path:
    return _resolved_config(config=config).semantic_dir / "figures" / "final_semantic"


# Legacy module-level names, resolved through the default (base) config so
# existing callers keep working. New code should pass a config explicitly.
def __getattr__(name):
    _legacy = {
        "ANNOTATIONS_SUBPATH": lambda: annotations_path().relative_to(
            _resolved_config().repo_root),
        "INPUT_SUBPATH": lambda: input_dir().relative_to(
            _resolved_config().repo_root),
        "FINAL_TABLES_SUBPATH": lambda: final_tables_dir().relative_to(
            _resolved_config().repo_root),
        "FINAL_FIGURES_SUBPATH": lambda: final_figures_dir().relative_to(
            _resolved_config().repo_root),
        "ARTIFACT_SUBPATH": lambda: _resolved_config().semantic_dir.relative_to(
            _resolved_config().repo_root),
    }
    if name in _legacy:
        return _legacy[name]()
    raise AttributeError(name)

BOOTSTRAP_SEED = 20260826
BOOTSTRAP_REPLICATES = 10_000

# Frozen at the annotation freeze point. If any of these change the corpus is
# not the one these tables describe, and every number below is stale.
INVARIANTS = {
    "justifications": 2292,
    "games": 191,
    "justifications_per_model": 764,
    "justifications_per_model_run": 191,
    "sentences": 8044,
    "labels": 11526,
}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _source_sentences(repo_root: Path, config=None) -> Dict[str, Dict[int, str]]:
    """The sentence text as it was SENT to the annotator, per justification.

    This is the authority for sentence text: the annotator was asked to echo
    each sentence back verbatim, so any disagreement between the two is an
    annotator transcription error, repaired from here.
    """
    source: Dict[str, Dict[int, str]] = {}
    for path in sorted(input_dir(config).glob("*.jsonl")):
        for record in _read_jsonl(path):
            source[record["justification_id"]] = {
                sentence["sentence_id"]: sentence["text"]
                for sentence in record["sentences"]
            }
    return source


def load_annotations(repo_root: Path = None, config=None) -> Dict[str, pd.DataFrame]:
    """Load the frozen annotations into three tidy frames plus a repair log.

    Returns a dict with:
      justifications - one row per justification, with binary presence columns
      sentences      - one row per (justification, sentence)
      labels         - one row per (justification, sentence, category assignment)
      repairs        - one row per sentence whose text was repaired from source

    The repair is applied here, once, so that nothing downstream sees the
    altered text and no analysis silently disagrees with another about what a
    sentence said.
    """
    config = _resolved_config(repo_root, config)
    config.require_semantic_inputs()
    repo_root = config.repo_root
    records = _read_jsonl(annotations_path(config))
    source = _source_sentences(repo_root, config)

    justification_rows: List[dict] = []
    sentence_rows: List[dict] = []
    label_rows: List[dict] = []
    repair_rows: List[dict] = []

    for record in records:
        meta = record["metadata"]
        annotation = record["annotation"]
        justification_id = meta["justification_id"]
        model_key = meta["model_under_annotation"]
        run_label = meta["run_label"]
        decoding = "Greedy" if run_label in GREEDY_RUNS else "Stochastic"
        source_sentences = source.get(justification_id, {})

        present = {category: False for category in ALL_CATEGORIES}
        n_labels = 0

        for sentence in annotation["sentences"]:
            sentence_id = sentence["sentence_id"]
            annotated_text = sentence["text"]
            original_text = source_sentences.get(sentence_id)

            text = annotated_text
            if original_text is not None and original_text != annotated_text:
                spans = [a.get("evidence_span") for a in sentence["annotations"]]
                repair_rows.append({
                    "justification_id": justification_id,
                    "sentence_id": sentence_id,
                    "annotated_text": annotated_text,
                    "source_text": original_text,
                    "n_labels_on_sentence": len(sentence["annotations"]),
                    "spans_verbatim_before": sum(
                        1 for s in spans if s is not None and s in annotated_text
                    ),
                    "spans_verbatim_after": sum(
                        1 for s in spans if s is not None and s in original_text
                    ),
                })
                text = original_text

            for assignment in sentence["annotations"]:
                category = assignment["category"]
                span = assignment.get("evidence_span")
                present[category] = True
                n_labels += 1
                label_rows.append({
                    "justification_id": justification_id,
                    "sentence_id": sentence_id,
                    "category": category,
                    "evidence_span": span,
                    "span_is_verbatim": bool(span is not None and span in text),
                    "other_description": assignment.get("other_description"),
                })

            sentence_rows.append({
                "justification_id": justification_id,
                "sentence_id": sentence_id,
                "text": text,
                "text_repaired": text != annotated_text,
                "n_labels": len(sentence["annotations"]),
            })

        row = {
            "justification_id": justification_id,
            "model_key": model_key,
            "model": MODEL_DISPLAY[model_key],
            "run_label": run_label,
            "decoding_group": decoding,
            "game_id": meta["game_id"],
            "is_correct": bool(meta["is_correct"]),
            "vote": annotation.get("vote"),
            "n_sentences": len(annotation["sentences"]),
            "n_sentences_metadata": meta["n_sentences"],
            "n_labels": n_labels,
            "n_validation_flags": len(meta.get("validation_flags") or []),
        }
        row.update({f"has_{c}": present[c] for c in ALL_CATEGORIES})
        row["n_distinct_categories"] = sum(present[c] for c in CATEGORY_ORDER)
        row["has_any_substantive"] = row["n_distinct_categories"] > 0
        justification_rows.append(row)

    justifications = _order(pd.DataFrame(justification_rows))
    sentences = pd.DataFrame(sentence_rows)
    labels = pd.DataFrame(label_rows)
    repairs = pd.DataFrame(repair_rows)

    return {
        "justifications": justifications,
        "sentences": sentences,
        "labels": labels,
        "repairs": repairs,
    }


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model" in out.columns:
        out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    if "decoding_group" in out.columns:
        out["decoding_group"] = pd.Categorical(
            out["decoding_group"], DECODING_ORDER, ordered=True
        )
    if "category" in out.columns:
        known = [c for c in ALL_CATEGORIES if c in set(out["category"])]
        out["category"] = pd.Categorical(out["category"], known, ordered=True)
    sort_cols = [c for c in ("model", "decoding_group", "run_label", "category")
                 if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


# ---------------------------------------------------------------------------
# S0 - integrity
# ---------------------------------------------------------------------------

def integrity_summary(data: Dict[str, pd.DataFrame],
                      repo_root: Path) -> pd.DataFrame:
    """Every count recomputed from the file, with its expected value.

    Nothing here trusts a previously reported number: the expectations are the
    frozen invariants, and anything that disagrees shows up as a FAIL row.
    """
    justifications = data["justifications"]
    sentences = data["sentences"]
    labels = data["labels"]

    source = _source_sentences(Path(repo_root))
    source_sentence_total = sum(len(v) for v in source.values())

    per_sentence = sentences["n_labels"]
    per_justification_categories = justifications["n_distinct_categories"]

    checks: List[Tuple[str, object, object]] = [
        ("justifications", len(justifications), INVARIANTS["justifications"]),
        ("unique justification ids",
         justifications["justification_id"].nunique(),
         INVARIANTS["justifications"]),
        ("duplicate justification ids",
         int(justifications["justification_id"].duplicated().sum()), 0),
        ("games", justifications["game_id"].nunique(), INVARIANTS["games"]),
        ("models", justifications["model"].nunique(), 3),
        ("runs", justifications["run_label"].nunique(), 4),
        ("justifications per model",
         sorted(int(n) for n in
                justifications.groupby("model", observed=True).size().unique()),
         [INVARIANTS["justifications_per_model"]]),
        ("justifications per model x run",
         sorted(int(n) for n in
                justifications.groupby(["model", "run_label"], observed=True)
                .size().unique()),
         [INVARIANTS["justifications_per_model_run"]]),
        ("model x game x run fully crossed",
         len(justifications.drop_duplicates(["model", "game_id", "run_label"])),
         INVARIANTS["justifications"]),
        ("sentences", len(sentences), INVARIANTS["sentences"]),
        ("sentences match the input shards",
         len(sentences), source_sentence_total),
        ("sentence count matches metadata n_sentences",
         int((justifications["n_sentences"]
              != justifications["n_sentences_metadata"]).sum()), 0),
        ("category assignments (labels)", len(labels), INVARIANTS["labels"]),
        ("labels use only frozen-schema categories",
         sorted(set(labels["category"].astype(str)) - set(ALL_CATEGORIES)), []),
        ("empty-label sentences", int((per_sentence == 0).sum()), "descriptive"),
        ("max labels on one sentence", int(per_sentence.max()), "descriptive"),
        ("justifications with no substantive category",
         int((per_justification_categories == 0).sum()), "descriptive"),
        ("Other assignments",
         int((labels["category"] == OTHER_CATEGORY).sum()), "descriptive"),
        ("missing vote", int(justifications["vote"].isna().sum()), 0),
        ("missing is_correct", int(justifications["is_correct"].isna().sum()), 0),
        ("missing evidence_span", int(labels["evidence_span"].isna().sum()),
         "descriptive"),
        ("non-verbatim evidence spans",
         int((~labels["span_is_verbatim"]).sum()), "descriptive"),
        ("justifications carrying a validation flag",
         int((justifications["n_validation_flags"] > 0).sum()), "descriptive"),
        ("sentence texts repaired from source", len(data["repairs"]),
         "descriptive"),
    ]

    rows = []
    for name, observed, expected in checks:
        if expected == "descriptive":
            status = "descriptive"
        else:
            status = "OK" if observed == expected else "FAIL"
        rows.append({
            "check": name,
            "observed": str(observed),
            "expected": str(expected),
            "status": status,
        })
    return pd.DataFrame(rows)


def multilabel_distribution(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Labels per sentence, with BOTH denominators stated explicitly.

    An earlier quick summary quoted percentages that did not sum to 100 because
    it mixed the two denominators. They are different questions:

      * of ALL sentences, how many carry k labels - includes the unlabelled
        ones, so the column sums to 100;
      * of LABELLED sentences only, how many carry k labels - drops the
        unlabelled ones and so describes the multi-label behaviour of the
        annotator where it actually applied a label.
    """
    counts = data["sentences"]["n_labels"].value_counts().sort_index()
    total = int(counts.sum())
    labelled_total = int(counts.loc[counts.index > 0].sum())

    rows = []
    for k, n in counts.items():
        rows.append({
            "labels_per_sentence": int(k),
            "n_sentences": int(n),
            "pct_of_all_sentences": 100 * n / total,
            "pct_of_labelled_sentences": (
                100 * n / labelled_total if k > 0 else np.nan
            ),
        })
    frame = pd.DataFrame(rows)
    frame.attrs["n_sentences"] = total
    frame.attrs["n_labelled_sentences"] = labelled_total
    return frame


def annotation_summary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """S1: corpus descriptives per model and decoding group."""
    justifications = data["justifications"]
    sentences = data["sentences"].merge(
        justifications[["justification_id", "model", "decoding_group"]],
        on="justification_id", how="left",
    )

    grouped = justifications.groupby(["model", "decoding_group"], observed=True)
    summary = grouped.agg(
        n_justifications=("justification_id", "size"),
        n_games=("game_id", "nunique"),
        n_sentences=("n_sentences", "sum"),
        mean_sentences_per_justification=("n_sentences", "mean"),
        sd_sentences_per_justification=("n_sentences", "std"),
        n_labels=("n_labels", "sum"),
        mean_labels_per_justification=("n_labels", "mean"),
        mean_distinct_categories=("n_distinct_categories", "mean"),
        pct_no_substantive_category=("has_any_substantive",
                                     lambda s: 100 * (~s).mean()),
        pct_correct=("is_correct", lambda s: 100 * s.mean()),
    ).reset_index()

    empty = (
        sentences.assign(is_empty=sentences["n_labels"].eq(0))
        .groupby(["model", "decoding_group"], observed=True)["is_empty"]
        .agg(n_empty_label_sentences="sum", pct_empty_label_sentences="mean")
        .reset_index()
    )
    empty["pct_empty_label_sentences"] *= 100

    summary = summary.merge(empty, on=["model", "decoding_group"], how="left")
    summary["labels_per_sentence"] = summary["n_labels"] / summary["n_sentences"]
    return _order(summary)


# ---------------------------------------------------------------------------
# Presence tensors - the shared substrate for every analysis below
# ---------------------------------------------------------------------------

def presence_tensor(justifications: pd.DataFrame,
                    decoding: str) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """(games, runs, presence[M, R, G, C], correct[M, R, G]) for one decoding.

    Games are in a fixed sorted order shared by every model and run, which is
    what makes the paired bootstrap below meaningful: replicate weights index
    the same game for 2B, 4B and 31B alike.
    """
    runs = RUNS_BY_DECODING[decoding]
    frame = justifications.loc[
        justifications["decoding_group"].astype(str).eq(decoding)
    ]
    games = sorted(frame["game_id"].unique())
    game_index = {game: i for i, game in enumerate(games)}

    shape = (len(MODEL_ORDER), len(runs), len(games))
    presence = np.zeros(shape + (len(CATEGORY_ORDER),), dtype=np.float64)
    correct = np.zeros(shape, dtype=np.float64)
    filled = np.zeros(shape, dtype=bool)

    columns = [f"has_{c}" for c in CATEGORY_ORDER]
    for row in frame.itertuples(index=False):
        m = MODEL_ORDER.index(str(row.model))
        r = runs.index(row.run_label)
        g = game_index[row.game_id]
        presence[m, r, g] = [bool(getattr(row, col)) for col in columns]
        correct[m, r, g] = float(row.is_correct)
        filled[m, r, g] = True

    assert filled.all(), (
        f"{decoding}: the model x run x game grid has holes "
        f"({int((~filled).sum())} missing cells)"
    )
    return games, runs, presence, correct


def _game_weights(n_games: int, n_replicates: int, seed: int) -> np.ndarray:
    """Multinomial multiplicities, equivalent to sampling game ids with
    replacement but reusable across models, runs, categories and metrics."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        n_games, np.full(n_games, 1 / n_games), size=n_replicates
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# S2 / S3 - semantic profile
# ---------------------------------------------------------------------------

def run_level_prevalence(justifications: pd.DataFrame) -> pd.DataFrame:
    """S2: one row per (model, decoding, run, category).

    prevalence = share of that run's 191 justifications in which the category
    appears at least once. `Other` is included here, and only here, so its
    rarity is on the record before it is dropped.
    """
    rows = []
    for (model, decoding, run), frame in justifications.groupby(
        RUN_KEYS, observed=True
    ):
        n = len(frame)
        for category in ALL_CATEGORIES:
            present = int(frame[f"has_{category}"].sum())
            rows.append({
                "model": model,
                "decoding_group": decoding,
                "run_label": run,
                "category": category,
                "is_substantive": category != OTHER_CATEGORY,
                "n_justifications": n,
                "n_present": present,
                "prevalence": present / n,
            })
    return _order(pd.DataFrame(rows))


def model_prevalence(run_level: pd.DataFrame) -> pd.DataFrame:
    """S3: mean and SD of prevalence across runs, per model and decoding.

    Greedy is a single run: its SD is NaN by construction, never 0, so a
    reader cannot mistake it for a measured spread of zero.
    """
    grouped = run_level.groupby(
        ["model", "decoding_group", "category", "is_substantive"], observed=True
    )
    summary = grouped.agg(
        n_runs=("run_label", "nunique"),
        prevalence_mean=("prevalence", "mean"),
        prevalence_sd=("prevalence", lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
        prevalence_min=("prevalence", "min"),
        prevalence_max=("prevalence", "max"),
        n_present_mean=("n_present", "mean"),
    ).reset_index()
    return _order(summary)


def sentence_length_summary(justifications: pd.DataFrame) -> pd.DataFrame:
    """Justification length by model and run - the precondition for deciding
    whether the sentence-normalised sensitivity analysis is needed at all."""
    rows = []
    for (model, decoding, run), frame in justifications.groupby(
        RUN_KEYS, observed=True
    ):
        counts = frame["n_sentences"]
        rows.append({
            "model": model,
            "decoding_group": decoding,
            "run_label": run,
            "n_justifications": len(frame),
            "n_sentences": int(counts.sum()),
            "mean_sentences": counts.mean(),
            "sd_sentences": counts.std(ddof=1),
            "median_sentences": counts.median(),
            "min_sentences": int(counts.min()),
            "max_sentences": int(counts.max()),
        })
    return _order(pd.DataFrame(rows))


def density_sensitivity(justifications: pd.DataFrame,
                        labels: pd.DataFrame) -> pd.DataFrame:
    """SENSITIVITY ONLY: category assignments per 100 sentences.

    This is the metric the primary analysis deliberately does not use. It is
    computed so the question "would normalising for length change the story?"
    can be answered with numbers instead of assertion. Promotion to a primary
    result would require the length differences to actually matter, which the
    notebook checks by comparing model orderings against S3.
    """
    joined = labels.merge(
        justifications[["justification_id"] + RUN_KEYS],
        on="justification_id", how="left",
    )
    totals = justifications.groupby(RUN_KEYS, observed=True)["n_sentences"].sum()

    rows = []
    for (model, decoding, run), n_sentences in totals.items():
        subset = joined.loc[
            joined["model"].astype(str).eq(str(model))
            & joined["decoding_group"].astype(str).eq(str(decoding))
            & joined["run_label"].eq(run)
        ]
        counts = subset["category"].value_counts()
        for category in CATEGORY_ORDER:
            n = int(counts.get(category, 0))
            rows.append({
                "model": model,
                "decoding_group": decoding,
                "run_label": run,
                "category": category,
                "n_assignments": n,
                "n_sentences": int(n_sentences),
                "assignments_per_100_sentences": 100 * n / n_sentences,
            })
    run_frame = _order(pd.DataFrame(rows))

    summary = run_frame.groupby(
        ["model", "decoding_group", "category"], observed=True
    ).agg(
        n_runs=("run_label", "nunique"),
        density_mean=("assignments_per_100_sentences", "mean"),
        density_sd=("assignments_per_100_sentences",
                    lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
    ).reset_index()
    return _order(summary)


# ---------------------------------------------------------------------------
# S4 - paired game-level bootstrap of pairwise model differences
# ---------------------------------------------------------------------------

def prevalence_bootstrap_differences(
    justifications: pd.DataFrame,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """S4: 95% percentile CIs for pairwise model differences in prevalence.

    One replicate: resample the 191 game ids with replacement; use the SAME
    ids for every model; recompute prevalence separately for each stochastic
    run; average the three run values into one model value; difference the
    models. Greedy is bootstrapped separately and has a single run, so its
    interval reflects between-game variation only.

    No p-values: the interval is a descriptive uncertainty statement about the
    difference, and `ci_excludes_zero` is recorded as a fact about the interval,
    not as a decision rule.
    """
    rows = []
    for decoding in DECODING_ORDER:
        games, runs, presence, _ = presence_tensor(justifications, decoding)
        n_games = len(games)
        weights = _game_weights(n_games, n_replicates, seed)      # (B, G)

        for c, category in enumerate(CATEGORY_ORDER):
            indicator = presence[:, :, :, c].reshape(-1, n_games)  # (M*R, G)
            boot = (indicator @ weights.T) / n_games               # (M*R, B)
            boot = boot.reshape(len(MODEL_ORDER), len(runs), n_replicates)
            model_values = boot.mean(axis=1)                       # (M, B)

            observed = presence[:, :, :, c].mean(axis=2).mean(axis=1)  # (M,)

            for i in range(len(MODEL_ORDER)):
                for j in range(i + 1, len(MODEL_ORDER)):
                    differences = model_values[i] - model_values[j]
                    low, high = np.percentile(differences, [2.5, 97.5])
                    rows.append({
                        "decoding_group": decoding,
                        "category": category,
                        "model_a": MODEL_ORDER[i],
                        "model_b": MODEL_ORDER[j],
                        "prevalence_a": observed[i],
                        "prevalence_b": observed[j],
                        "difference": observed[i] - observed[j],
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                        "n_runs_averaged": len(runs),
                        "n_games": n_games,
                        "n_replicates": n_replicates,
                        "seed": seed,
                    })

    table = pd.DataFrame(rows)
    return _order(table).sort_values(
        ["decoding_group", "category", "model_a", "model_b"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# S5 / S6 - justification-level co-occurrence
# ---------------------------------------------------------------------------

def cooccurrence(justifications: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Joint prevalence, support and lift over the seven substantive categories.

    Joint prevalence is the MAIN quantity: the share of justifications carrying
    both categories. Lift divides that by the product of the marginals and is
    diagnostic only - with a rare category the ratio moves violently on a
    handful of justifications, which is exactly why raw support travels beside
    it in the same table and is never dropped.

    Diagonal: joint prevalence on the diagonal is the marginal prevalence, which
    is meaningful and kept. Lift on the diagonal would be 1/P(c) - an artefact
    of the definition carrying no information about association - so it is set
    to NaN rather than plotted as a very large number.

    Computed per run, then averaged, following the same rule as every other
    descriptive here.
    """
    run_rows = []
    for (model, decoding, run), frame in justifications.groupby(
        RUN_KEYS, observed=True
    ):
        n = len(frame)
        matrix = frame[[f"has_{c}" for c in CATEGORY_ORDER]].to_numpy(dtype=float)
        marginals = matrix.mean(axis=0)
        joint_counts = matrix.T @ matrix                       # (C, C)
        joint = joint_counts / n
        expected = np.outer(marginals, marginals)
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(expected > 0, joint / expected, np.nan)
        np.fill_diagonal(lift, np.nan)

        for i, category_a in enumerate(CATEGORY_ORDER):
            for j, category_b in enumerate(CATEGORY_ORDER):
                run_rows.append({
                    "model": model,
                    "decoding_group": decoding,
                    "run_label": run,
                    "category_a": category_a,
                    "category_b": category_b,
                    "is_diagonal": i == j,
                    "n_justifications": n,
                    "support": int(joint_counts[i, j]),
                    "joint_prevalence": joint[i, j],
                    "prevalence_a": marginals[i],
                    "prevalence_b": marginals[j],
                    "lift": lift[i, j],
                })

    run_level = pd.DataFrame(run_rows)

    grouped = run_level.groupby(
        ["model", "decoding_group", "category_a", "category_b", "is_diagonal"],
        observed=True,
    )
    summary = grouped.agg(
        n_runs=("run_label", "nunique"),
        support_mean=("support", "mean"),
        support_total=("support", "sum"),
        joint_prevalence_mean=("joint_prevalence", "mean"),
        joint_prevalence_sd=("joint_prevalence",
                             lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
        prevalence_a_mean=("prevalence_a", "mean"),
        prevalence_b_mean=("prevalence_b", "mean"),
        lift_mean=("lift", "mean"),
        lift_sd=("lift", lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
    ).reset_index()

    joint_table = summary[[
        "model", "decoding_group", "category_a", "category_b", "is_diagonal",
        "n_runs", "support_mean", "support_total",
        "joint_prevalence_mean", "joint_prevalence_sd",
        "prevalence_a_mean", "prevalence_b_mean",
    ]].copy()
    lift_table = summary[[
        "model", "decoding_group", "category_a", "category_b", "is_diagonal",
        "n_runs", "support_mean", "support_total",
        "joint_prevalence_mean", "lift_mean", "lift_sd",
    ]].copy()

    return {
        "run_level": _order(run_level),
        "joint": _order(joint_table),
        "lift": _order(lift_table),
    }


def ranked_pairs(joint: pd.DataFrame, lift: pd.DataFrame,
                 decoding: str = "Stochastic",
                 min_support: float = 10.0) -> pd.DataFrame:
    """Unordered category pairs, one row each, joint prevalence beside lift.

    `min_support` does not delete rows - it flags them. A pair resting on four
    justifications can carry a spectacular lift, and the honest response is to
    keep it visible and marked, not to silently drop it or quietly rank it
    first.
    """
    keys = ["model", "decoding_group", "category_a", "category_b"]
    merged = joint.merge(
        lift[keys + ["lift_mean", "lift_sd"]], on=keys, how="left"
    )
    merged = merged.loc[
        ~merged["is_diagonal"]
        & merged["decoding_group"].astype(str).eq(decoding)
    ].copy()

    merged["pair"] = [
        " + ".join(sorted((str(a), str(b))))
        for a, b in zip(merged["category_a"], merged["category_b"])
    ]
    merged = merged.drop_duplicates(["model", "pair"])
    merged["support_is_thin"] = merged["support_mean"] < min_support
    merged["min_support_threshold"] = min_support

    columns = [
        "model", "decoding_group", "pair", "joint_prevalence_mean",
        "prevalence_a_mean", "prevalence_b_mean", "support_mean",
        "support_total", "lift_mean", "lift_sd", "support_is_thin",
        "min_support_threshold",
    ]
    return _order(merged[columns]).sort_values(
        ["model", "joint_prevalence_mean"], ascending=[True, False]
    ).reset_index(drop=True)


def cooccurrence_matrix(joint: pd.DataFrame, model: str, decoding: str,
                        value: str = "joint_prevalence_mean") -> pd.DataFrame:
    """Square matrix view of one model's co-occurrence, for plotting/checking."""
    subset = joint.loc[
        joint["model"].astype(str).eq(model)
        & joint["decoding_group"].astype(str).eq(decoding)
    ]
    matrix = subset.pivot(index="category_a", columns="category_b", values=value)
    return matrix.reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER)


# ---------------------------------------------------------------------------
# Game-level correctness arrays - shared by 3A, 3B and 3C
# ---------------------------------------------------------------------------

def correctness_arrays(justifications: pd.DataFrame,
                       decoding: str = "Stochastic") -> Dict[str, object]:
    """Per (model, game): how often each category appeared and how often the
    vote was correct, counted over that decoding group's runs.

    Every correctness analysis below is a different reading of these same
    arrays, which is the point: three views that cannot silently disagree about
    their inputs.
    """
    games, runs, presence, correct = presence_tensor(justifications, decoding)
    n_runs = len(runs)

    n_present = presence.sum(axis=1)                       # (M, G, C)
    n_correct = correct.sum(axis=1)                        # (M, G)
    n_correct_present = np.einsum(
        "mrg,mrgc->mgc", correct, presence
    )                                                      # (M, G, C)

    return {
        "games": games,
        "runs": runs,
        "n_runs": n_runs,
        "presence": presence,
        "correct": correct,
        "n_present": n_present,
        "n_correct": n_correct,
        "n_correct_present": n_correct_present,
    }


# ---------------------------------------------------------------------------
# S7 - 3A, overall category-presence association with correctness
# ---------------------------------------------------------------------------

def correctness_presence_association(
    justifications: pd.DataFrame,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """S7: P(correct | category present) - P(correct | absent), per model.

    ASSOCIATIONAL. A positive delta says outputs that mention the category were
    more often correct, not that mentioning it helped: the same game content
    can drive both what gets mentioned and whether the vote lands.

    Uncertainty is a GAME-LEVEL CLUSTER bootstrap - a game is resampled with
    all of its runs attached, because the three stochastic realisations of one
    transcript are not three independent observations.
    """
    rows = []
    for decoding in DECODING_ORDER:
        arrays = correctness_arrays(justifications, decoding)
        n_games = len(arrays["games"])
        n_runs = arrays["n_runs"]
        weights = _game_weights(n_games, n_replicates, seed)

        n_present = arrays["n_present"]
        n_correct = arrays["n_correct"]
        n_correct_present = arrays["n_correct_present"]

        for m, model in enumerate(MODEL_ORDER):
            for c, category in enumerate(CATEGORY_ORDER):
                present = n_present[m, :, c]
                correct_present = n_correct_present[m, :, c]
                absent = n_runs - present
                correct_absent = n_correct[m] - correct_present

                total_present = float(present.sum())
                total_absent = float(absent.sum())
                p_present = (
                    correct_present.sum() / total_present
                    if total_present else np.nan
                )
                p_absent = (
                    correct_absent.sum() / total_absent
                    if total_absent else np.nan
                )

                boot_present_n = present @ weights.T
                boot_absent_n = absent @ weights.T
                with np.errstate(divide="ignore", invalid="ignore"):
                    boot_p_present = np.where(
                        boot_present_n > 0,
                        (correct_present @ weights.T) / boot_present_n, np.nan)
                    boot_p_absent = np.where(
                        boot_absent_n > 0,
                        (correct_absent @ weights.T) / boot_absent_n, np.nan)
                deltas = boot_p_present - boot_p_absent
                valid = int(np.isfinite(deltas).sum())
                if valid:
                    low, high = np.nanpercentile(deltas, [2.5, 97.5])
                else:
                    low = high = np.nan

                rows.append({
                    "decoding_group": decoding,
                    "model": model,
                    "category": category,
                    "n_outputs": int(total_present + total_absent),
                    "n_present": int(total_present),
                    "n_absent": int(total_absent),
                    "n_correct_present": int(correct_present.sum()),
                    "n_correct_absent": int(correct_absent.sum()),
                    "p_correct_present": p_present,
                    "p_correct_absent": p_absent,
                    "delta": p_present - p_absent,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_excludes_zero": bool(
                        np.isfinite(low) and np.isfinite(high)
                        and (low > 0 or high < 0)
                    ),
                    "n_valid_replicates": valid,
                    "n_games": n_games,
                    "n_replicates": n_replicates,
                    "seed": seed,
                })
    return _order(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# S8 / S9 - 3B, correctness stability across games
# ---------------------------------------------------------------------------

def correctness_stability_groups(justifications: pd.DataFrame) -> pd.DataFrame:
    """S8: how many of the 191 games each model gets right 0, 1, 2 or 3 times.

    K is defined over the three STOCHASTIC runs only. Greedy has one run and
    cannot produce a 0-3 count, so it does not appear here.
    """
    arrays = correctness_arrays(justifications, "Stochastic")
    n_correct = arrays["n_correct"]
    n_games = len(arrays["games"])

    rows = []
    for m, model in enumerate(MODEL_ORDER):
        counts = np.bincount(n_correct[m].astype(int), minlength=4)
        for k in range(4):
            rows.append({
                "model": model,
                "k_correct_runs": k,
                "label": {0: "0/3 consistently incorrect", 1: "1/3", 2: "2/3",
                          3: "3/3 consistently correct"}[k],
                "n_games": int(counts[k]),
                "pct_of_games": 100 * counts[k] / n_games,
            })
    return _order(pd.DataFrame(rows))


def correctness_stability_semantics(
    justifications: pd.DataFrame,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """S9: mean Q (share of a game's runs invoking c) within each K group.

    Descriptive. It asks whether a semantic basis is more prevalent in games a
    model solves more consistently - which conflates the model's behaviour with
    the difficulty and content of the game, and so is read as a pattern, never
    as an effect.

    All 191 games are retained. The bootstrap resamples games, so group sizes
    vary across replicates exactly as they would in a fresh sample of games.
    """
    arrays = correctness_arrays(justifications, "Stochastic")
    n_games = len(arrays["games"])
    n_runs = arrays["n_runs"]
    n_correct = arrays["n_correct"]
    q = arrays["n_present"] / n_runs                       # (M, G, C)
    weights = _game_weights(n_games, n_replicates, seed)

    rows = []
    for m, model in enumerate(MODEL_ORDER):
        k_values = n_correct[m].astype(int)
        for k in range(4):
            mask = (k_values == k).astype(float)
            group_n = int(mask.sum())
            boot_denominator = mask @ weights.T             # (B,)
            for c, category in enumerate(CATEGORY_ORDER):
                values = q[m, :, c]
                mean_q = (values * mask).sum() / group_n if group_n else np.nan

                with np.errstate(divide="ignore", invalid="ignore"):
                    boot = np.where(
                        boot_denominator > 0,
                        ((values * mask) @ weights.T) / boot_denominator,
                        np.nan,
                    )
                valid = int(np.isfinite(boot).sum())
                if valid:
                    low, high = np.nanpercentile(boot, [2.5, 97.5])
                else:
                    low = high = np.nan

                rows.append({
                    "model": model,
                    "k_correct_runs": k,
                    "category": category,
                    "n_games_in_group": group_n,
                    "mean_q": mean_q,
                    "ci_low": low,
                    "ci_high": high,
                    "n_valid_replicates": valid,
                    "n_replicates": n_replicates,
                    "seed": seed,
                })
    return _order(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# S10 - 3C, within-game correct vs incorrect contrast
# ---------------------------------------------------------------------------

def within_game_contrasts(
    justifications: pd.DataFrame,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """S10: within mixed games (K = 1 or 2), category presence in the correct
    realisations minus the incorrect ones.

    This is the strongest control available without a new experiment: model and
    transcript are held fixed, and only the sampled realisation varies. It is
    still an association between STATED semantic content and outcome - it does
    not show the content caused the outcome, and it does not show the stated
    content reflects the model's internal computation.

    The resampling unit is the mixed model-game, with its complete run set.
    """
    arrays = correctness_arrays(justifications, "Stochastic")
    n_runs = arrays["n_runs"]
    n_present = arrays["n_present"]
    n_correct = arrays["n_correct"]
    n_correct_present = arrays["n_correct_present"]

    rows = []
    for m, model in enumerate(MODEL_ORDER):
        k_values = n_correct[m]
        mixed = np.where((k_values >= 1) & (k_values <= n_runs - 1))[0]
        n_mixed = len(mixed)
        if n_mixed == 0:
            continue

        k_mixed = k_values[mixed]                          # (Gm,)
        present_mixed = n_present[m, mixed]                # (Gm, C)
        correct_present_mixed = n_correct_present[m, mixed]

        share_correct = correct_present_mixed / k_mixed[:, None]
        share_incorrect = (
            (present_mixed - correct_present_mixed) / (n_runs - k_mixed)[:, None]
        )
        delta = share_correct - share_incorrect            # (Gm, C)

        weights = _game_weights(n_mixed, n_replicates, seed)
        boot = (weights @ delta) / n_mixed                 # (B, C)

        for c, category in enumerate(CATEGORY_ORDER):
            low, high = np.percentile(boot[:, c], [2.5, 97.5])
            rows.append({
                "model": model,
                "category": category,
                "n_mixed_games": n_mixed,
                "n_games_category_in_correct_only": int(
                    ((share_correct[:, c] > 0) & (share_incorrect[:, c] == 0)).sum()
                ),
                "n_games_category_in_incorrect_only": int(
                    ((share_correct[:, c] == 0) & (share_incorrect[:, c] > 0)).sum()
                ),
                "mean_share_correct_runs": float(share_correct[:, c].mean()),
                "mean_share_incorrect_runs": float(share_incorrect[:, c].mean()),
                "delta_within": float(delta[:, c].mean()),
                "ci_low": low,
                "ci_high": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "n_replicates": n_replicates,
                "seed": seed,
            })
    return _order(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_final_tables(data: Dict[str, pd.DataFrame],
                       repo_root: Path) -> Dict[str, pd.DataFrame]:
    """Every final table, built once, in dependency order."""
    justifications = data["justifications"]
    run_level = run_level_prevalence(justifications)
    co = cooccurrence(justifications)

    tables = {
        "S0_integrity_summary": integrity_summary(data, repo_root),
        "S0b_repaired_sentences": data["repairs"],
        "S0c_multilabel_distribution": multilabel_distribution(data),
        "S1_annotation_summary": annotation_summary(data),
        "S1b_sentence_length": sentence_length_summary(justifications),
        "S2_run_level_prevalence": run_level,
        "S3_model_semantic_prevalence": model_prevalence(run_level),
        "S3b_density_sensitivity": density_sensitivity(
            justifications, data["labels"]
        ),
        "S4_prevalence_bootstrap_differences":
            prevalence_bootstrap_differences(justifications),
        "S5_cooccurrence_joint_prevalence": co["joint"],
        "S5b_cooccurrence_run_level": co["run_level"],
        "S6_cooccurrence_lift": co["lift"],
        "S6b_cooccurrence_ranked_pairs": pd.concat(
            [ranked_pairs(co["joint"], co["lift"], decoding)
             for decoding in DECODING_ORDER],
            ignore_index=True,
        ),
        "S7_correctness_presence_association":
            correctness_presence_association(justifications),
        "S8_correctness_stability_groups":
            correctness_stability_groups(justifications),
        "S9_correctness_stability_semantics":
            correctness_stability_semantics(justifications),
        "S10_within_game_correctness_contrasts":
            within_game_contrasts(justifications),
    }
    return tables


LATEX_TABLES = (
    "S1_annotation_summary",
    "S3_model_semantic_prevalence",
    "S4_prevalence_bootstrap_differences",
    "S7_correctness_presence_association",
    "S10_within_game_correctness_contrasts",
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
        if name in LATEX_TABLES:
            tex_path = directory / f"{name}.tex"
            to_latex(frame.set_index(frame.columns[0]), tex_path)
            written.append(tex_path)
    return written
