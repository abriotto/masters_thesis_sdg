"""The ONE canonical corpus loader, and the fingerprint that identifies it.

Before this module the same load existed twice - once in
`discourse_statistics.load_justification_frame` and once in
`run_discopy_on_justifications.load_justifications`, the second carrying the
comment "Reproduce the notebook's loading logic exactly, including id
assignment". Two implementations of the same load is precisely how a parser
artifact silently stops matching the corpus it is used against.

Both now call `load_corpus`.

## On `justification_id`

It is a bare positional index over the concatenation of the per-model tables,
in configured model order. That is what the frozen base artifact was built
with, so it is preserved exactly - but it is POSITIONAL, not intrinsic:
insert one row anywhere and every id after it shifts.

The fingerprint therefore deliberately ignores it. Identity comes from
(model, game_id, run_label, decoding, justification text), sorted into a
canonical order first, so the hash is invariant to row order and to the id
assignment while still changing the moment any justification text changes.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.justification_analysis.pipeline.config import AnalysisConfig
from src.utils.sentences import split_sentences

# The token pattern that defines the word denominator for every density in the
# thesis. NOTE: this is a TOKENIZER and is deliberately not the same thing as
# `src.utils.sentences.WORD_PATTERN`, which is a single-character predicate
# used to decide whether a segmentation chunk contains any word character at
# all. They share a name for historical reasons and do different jobs; merging
# them would silently change the 169,748-token denominator and break every
# frozen density. If either is ever renamed, rename the segmentation one.
WORD_PATTERN = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", flags=re.UNICODE)

# Fields that define corpus identity. Deliberately excludes justification_id.
FINGERPRINT_FIELDS = ("model", "game_id", "run_label", "decoding",
                      "justification")
FINGERPRINT_VERSION = "1"


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(str(text)))


def load_corpus(config: AnalysisConfig) -> pd.DataFrame:
    """One row per justification for the CONFIGURED stage.

    Fails loudly if the stage has no inputs - it never falls back to another
    stage. Row order and `justification_id` reproduce the frozen behaviour:
    per-model tables concatenated in configured model order, then a global
    arange.
    """
    config.require_inputs()

    frames = []
    for model, path in config.vote_tables():
        frame = pd.read_csv(path)
        frame["model"] = model.display
        frames.append(frame)

    votes = pd.concat(frames, ignore_index=True)
    votes["run_number"] = (
        votes["run_label"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
    )
    votes["decoding_group"] = np.where(
        votes["decoding"].astype(str).str.lower().eq("stochastic"),
        "Stochastic", "Greedy",
    )
    votes["justification_id"] = np.arange(len(votes))
    votes["justification"] = votes["justification"].fillna("").astype(str)
    votes["n_words"] = votes["justification"].map(count_words)
    votes["n_sentences"] = votes["justification"].map(
        lambda text: len(split_sentences(text)))
    votes.attrs["stage"] = config.stage
    votes.attrs["prompt_version"] = config.prompt_version
    return votes


def canonical_records(corpus: pd.DataFrame) -> pd.DataFrame:
    """The identity-bearing fields, in a deterministic order."""
    missing = [c for c in FINGERPRINT_FIELDS if c not in corpus.columns]
    if missing:
        raise KeyError(f"corpus is missing fingerprint fields: {missing}")
    records = corpus[list(FINGERPRINT_FIELDS)].copy()
    for column in FINGERPRINT_FIELDS:
        records[column] = records[column].astype(str)
    return records.sort_values(list(FINGERPRINT_FIELDS)).reset_index(drop=True)


def corpus_fingerprint(corpus: pd.DataFrame) -> str:
    """SHA-256 over the canonical records.

    Order-invariant and id-invariant; changes if any justification text,
    model, game, run or decoding label changes.
    """
    records = canonical_records(corpus)
    digest = hashlib.sha256()
    digest.update(f"v{FINGERPRINT_VERSION}\n".encode("utf-8"))
    digest.update(("\x1f".join(FINGERPRINT_FIELDS) + "\x1e").encode("utf-8"))
    for row in records.itertuples(index=False):
        digest.update(("\x1f".join(row) + "\x1e").encode("utf-8"))
    return digest.hexdigest()


def corpus_summary(corpus: pd.DataFrame, config: AnalysisConfig) -> Dict:
    """Everything a manifest needs to say about the corpus it came from."""
    by_model_run = (corpus.groupby(["model", "run_label"], observed=True)
                    .size().to_dict())
    return {
        "fingerprint": corpus_fingerprint(corpus),
        "fingerprint_version": FINGERPRINT_VERSION,
        "fingerprint_fields": list(FINGERPRINT_FIELDS),
        "stage": config.stage,
        "prompt_version": config.prompt_version,
        "n_justifications": int(len(corpus)),
        "n_games": int(corpus["game_id"].nunique()),
        "n_sentences": int(corpus["n_sentences"].sum()),
        "n_words": int(corpus["n_words"].sum()),
        "models": sorted(corpus["model"].astype(str).unique()),
        "runs": sorted(corpus["run_label"].astype(str).unique()),
        "run_structure": {f"{model} / {run}": int(n)
                          for (model, run), n in sorted(by_model_run.items())},
    }


def integrity_checks(corpus: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Structural checks derived from the ACTIVE input, not from frozen counts.

    Nothing here asserts 2,292 or 169,748. Those are properties of the base
    corpus and live in the base regression test; a fine-tuned corpus is a
    different size and must still pass these.
    """
    per_model_run = corpus.groupby(["model", "run_label"], observed=True).size()
    games_per_model_run = (corpus.groupby(["model", "run_label"], observed=True)
                           ["game_id"].nunique())
    expected_models = set(config.model_order)
    expected_runs = set(config.all_runs)

    checks = [
        ("every configured model is present",
         set(corpus["model"].astype(str)) == expected_models,
         f"{sorted(set(corpus['model'].astype(str)))}"),
        ("every configured run is present",
         set(corpus["run_label"].astype(str)) == expected_runs,
         f"{sorted(set(corpus['run_label'].astype(str)))}"),
        ("every model x run has the same number of justifications",
         per_model_run.nunique() == 1,
         f"{sorted(per_model_run.unique().tolist())}"),
        ("every model x run covers the same game set",
         games_per_model_run.nunique() == 1,
         f"{sorted(games_per_model_run.unique().tolist())}"),
        ("justification ids are unique",
         int(corpus["justification_id"].duplicated().sum()) == 0, ""),
        ("model x game x run identifies a justification uniquely",
         int(corpus.duplicated(["model", "game_id", "run_label"]).sum()) == 0, ""),
        ("no empty justification text",
         int((corpus["justification"].str.strip() == "").sum()) == 0, ""),
        ("word counts are positive",
         bool((corpus["n_words"] > 0).all()), ""),
        ("sentence counts are positive",
         bool((corpus["n_sentences"] > 0).all()), ""),
        (f"decoding groups are exactly {', '.join(config.decoding_groups)}",
         set(corpus["decoding_group"]) == set(config.decoding_groups),
         f"{sorted(set(corpus['decoding_group']))}"),
    ]
    return pd.DataFrame([
        {"check": name, "passed": bool(passed), "observed": observed,
         "status": "OK" if passed else "FAIL"}
        for name, passed, observed in checks
    ])


def sentence_frame(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per (justification, sentence), using the project segmentation.

    The single place a sentence table is built from the corpus, so the parser
    runner and the analyses cannot disagree about sentence ids.
    """
    rows = []
    for row in corpus.itertuples(index=False):
        for index, text in enumerate(split_sentences(row.justification), start=1):
            rows.append({
                "justification_id": row.justification_id,
                "model": row.model,
                "game_id": row.game_id,
                "run_label": row.run_label,
                "decoding_group": row.decoding_group,
                "sentence_id": index,
                "sentence_text": text,
            })
    return pd.DataFrame(rows)
