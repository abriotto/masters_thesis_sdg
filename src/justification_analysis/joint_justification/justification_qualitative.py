"""Qualitative inspection driven by the JUSTIFICATION-level analysis.

Replaces an earlier sample that was selected from same-sentence pairings.
That selection turned out to be a packaging artefact for at least one of the
two patterns, so its examples could not characterise a justification-level
association; it has been deleted along with the rest of the sentence-level
strand.

Selection here is at the justification level: a justification qualifies if it
contains the semantic category anywhere and the discourse relation anywhere.
Whether they share a sentence is then REPORTED, not required - that is the
localization diagnostic, and it is the thing being inspected rather than a
precondition for being inspected.

No new annotation. The parser still does not recover discourse arguments, and
nothing here treats a connective as attaching to a semantic span.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.justification_analysis.joint_justification import justification_joint as kj

SAMPLE_SEED = 20260826
SAMPLE_SIZE = 20

PATTERNS = {
    "A_E2B_Payoff_x_Contingency": {
        "model": "Gemma 4 2B",
        "category": "Payoff",
        "relation_column": "disc_Contingency",
        "sentence_column": "disc_Contingency",
        "label": "E2B  Payoff x Contingency (top level)",
    },
    "B_31B_Mechanical_x_Contingency.Condition": {
        "model": "Gemma 4 31B",
        "category": "Mechanical",
        "relation_column": "sense_Contingency.Condition",
        "sentence_column": "sense_Contingency.Condition",
        "label": "31B  Mechanical x Contingency.Condition",
    },
}


def eligible(justifications: pd.DataFrame, model: str, category: str,
             relation_column: str) -> pd.DataFrame:
    return justifications.loc[
        justifications["decoding_group"].astype(str).eq("Stochastic")
        & justifications["model"].astype(str).eq(model)
        & justifications[f"sem_{category}"]
        & justifications[relation_column]
    ].copy()


def draw_sample(frame: pd.DataFrame, n: int = SAMPLE_SIZE,
                seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """One justification per game, games shuffled under the fixed seed."""
    rng = np.random.default_rng(seed)
    games = sorted(frame["game_id"].unique())
    order = rng.permutation(len(games))
    picked = []
    for position in order:
        candidates = frame.loc[frame["game_id"].eq(games[position])]
        picked.append(candidates.index[rng.integers(len(candidates))])
        if len(picked) == n:
            break
    return frame.loc[picked].copy()


def describe_example(row, sentences_joint: pd.DataFrame,
                     aligned: pd.DataFrame, category: str,
                     sentence_column: str) -> Dict:
    """Where the category and the relation sit, and how far apart."""
    sentences = sentences_joint.loc[
        sentences_joint["justification_id"].eq(row.justification_id)
    ].sort_values("sentence_id")

    category_sentences = sentences.loc[
        sentences[f"sem_{category}"], "sentence_id"].tolist()
    relation_sentences = sentences.loc[
        sentences[sentence_column], "sentence_id"].tolist()

    same = sorted(set(category_sentences) & set(relation_sentences))
    if same:
        arrangement = "same sentence"
        distance = 0
    else:
        distance = min(abs(a - b) for a in category_sentences
                       for b in relation_sentences)
        arrangement = "adjacent sentences" if distance == 1 else "separate sentences"

    relations = aligned.loc[
        aligned["justification_id"].eq(row.justification_id)]
    labels = sentences[["sentence_id"] + [f"sem_{c}" for c in kj.CATEGORY_ORDER]]
    per_sentence = []
    for record in labels.itertuples(index=False):
        present = [c for c in kj.CATEGORY_ORDER
                   if getattr(record, f"sem_{c}")]
        if present:
            per_sentence.append(f"s{record.sentence_id}:{'+'.join(present)}")

    return {
        "model": str(row.model),
        "game_id": row.game_id,
        "run_label": row.run_label,
        "justification_text": None,      # filled by the caller
        "vote": row.vote,
        "is_correct": bool(row.is_correct),
        "n_sentences": int(row.n_sentences),
        "category_sentences": ",".join(str(s) for s in category_sentences),
        "relation_sentences": ",".join(str(s) for s in relation_sentences),
        "arrangement": arrangement,
        "sentence_distance": distance,
        "semantic_labels_by_sentence": " | ".join(per_sentence),
        "connectives": " | ".join(
            f"{s}({r})" for s, r in zip(
                relations["connective_surface"].astype(str),
                relations["raw_sense"].astype(str))),
    }


def build_samples(justifications: pd.DataFrame,
                  sentences_joint: pd.DataFrame,
                  alignment: Dict[str, pd.DataFrame],
                  metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    aligned = alignment["aligned"].rename(
        columns={"justification_id": "discourse_row_id",
                 "justification_id_canonical": "justification_id"})
    texts = metadata.set_index("justification_id")["justification_text"]

    samples = {}
    for name, spec in PATTERNS.items():
        pool = eligible(justifications, spec["model"], spec["category"],
                        spec["relation_column"])
        sample = draw_sample(pool)
        rows = []
        for row in sample.itertuples():
            described = describe_example(
                row, sentences_joint, aligned, spec["category"],
                spec["sentence_column"])
            described["justification_text"] = texts.loc[row.justification_id]
            rows.append(described)
        frame = pd.DataFrame(rows)
        frame.attrs["label"] = spec["label"]
        frame.attrs["n_eligible"] = len(pool)
        frame.attrs["n_eligible_games"] = pool["game_id"].nunique()
        frame.attrs["run_distribution"] = pool["run_label"].value_counts().to_dict()
        samples[name] = frame
    return samples


# ---------------------------------------------------------------------------
# Lexical diagnostics over the COMPLETE local-realization set
# ---------------------------------------------------------------------------

DIAGNOSTICS = {
    "opens with 'Since'": r"^since",
    "opens with 'If'": r"^if",
    "'Team Village wins/needs'": r"team village (wins|needs)",
    "'eliminate a werewolf'": r"eliminat\w* a werewolf",
    "evaluative adjective": r"\b(safest|optimal|best|logical|necessary|prime|primary)\b",
    "card-movement verb": r"(swap|switch|rob|steal|transferr?ed|moved)",
    "'center'": r"\bcenter\b",
    "'card'": r"\bcard\b",
    "modal/state consequent": r"(would|must|remains|is now)",
    "claim-conditional wording": r"(claim|says|stated|told|information|true|accurate)",
}


def local_realization_set(sentences_joint: pd.DataFrame, model: str,
                          category: str, sentence_column: str) -> pd.DataFrame:
    """Every stochastic sentence where the category and the relation are
    locally co-present - the complete set, not a sample."""
    return sentences_joint.loc[
        sentences_joint["decoding_group"].astype(str).eq("Stochastic")
        & sentences_joint["model"].astype(str).eq(model)
        & sentences_joint[f"sem_{category}"]
        & sentences_joint[sentence_column]
    ].copy()


def lexical_diagnostics(sentences: pd.DataFrame) -> pd.DataFrame:
    """Transparent surface counts. No judgement, just proportions."""
    text = sentences["text"].astype(str).str.lower()
    n = len(text)
    rows = [{"diagnostic": name,
             "n": int(text.str.contains(pattern, regex=True).sum()),
             "pct": 100 * float(text.str.contains(pattern, regex=True).mean())}
            for name, pattern in DIAGNOSTICS.items()]
    frame = pd.DataFrame(rows)
    frame.attrs["n_sentences"] = n
    return frame


def opening_repetition(sentences: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Distinct opening k-grams - the direct measure of formulaicity."""
    opening = (sentences["text"].astype(str).str.lower()
               .str.replace(r"[^a-z ]", "", regex=True)
               .str.split().str[:k].str.join(" "))
    counts = opening.value_counts().rename("n").to_frame().reset_index()
    counts.columns = ["opening", "n"]
    counts["pct"] = 100 * counts["n"] / len(sentences)
    counts.attrs["n_distinct"] = int(opening.nunique())
    counts.attrs["n_sentences"] = len(sentences)
    return counts


def write_samples(samples: Dict[str, pd.DataFrame],
                  directory: Path) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in samples.items():
        path = directory / f"Q2_{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)
    return written
