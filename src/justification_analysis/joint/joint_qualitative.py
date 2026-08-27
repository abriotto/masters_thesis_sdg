"""Qualitative sampling behind two joint discourse x semantic pairings.

Read-only inspection support. Nothing here computes a statistic, changes a
label, or feeds back into the quantitative tables - it draws representative
sentences so a human can see what a detected pairing actually looks like.

Two pairings are sampled, both stochastic decoding only:

  A. E2B  sentences labelled Payoff    whose sentence carries Contingency.Cause
  B. 31B  sentences labelled Mechanical whose sentence carries Contingency.Condition

Sampling is seeded and spread across games - at most one sentence per game
while enough eligible games remain - so no single transcript can dominate and
nothing is hand-picked.

INTERPRETATION LIMIT, restated because these examples invite over-reading:
the parser detects a connective and its PDTB sense somewhere in the sentence.
It does NOT tell us that the connective attaches to the semantically labelled
material, nor that the evidence span is Arg1 or Arg2 of the relation. The
evidence spans are printed for human reading only and carry no quantitative
weight.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.justification_analysis.joint import joint_final as jf

SAMPLE_SEED = 20260826
SAMPLE_SIZE = 20

PAIRINGS = {
    "A_E2B_Payoff_x_Contingency.Cause": {
        "model": "Gemma 4 2B",
        "category": "Payoff",
        "sense": "Contingency.Cause",
    },
    "B_31B_Mechanical_x_Contingency.Condition": {
        "model": "Gemma 4 31B",
        "category": "Mechanical",
        "sense": "Contingency.Condition",
    },
}

QUALITATIVE_SUBPATH = jf.ARTIFACT_SUBPATH / "qualitative"


def load_justification_texts(repo_root: Path) -> pd.DataFrame:
    """The full justification text and vote, from the annotator's input shards.

    Read from the input rather than reassembled from sentences: the shard is
    what was actually sent, so the printed justification is the real one and
    not a join artefact.
    """
    rows = []
    directory = Path(repo_root) / jf.sem.INPUT_SUBPATH
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
                    "is_correct": record["is_correct"],
                })
    return pd.DataFrame(rows)


def eligible_sentences(joint: pd.DataFrame, model: str, category: str,
                       sense: str) -> pd.DataFrame:
    """Stochastic sentences carrying both the category and the sense."""
    return joint.loc[
        joint["decoding_group"].astype(str).eq("Stochastic")
        & joint["model"].astype(str).eq(model)
        & joint[f"sem_{category}"]
        & joint[f"sense_{sense}"]
    ].copy()


def draw_sample(eligible: pd.DataFrame, n: int = SAMPLE_SIZE,
                seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """At most one sentence per game while enough games remain.

    Games are shuffled first and one eligible sentence is drawn from each. Only
    if the eligible games run out does a second sentence from an already-used
    game get taken - and the fallback draw is reported by `games_exhausted`.
    """
    rng = np.random.default_rng(seed)
    games = sorted(eligible["game_id"].unique())
    order = rng.permutation(len(games))

    picked_index: List[int] = []
    for position in order:
        game = games[position]
        candidates = eligible.loc[eligible["game_id"].eq(game)]
        choice = rng.integers(len(candidates))
        picked_index.append(candidates.index[choice])
        if len(picked_index) == n:
            break

    games_exhausted = len(picked_index) < n
    if games_exhausted:
        remaining = eligible.drop(index=picked_index)
        extra = rng.permutation(len(remaining))[: n - len(picked_index)]
        picked_index.extend(remaining.index[extra])

    sample = eligible.loc[picked_index].copy()
    sample.attrs["games_exhausted"] = games_exhausted
    return sample


def annotate_sample(sample: pd.DataFrame, layers: Dict[str, pd.DataFrame],
                    alignment: Dict[str, pd.DataFrame],
                    texts: pd.DataFrame) -> pd.DataFrame:
    """Attach everything a human needs to read the example."""
    labels = layers["semantic"]["labels"]
    aligned = alignment["aligned"]

    rows = []
    for row in sample.itertuples():
        key = (row.model, row.game_id, row.run_label, row.sentence_id)

        sentence_labels = labels.loc[
            labels["justification_id"].eq(row.justification_id)
            & labels["sentence_id"].eq(row.sentence_id)
        ]
        sentence_relations = aligned.loc[
            aligned["model"].astype(str).eq(str(row.model))
            & aligned["game_id"].eq(row.game_id)
            & aligned["run_label"].eq(row.run_label)
            & aligned["sentence_id"].eq(row.sentence_id)
        ]
        text_row = texts.loc[
            texts["justification_id"].eq(row.justification_id)].iloc[0]

        rows.append({
            "model": str(row.model),
            "game_id": row.game_id,
            "run_label": row.run_label,
            "sentence_id": row.sentence_id,
            "target_sentence": row.text,
            "justification_text": text_row["justification_text"],
            "vote": text_row["vote"],
            "voted_player_end_role": text_row["voted_player_end_role"],
            "is_correct": bool(text_row["is_correct"]),
            "semantic_categories": " | ".join(
                sentence_labels["category"].astype(str)),
            "evidence_spans": " | ".join(
                str(s) for s in sentence_labels["evidence_span"]),
            "discourse_senses": " | ".join(
                sentence_relations["raw_sense"].astype(str)),
            "connective_surfaces": " | ".join(
                sentence_relations["connective_surface"].astype(str)),
            "n_relations_in_sentence": len(sentence_relations),
        })
    frame = pd.DataFrame(rows)
    frame.attrs.update(sample.attrs)
    return frame


def build_samples(joint: pd.DataFrame, layers: Dict[str, pd.DataFrame],
                  alignment: Dict[str, pd.DataFrame],
                  repo_root: Path) -> Dict[str, pd.DataFrame]:
    texts = load_justification_texts(repo_root)
    samples = {}
    for name, spec in PAIRINGS.items():
        eligible = eligible_sentences(
            joint, spec["model"], spec["category"], spec["sense"])
        sample = draw_sample(eligible)
        annotated = annotate_sample(sample, layers, alignment, texts)
        annotated.attrs["n_eligible"] = len(eligible)
        annotated.attrs["n_eligible_games"] = eligible["game_id"].nunique()
        annotated.attrs["run_distribution"] = (
            eligible["run_label"].value_counts().to_dict())
        samples[name] = annotated
    return samples


def format_example(row, index: int) -> str:
    """One example as a readable block."""
    lines = [
        f"--- {index:02d} --- {row.model} | {row.game_id} | {row.run_label} "
        f"| sentence {row.sentence_id}",
        f"  TARGET     : {row.target_sentence}",
        f"  categories : {row.semantic_categories}",
        f"  spans      : {row.evidence_spans}",
        f"  senses     : {row.discourse_senses}",
        f"  connectives: {row.connective_surfaces}",
        f"  vote       : {row.vote} ({row.voted_player_end_role}, "
        f"{'correct' if row.is_correct else 'incorrect'})",
        f"  FULL       : {row.justification_text}",
    ]
    return "\n".join(lines)


def write_samples(samples: Dict[str, pd.DataFrame],
                  directory: Path) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in samples.items():
        path = directory / f"Q_{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)
    return written
