"""
Reviewing machinery for DeepSeek justification annotations.

Backs notebooks/05_annotation_val/1_justification_pilot_review.ipynb. The
logic lives here rather than in the notebook so that the same reviewer can be
pointed at the full-corpus run later without copy-pasting cells, and so the
span-highlighting can be tested.

The review model: you are checking the annotator's output, one justification
at a time. Every annotation gets a verdict; every sentence can also be marked
as having a MISSED annotation, which is the failure the review sheet format
cannot otherwise express -- a label that should be there but is not leaves no
row to disagree with.

Verdicts persist to CSV after every judgement, so the session is resumable and
a closed kernel costs nothing.
"""

import html
import json
from pathlib import Path

import pandas as pd

from src.pt_annotation.annotation_schema import (  # noqa: E402
    CATEGORY_COLORS,
    detect_schema,
    get_schema,
)

VERDICTS = [
    "",                # not yet reviewed
    "ok",
    "wrong category",
    "wrong use",       # v1 only; hidden when the schema has no `use` field
    "wrong span",
    "spurious",        # no annotation belonged here at all
]



# ============================================================
# Loading
# ============================================================

def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_pilot(pilot_dir, schema=None):
    """Return (records, {justification_id: sample item}, schema).

    The schema is read from the saved metadata rather than assumed, so
    opening pilot_v1 and pilot_v2 both work without an argument.
    """
    pilot_dir = Path(pilot_dir)
    records = read_jsonl(pilot_dir / "pilot_annotations.jsonl")
    sample = {
        item["justification_id"]: item
        for item in read_jsonl(pilot_dir / "pilot_sample.jsonl")
    }
    records = [r for r in records if not r["metadata"].get("dry_run")]
    records.sort(key=lambda r: r["metadata"]["justification_id"])

    resolved = get_schema(schema) if isinstance(schema, str) else (schema or detect_schema(records))
    return records, sample, resolved


def annotations_long(records):
    """One row per annotation, with annotation_index identifying it inside
    its sentence. That triple (justification_id, sentence_id,
    annotation_index) is the stable key the verdict file joins on."""
    rows = []
    for record in records:
        metadata = record["metadata"]
        annotation = record.get("annotation") or {}
        for sentence in annotation.get("sentences", []):
            for index, item in enumerate(sentence.get("annotations", [])):
                rows.append({
                    "justification_id": metadata["justification_id"],
                    "model": metadata.get("model_under_annotation"),
                    "is_correct": metadata.get("is_correct"),
                    "sentence_id": sentence.get("sentence_id"),
                    "annotation_index": index,
                    "text": sentence.get("text"),
                    "rule_mentioned": sentence.get("rule_mentioned"),
                    "category": item.get("category"),
                    "use": item.get("use"),          # None under v2
                    "evidence_span": item.get("evidence_span"),
                    "other_description": item.get("other_description"),
                })
    return pd.DataFrame(rows)


def sentences_long(records):
    """One row per sentence, including sentences with no annotations."""
    rows = []
    for record in records:
        metadata = record["metadata"]
        annotation = record.get("annotation") or {}
        for sentence in annotation.get("sentences", []):
            items = sentence.get("annotations", [])
            rows.append({
                "justification_id": metadata["justification_id"],
                "model": metadata.get("model_under_annotation"),
                "is_correct": metadata.get("is_correct"),
                "sentence_id": sentence.get("sentence_id"),
                "text": sentence.get("text"),
                "rule_mentioned": sentence.get("rule_mentioned"),
                "n_annotations": len(items),
                "is_labelled": len(items) > 0,
                "categories": "|".join(
                    sorted({i.get("category") for i in items if i.get("category")})
                ),
            })
    return pd.DataFrame(rows)


def usage_frame(records):
    """Token usage per call -- the basis for projecting the full-corpus run."""
    rows = []
    for record in records:
        usage = (record["metadata"].get("usage") or {})
        rows.append({
            "justification_id": record["metadata"]["justification_id"],
            "n_sentences": record["metadata"].get("n_sentences"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "finish_reason": record["metadata"].get("finish_reason"),
        })
    return pd.DataFrame(rows)


# ============================================================
# Rendering
# ============================================================

def highlight_spans(text, spans):
    """Wrap each evidence span in a coloured mark.

    Spans are located by first occurrence and overlaps are dropped rather than
    nested, so the output stays valid HTML. A span that cannot be found is
    reported by the caller instead of silently vanishing -- an unlocatable
    span is exactly the failure worth seeing.
    """
    placements = []
    for span, category in spans:
        if not span:
            continue
        start = text.find(span)
        if start == -1:
            continue
        placements.append((start, start + len(span), category))

    placements.sort(key=lambda p: (p[0], -(p[1] - p[0])))

    kept = []
    last_end = 0
    for start, end, category in placements:
        if start >= last_end:
            kept.append((start, end, category))
            last_end = end

    out = []
    cursor = 0
    for start, end, category in kept:
        out.append(html.escape(text[cursor:start]))
        colour = CATEGORY_COLORS.get(category, "#e2e2e2")
        out.append(
            f'<mark style="background:{colour};color:#111;padding:1px 2px;'
            f'border-radius:3px">{html.escape(text[start:end])}</mark>'
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out)


def category_chip(category, use=None):
    """The use suffix is omitted when there is no use -- v2 dropped the field,
    and rendering a literal "None" beside every category would be noise."""
    colour = CATEGORY_COLORS.get(category, "#e2e2e2")
    chip = (
        f'<span style="background:{colour};color:#111;padding:2px 8px;'
        f'border-radius:10px;font-size:0.85em;font-weight:600">'
        f'{html.escape(str(category))}</span>'
    )
    if use:
        chip += (
            f'<span style="color:#666;font-size:0.85em"> &middot; '
            f'{html.escape(str(use))}</span>'
        )
    return chip


def render_justification(record, sample_item=None):
    """Full justification as HTML: header, then each sentence with its
    evidence spans highlighted and its annotations listed beneath."""
    metadata = record["metadata"]
    annotation = record.get("annotation") or {}

    correct = metadata.get("is_correct")
    correct_label = "correct vote" if correct else "incorrect vote"

    parts = [
        '<div style="font-family:system-ui,sans-serif;line-height:1.5">',
        '<div style="color:#666;font-size:0.85em;margin-bottom:2px">'
        f'{html.escape(str(metadata.get("model")or metadata.get("model_under_annotation")))} '
        f'&middot; {html.escape(str(metadata.get("run_label")))} '
        f'&middot; {correct_label}</div>',
        f'<div style="font-size:1.05em;margin-bottom:10px">'
        f'<b>Vote:</b> {html.escape(str(annotation.get("vote")))}</div>',
    ]

    for sentence in annotation.get("sentences", []):
        text = sentence.get("text", "")
        items = sentence.get("annotations", [])
        spans = [(i.get("evidence_span"), i.get("category")) for i in items]

        rule_badge = (
            '<span style="background:#333;color:#fff;padding:1px 6px;'
            'border-radius:8px;font-size:0.72em">rule</span> '
            if sentence.get("rule_mentioned") else ""
        )

        border = "#bbb" if items else "#eee"
        parts.append(
            f'<div style="border-left:3px solid {border};padding:4px 0 4px 10px;'
            f'margin-bottom:8px">'
            f'<div style="color:#888;font-size:0.75em">'
            f'{rule_badge}sentence {sentence.get("sentence_id")}</div>'
            f'<div style="margin:3px 0">{highlight_spans(text, spans)}</div>'
        )

        if not items:
            parts.append(
                '<div style="color:#999;font-size:0.85em;font-style:italic">'
                'no annotation</div>'
            )

        for item in items:
            span = item.get("evidence_span") or ""
            missing = span and span not in text
            warn = (
                ' <span style="color:#b00;font-weight:600">[span not found in '
                'sentence]</span>' if missing else ""
            )
            description = item.get("other_description")
            description_html = (
                f'<div style="color:#666;font-size:0.8em;margin-left:6px">'
                f'{html.escape(str(description))}</div>' if description else ""
            )
            parts.append(
                f'<div style="margin:2px 0 2px 6px;font-size:0.9em">'
                f'{category_chip(item.get("category"), item.get("use"))}'
                f'<span style="color:#444"> &ldquo;{html.escape(span)}&rdquo;</span>'
                f'{warn}</div>{description_html}'
            )

        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


# ============================================================
# Verdict persistence
# ============================================================

VERDICT_COLUMNS = [
    "justification_id", "sentence_id", "annotation_index",
    "category", "use", "evidence_span",
    "verdict", "corrected_category", "corrected_use", "note",
]

MISSED_COLUMNS = [
    "justification_id", "sentence_id", "missed_category", "missed_note",
]


class VerdictStore:
    """Two CSVs: one verdict per annotation, plus any missed annotations.

    Kept separate because they have different shapes -- a missed annotation
    has no model output to attach a verdict to. Both are written on every
    change, so an interrupted session loses nothing.
    """

    def __init__(self, verdict_path, missed_path):
        self.verdict_path = Path(verdict_path)
        self.missed_path = Path(missed_path)
        self.verdicts = self._load(self.verdict_path, VERDICT_COLUMNS)
        self.missed = self._load(self.missed_path, MISSED_COLUMNS)

    @staticmethod
    def _load(path, columns):
        if Path(path).exists():
            frame = pd.read_csv(path, dtype={"note": str, "missed_note": str})
            for column in columns:
                if column not in frame.columns:
                    frame[column] = ""
            return frame[columns].fillna("")
        return pd.DataFrame(columns=columns)

    def key_mask(self, frame, justification_id, sentence_id, annotation_index=None):
        mask = (
            frame["justification_id"].eq(justification_id)
            & frame["sentence_id"].astype(int).eq(int(sentence_id))
        )
        if annotation_index is not None:
            mask &= frame["annotation_index"].astype(int).eq(int(annotation_index))
        return mask

    def get(self, justification_id, sentence_id, annotation_index):
        if self.verdicts.empty:
            return {}
        hit = self.verdicts[
            self.key_mask(self.verdicts, justification_id, sentence_id, annotation_index)
        ]
        return {} if hit.empty else hit.iloc[0].to_dict()

    def set(self, row):
        mask = self.key_mask(
            self.verdicts, row["justification_id"],
            row["sentence_id"], row["annotation_index"],
        ) if not self.verdicts.empty else None

        if mask is not None and mask.any():
            self.verdicts = self.verdicts[~mask]

        self.verdicts = pd.concat(
            [self.verdicts, pd.DataFrame([row])[VERDICT_COLUMNS]],
            ignore_index=True,
        )
        self.save()

    def set_missed(self, justification_id, sentence_id, missed_category, missed_note):
        if not self.missed.empty:
            mask = self.key_mask(self.missed, justification_id, sentence_id)
            self.missed = self.missed[~mask]

        if missed_category or missed_note:
            row = {
                "justification_id": justification_id,
                "sentence_id": sentence_id,
                "missed_category": missed_category,
                "missed_note": missed_note,
            }
            self.missed = pd.concat(
                [self.missed, pd.DataFrame([row])[MISSED_COLUMNS]],
                ignore_index=True,
            )
        self.save()

    def get_missed(self, justification_id, sentence_id):
        if self.missed.empty:
            return {}
        hit = self.missed[self.key_mask(self.missed, justification_id, sentence_id)]
        return {} if hit.empty else hit.iloc[0].to_dict()

    def save(self):
        self.verdict_path.parent.mkdir(parents=True, exist_ok=True)
        self.verdicts.sort_values(
            ["justification_id", "sentence_id", "annotation_index"]
        ).to_csv(self.verdict_path, index=False, encoding="utf-8")
        self.missed.sort_values(
            ["justification_id", "sentence_id"]
        ).to_csv(self.missed_path, index=False, encoding="utf-8")

    def reviewed_ids(self):
        """Justifications with a verdict on every one of their annotations."""
        if self.verdicts.empty:
            return set()
        judged = self.verdicts[self.verdicts["verdict"].ne("")]
        return set(judged["justification_id"].unique())

    def progress(self, total_annotations):
        judged = 0 if self.verdicts.empty else int(
            self.verdicts["verdict"].ne("").sum()
        )
        return judged, total_annotations


# ============================================================
# Scoring
# ============================================================

def score(verdicts, annotations, schema):
    """Agreement rate overall and per category, over reviewed annotations only.

    This is an error rate against your adjudication of the model's output, not
    a chance-corrected coefficient: you reviewed the model's labels rather
    than coding blind, so the two label sets are not independent.
    """
    judged = verdicts[verdicts["verdict"].ne("")].copy()
    if judged.empty:
        return pd.DataFrame(), {}

    judged["accepted"] = judged["verdict"].eq("ok")

    per_category = (
        judged
        .groupby("category")
        .agg(n_reviewed=("verdict", "size"), n_accepted=("accepted", "sum"))
        .reindex(list(schema.categories))
        .fillna(0)
        .astype(int)
    )
    per_category["accept_rate"] = (
        100 * per_category["n_accepted"] / per_category["n_reviewed"].replace(0, pd.NA)
    )

    summary = {
        "n_reviewed": int(len(judged)),
        "n_total": int(len(annotations)),
        "accept_rate": 100 * judged["accepted"].mean(),
        "verdict_counts": judged["verdict"].value_counts().to_dict(),
    }
    return per_category, summary
