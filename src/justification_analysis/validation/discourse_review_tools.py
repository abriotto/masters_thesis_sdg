"""Click-through reviewer for the 50-item discourse validation sample.

Follows the same shape as `src/pt_annotation/justification_review_widget.py`:
one case on screen at a time, every control writing straight through to a store
that saves on each change. There is no submit step to forget, and closing the
kernel mid-review loses nothing.

The sample itself is read-only here. Nothing in this module changes a parser
prediction, the sampling, or `manual_validation_sample_50.csv`; manual answers
go to a separate completed file.

`COMPLETED_COLUMNS` below is the canonical schema for a completed sheet;
`evaluate_manual_validation.py` reads exactly these columns.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Canonical schema for a completed review sheet. Consumed by
# evaluate_manual_validation.py, which accepts exactly these columns.
COMPLETED_COLUMNS = [
    "validation_id", "failure_type", "model", "game_id", "run_label",
    "decoding_group", "justification_id", "sentence_id", "marker",
    "discopy_raw_sense", "discopy_top_level", "discopy_confidence",
    "dimlex_category", "sentence_text",
    "manual_is_connective", "manual_top_level_category",
    "manual_valid_relation_missed_by_discopy", "notes",
]

MANUAL_COLUMNS = [
    "manual_is_connective",
    "manual_top_level_category",
    "manual_valid_relation_missed_by_discopy",
    "notes",
]

TOP_LEVEL_CATEGORIES = ["Comparison", "Contingency", "Expansion", "Temporal"]

ACCEPTED = "accepted"


# ---------------------------------------------------------------------------
# Case preparation
# ---------------------------------------------------------------------------

def build_cases(sample: pd.DataFrame, justifications: pd.DataFrame,
                split_sentences) -> List[dict]:
    """Attach context sentences and an exact-offset highlight to each case.

    Highlighting uses the character spans recorded in the sample, not a string
    search, so the marker shown is the span the parser actually scored - the
    same basis the HTML tool uses.
    """
    text_by_id = dict(
        zip(justifications["justification_id"], justifications["justification"])
    )

    cases = []
    for row in sample.itertuples(index=False):
        text = text_by_id.get(row.justification_id, "")
        sentences = split_sentences(text)
        index = int(row.sentence_id) - 1 if pd.notna(row.sentence_id) else -1
        sentence = (
            sentences[index] if 0 <= index < len(sentences)
            else str(row.sentence_text)
        )

        sentence_start = text.find(sentence)
        spans = []
        if isinstance(row.char_spans, str) and row.char_spans and sentence_start >= 0:
            for chunk in row.char_spans.split(";"):
                begin, _, end = chunk.partition("-")
                try:
                    spans.append((int(begin) - sentence_start, int(end) - sentence_start))
                except ValueError:
                    pass
        spans = [(a, b) for a, b in spans if 0 <= a < b <= len(sentence)]

        case = {column: getattr(row, column) for column in sample.columns}
        case["sentence_text"] = sentence
        case["highlight_spans"] = spans
        case["prev_sentence"] = sentences[index - 1] if index > 0 else ""
        case["next_sentence"] = (
            sentences[index + 1] if 0 <= index < len(sentences) - 1 else ""
        )
        cases.append(case)
    return cases


def render_case_html(case: dict, position: int, total: int) -> str:
    """Self-contained HTML for one case, used by the widget and standalone."""
    sentence = case["sentence_text"]
    spans = case["highlight_spans"]

    if spans:
        pieces, cursor = [], 0
        for begin, end in sorted(spans):
            if begin < cursor:
                continue
            pieces.append(html.escape(sentence[cursor:begin]))
            pieces.append(
                f'<mark style="background:#fde68a;padding:1px 3px;'
                f'border-radius:3px;font-weight:650">'
                f'{html.escape(sentence[begin:end])}</mark>'
            )
            cursor = end
        pieces.append(html.escape(sentence[cursor:]))
        sentence_html = "".join(pieces)
    else:
        sentence_html = html.escape(sentence)

    colours = {
        "accepted": "#2563eb",
        "rejected_nosense": "#b45309",
        "not_enumerated": "#7c3aed",
    }
    colour = colours.get(case["failure_type"], "#555")

    def value(name, default="-"):
        raw = case.get(name)
        return default if pd.isna(raw) or raw == "" else raw

    if case["failure_type"] == ACCEPTED:
        confidence = case.get("discopy_confidence")
        prediction = (
            f'<span>discopy sense <b>{value("discopy_raw_sense")}</b></span>'
            f'<span style="margin-left:18px">top level '
            f'<b>{value("discopy_top_level")}</b></span>'
            f'<span style="margin-left:18px">confidence '
            f'<b>{"-" if pd.isna(confidence) else f"{confidence:.3f}"}</b></span>'
        )
    else:
        seen = ("enumerated, classified NoSense"
                if case["failure_type"] == "rejected_nosense"
                else "never enumerated as a candidate")
        prediction = (
            f'<span>DiMLex category <b>{value("dimlex_category")}</b></span>'
            f'<span style="margin-left:18px">discopy <b>{seen}</b></span>'
        )

    context_style = "color:#6b6b76;font-size:13.5px;margin:4px 0"
    prev_html = (f'<div style="{context_style}">&#8230; '
                 f'{html.escape(case["prev_sentence"])}</div>'
                 if case["prev_sentence"] else "")
    next_html = (f'<div style="{context_style}">'
                 f'{html.escape(case["next_sentence"])} &#8230;</div>'
                 if case["next_sentence"] else "")

    return f"""
    <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;
                border:1px solid #e2e2e8;border-radius:10px;padding:14px 16px">
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <span style="font-weight:700">case {position} / {total}</span>
        <span style="background:{colour};color:#fff;font-size:12px;
                     font-weight:650;padding:3px 10px;border-radius:99px">
          {case["failure_type"]}</span>
        <span style="color:#6b6b76;font-size:12.5px;margin-left:auto">
          {case["model"]} &middot; {case["run_label"]} &middot;
          {case["decoding_group"]} &middot; just {case["justification_id"]}
          &middot; sent {case["sentence_id"]}</span>
      </div>
      {prev_html}
      <div style="font-size:16px;padding:11px 13px;background:#f6f6fa;
                  border-radius:8px;margin:7px 0">{sentence_html}</div>
      {next_html}
      <div style="display:flex;flex-wrap:wrap;font-size:13.5px;color:#6b6b76;
                  margin-top:9px;padding-top:9px;border-top:1px dashed #e2e2e8">
        <span>marker <b>{value("marker")}</b></span>
        <span style="margin-left:18px">{prediction}</span>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ValidationStore:
    """Manual answers for the 50 cases, saved on every change.

    The file always holds all 50 rows with every sample column, so a partially
    completed file is still a valid input to the evaluator - unanswered rows
    simply have blank manual fields, which the evaluator reports as
    unlabelled rather than treating as answers.
    """

    def __init__(self, path, cases: List[dict]):
        self.path = Path(path)
        self.cases = cases
        self.answers: Dict[int, Dict[str, str]] = {}
        if self.path.exists():
            self._load()

    def _load(self):
        existing = pd.read_csv(self.path, encoding="utf-8-sig", dtype=str)
        for row in existing.to_dict("records"):
            try:
                case_id = int(row["validation_id"])
            except (KeyError, TypeError, ValueError):
                continue
            answer = {
                column: ("" if pd.isna(row.get(column)) else str(row.get(column, "")))
                for column in MANUAL_COLUMNS
            }
            if any(value for value in answer.values()):
                self.answers[case_id] = answer

    def get(self, case_id: int, field: str) -> str:
        return self.answers.get(int(case_id), {}).get(field, "")

    def set(self, case_id: int, field: str, value: str):
        self.answers.setdefault(int(case_id), {})[field] = value or ""
        self.save()

    def is_answered(self, case: dict) -> bool:
        stored = self.answers.get(int(case["validation_id"]), {})
        key = ("manual_is_connective" if case["failure_type"] == ACCEPTED
               else "manual_valid_relation_missed_by_discopy")
        return bool(stored.get(key, ""))

    def n_answered(self) -> int:
        return sum(1 for case in self.cases if self.is_answered(case))

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for case in self.cases:
            row = {column: case.get(column, "") for column in COMPLETED_COLUMNS}
            row.update(self.answers.get(int(case["validation_id"]), {}))
            for column in MANUAL_COLUMNS:
                row.setdefault(column, "")
                if pd.isna(row[column]):
                    row[column] = ""
            rows.append(row)
        return pd.DataFrame(rows)[COMPLETED_COLUMNS]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(self.path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class DiscourseReviewApp:
    """One case at a time, with prev/next/jump and save-on-change controls."""

    def __init__(self, cases: List[dict], store: ValidationStore):
        import ipywidgets as widgets

        self.widgets = widgets
        self.cases = cases
        self.store = store
        self.position = 0

        self.progress = widgets.HTML()
        self.body = widgets.HTML()
        self.controls = widgets.VBox()

        self.previous_button = widgets.Button(
            description="< Prev", layout=widgets.Layout(width="90px"))
        self.next_button = widgets.Button(
            description="Next >", layout=widgets.Layout(width="90px"))
        self.unanswered_button = widgets.Button(
            description="Next unanswered", layout=widgets.Layout(width="150px"))
        self.jump = widgets.IntText(value=1, layout=widgets.Layout(width="60px"))
        self.jump_button = widgets.Button(
            description="Go", layout=widgets.Layout(width="50px"))

        self.previous_button.on_click(lambda _: self.move(-1))
        self.next_button.on_click(lambda _: self.move(1))
        self.unanswered_button.on_click(lambda _: self.go_to_unanswered())
        self.jump_button.on_click(lambda _: self.go_to(self.jump.value - 1))

        self.ui = widgets.VBox([
            widgets.HBox([
                self.previous_button, self.next_button, self.unanswered_button,
                self.jump, self.jump_button, self.progress,
            ]),
            self.body,
            self.controls,
        ])
        self.render()

    # -- navigation -------------------------------------------------------
    def move(self, step):
        self.go_to(self.position + step)

    def go_to(self, index):
        self.position = max(0, min(len(self.cases) - 1, index))
        self.render()

    def go_to_unanswered(self):
        for offset in range(1, len(self.cases) + 1):
            index = (self.position + offset) % len(self.cases)
            if not self.store.is_answered(self.cases[index]):
                self.go_to(index)
                return
        self.render()

    # -- rendering --------------------------------------------------------
    def render(self):
        widgets = self.widgets
        case = self.cases[self.position]
        case_id = int(case["validation_id"])

        self.progress.value = (
            f'<span style="font-family:sans-serif;color:#6b6b76">'
            f'{self.store.n_answered()} / {len(self.cases)} answered &middot; '
            f'saving to <code>{self.store.path.name}</code></span>'
        )
        self.body.value = render_case_html(
            case, self.position + 1, len(self.cases)
        )

        rows = []

        def toggle(field, options, label):
            control = widgets.ToggleButtons(
                options=[("-", "")] + [(o, o) for o in options],
                value=self.store.get(case_id, field) or "",
                style={"button_width": "auto"},
            )
            control.observe(
                lambda change, f=field: (
                    self.store.set(case_id, f, change["new"]),
                    self._refresh_progress(),
                ) if change["name"] == "value" else None,
                names="value",
            )
            return widgets.VBox([
                widgets.HTML(f'<b style="font-size:13px">{label}</b>'), control
            ])

        if case["failure_type"] == ACCEPTED:
            rows.append(toggle(
                "manual_is_connective", ["yes", "no"],
                "Is the highlighted span actually functioning as a discourse "
                "connective?"))
            rows.append(toggle(
                "manual_top_level_category", TOP_LEVEL_CATEGORIES,
                "If yes: correct top-level PDTB category"))
        else:
            rows.append(toggle(
                "manual_valid_relation_missed_by_discopy", ["yes", "no"],
                "Is this a valid discourse relation that discopy failed to "
                "report?"))
            rows.append(toggle(
                "manual_top_level_category", TOP_LEVEL_CATEGORIES,
                "If yes: correct top-level PDTB category"))

        notes = widgets.Textarea(
            value=self.store.get(case_id, "notes"),
            placeholder="notes (optional)",
            layout=widgets.Layout(width="99%", height="52px"),
        )
        notes.observe(
            lambda change: self.store.set(case_id, "notes", change["new"])
            if change["name"] == "value" else None,
            names="value",
        )
        rows.append(widgets.VBox([
            widgets.HTML('<b style="font-size:13px">Notes</b>'), notes
        ]))

        self.controls.children = tuple(rows)

    def _refresh_progress(self):
        self.progress.value = (
            f'<span style="font-family:sans-serif;color:#6b6b76">'
            f'{self.store.n_answered()} / {len(self.cases)} answered &middot; '
            f'saving to <code>{self.store.path.name}</code></span>'
        )


def progress_summary(cases: List[dict], store: ValidationStore) -> pd.DataFrame:
    """Answered vs remaining, per stratum."""
    rows = []
    for failure_type in ("accepted", "rejected_nosense", "not_enumerated"):
        subset = [c for c in cases if c["failure_type"] == failure_type]
        answered = sum(1 for c in subset if store.is_answered(c))
        rows.append({
            "failure_type": failure_type,
            "n": len(subset),
            "answered": answered,
            "remaining": len(subset) - answered,
        })
    frame = pd.DataFrame(rows)
    frame.loc[len(frame)] = {
        "failure_type": "TOTAL",
        "n": int(frame["n"].sum()),
        "answered": int(frame["answered"].sum()),
        "remaining": int(frame["remaining"].sum()),
    }
    return frame
