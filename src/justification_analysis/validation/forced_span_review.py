"""Notebook review tooling for the forced-span probe.

Separate from `discourse_review_tools` because the question is different: there
the judgement was binary (is this a connective / was a relation missed), here
it is layered - is the span an Explicit connective at all, and if so is the
top-level sense right, and is the finer sense reasonable.

Save-on-change, same as the other reviewers: closing the notebook mid-review
loses nothing.

ANNOTATION CRITERION, applied throughout:

    Given the PDTB-style Explicit-relation criterion that discopy implements,
    is this forced `given` / `given that` span a valid Explicit discourse
    connective - and if so, is the classifier's sense appropriate?

NOT the broader question "does this expression convey some discourse
relation?". A construction can express causality and still not be a
PDTB Explicit connective, which is exactly why these forms sit outside
discopy's inventory in the first place.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List

import pandas as pd

TOP_LEVEL_CATEGORIES = ["Comparison", "Contingency", "Expansion", "Temporal"]

MANUAL_COLUMNS = [
    "manual_is_explicit_connective",
    "manual_top_level_correct",
    "manual_full_sense_correct",
    "manual_expected_top_level",
    "manual_expected_full_sense",
    "manual_notes",
]

CARRY_COLUMNS = [
    "probe_id", "form", "marker", "surface", "model", "run_label",
    "decoding_group", "justification_id", "sentence_id", "sentence_text",
    "char_spans", "start", "end", "predicted_sense", "predicted_top_level",
    "accepted", "confidence", "p_nosense", "triage", "dimlex_category",
]

COMPLETED_COLUMNS = CARRY_COLUMNS + MANUAL_COLUMNS


def render_forced_case(case: dict, position: int, total: int) -> str:
    """One case as HTML, marker highlighted from its exact offsets."""
    sentence = str(case.get("sentence_text", ""))
    start, end = case.get("sent_start", -1), case.get("sent_end", -1)

    if 0 <= start < end <= len(sentence):
        marked = (
            html.escape(sentence[:start])
            + '<mark style="background:#fde68a;padding:1px 3px;'
              'border-radius:3px;font-weight:650">'
            + html.escape(sentence[start:end]) + "</mark>"
            + html.escape(sentence[end:])
        )
    else:
        marked = html.escape(sentence)

    accepted = bool(case.get("accepted"))
    colour = "#2563eb" if accepted else "#b45309"
    verdict = "ACCEPTED" if accepted else "NoSense"
    confidence = case.get("confidence")
    confidence_text = "-" if pd.isna(confidence) else f"{float(confidence):.3f}"

    return f"""
    <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;
                border:1px solid #e2e2e8;border-radius:10px;padding:14px 16px">
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <span style="font-weight:700">case {position} / {total}</span>
        <span style="background:{colour};color:#fff;font-size:12px;
                     font-weight:650;padding:3px 10px;border-radius:99px">
          {verdict}</span>
        <span style="background:#475569;color:#fff;font-size:12px;
                     font-weight:650;padding:3px 10px;border-radius:99px">
          {case.get('form','')}</span>
        <span style="color:#6b6b76;font-size:12.5px;margin-left:auto">
          {case.get('model','')} &middot; {case.get('run_label','')} &middot;
          {case.get('decoding_group','')} &middot;
          just {case.get('justification_id','')} &middot;
          sent {case.get('sentence_id','')}</span>
      </div>
      <div style="font-size:16px;padding:11px 13px;background:#f6f6fa;
                  border-radius:8px;margin:8px 0">{marked}</div>
      <div style="display:flex;gap:18px;flex-wrap:wrap;font-size:13.5px;
                  color:#6b6b76;padding-top:9px;border-top:1px dashed #e2e2e8">
        <span>classifier sense <b style="color:#1a1a1e">
          {case.get('predicted_sense','')}</b></span>
        <span>top level <b style="color:#1a1a1e">
          {case.get('predicted_top_level') or '-'}</b></span>
        <span>confidence <b style="color:#1a1a1e">{confidence_text}</b></span>
        <span>DiMLex <b style="color:#1a1a1e">
          {case.get('dimlex_category','')}</b></span>
      </div>
    </div>
    """


class ForcedSpanStore:
    """Manual answers, written to CSV on every change."""

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
                case_id = int(row["probe_id"])
            except (KeyError, TypeError, ValueError):
                continue
            answer = {
                column: ("" if pd.isna(row.get(column)) else str(row.get(column, "")))
                for column in MANUAL_COLUMNS
            }
            if any(answer.values()):
                self.answers[case_id] = answer

    def get(self, case_id: int, field: str) -> str:
        return self.answers.get(int(case_id), {}).get(field, "")

    def set(self, case_id: int, field: str, value: str):
        self.answers.setdefault(int(case_id), {})[field] = value or ""
        self.save()

    def is_answered(self, case: dict) -> bool:
        stored = self.answers.get(int(case["probe_id"]), {})
        return bool(stored.get("manual_is_explicit_connective", ""))

    def n_answered(self) -> int:
        return sum(1 for case in self.cases if self.is_answered(case))

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for case in self.cases:
            row = {c: case.get(c, "") for c in CARRY_COLUMNS}
            row.update(self.answers.get(int(case["probe_id"]), {}))
            for column in MANUAL_COLUMNS:
                row.setdefault(column, "")
            rows.append(row)
        return pd.DataFrame(rows)[COMPLETED_COLUMNS]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(self.path, index=False, encoding="utf-8-sig")


class ForcedSpanReviewApp:
    """One case at a time, with the layered judgement the probe needs."""

    def __init__(self, cases: List[dict], store: ForcedSpanStore):
        import ipywidgets as widgets

        self.widgets = widgets
        self.cases = cases
        self.store = store
        self.position = 0

        self.progress = widgets.HTML()
        self.body = widgets.HTML()
        self.controls = widgets.VBox()

        previous_button = widgets.Button(description="< Prev",
                                         layout=widgets.Layout(width="90px"))
        next_button = widgets.Button(description="Next >",
                                     layout=widgets.Layout(width="90px"))
        unanswered_button = widgets.Button(description="Next unanswered",
                                           layout=widgets.Layout(width="150px"))
        self.jump = widgets.IntText(value=1, layout=widgets.Layout(width="60px"))
        jump_button = widgets.Button(description="Go",
                                     layout=widgets.Layout(width="50px"))

        previous_button.on_click(lambda _: self.go_to(self.position - 1))
        next_button.on_click(lambda _: self.go_to(self.position + 1))
        unanswered_button.on_click(lambda _: self.go_to_unanswered())
        jump_button.on_click(lambda _: self.go_to(self.jump.value - 1))

        self.ui = widgets.VBox([
            widgets.HBox([previous_button, next_button, unanswered_button,
                          self.jump, jump_button, self.progress]),
            self.body,
            self.controls,
        ])
        self.render()

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

    def _progress_html(self):
        return (f'<span style="font-family:sans-serif;color:#6b6b76">'
                f'{self.store.n_answered()} / {len(self.cases)} answered '
                f'&middot; saving to <code>{self.store.path.name}</code></span>')

    def render(self):
        widgets = self.widgets
        case = self.cases[self.position]
        case_id = int(case["probe_id"])

        self.progress.value = self._progress_html()
        self.body.value = render_forced_case(case, self.position + 1,
                                             len(self.cases))

        def toggle(field, options, label):
            control = widgets.ToggleButtons(
                options=[("-", "")] + [(o, o) for o in options],
                value=self.store.get(case_id, field) or "",
                style={"button_width": "auto"},
            )
            control.observe(
                lambda change, f=field: (
                    self.store.set(case_id, f, change["new"]),
                    setattr(self.progress, "value", self._progress_html()),
                ) if change["name"] == "value" else None,
                names="value",
            )
            return widgets.VBox([
                widgets.HTML(f'<b style="font-size:13px">{label}</b>'), control])

        def text(field, label, placeholder=""):
            control = widgets.Text(
                value=self.store.get(case_id, field),
                placeholder=placeholder,
                layout=widgets.Layout(width="99%"))
            control.observe(
                lambda change, f=field: self.store.set(case_id, f, change["new"])
                if change["name"] == "value" else None,
                names="value")
            return widgets.VBox([
                widgets.HTML(f'<b style="font-size:13px">{label}</b>'), control])

        rows = [
            toggle("manual_is_explicit_connective", ["yes", "no"],
                   "Is this span a valid PDTB-style Explicit discourse "
                   "connective? (not merely: does it convey a relation)"),
            toggle("manual_top_level_correct", ["yes", "no", "n/a"],
                   "Is the classifier's TOP-LEVEL sense correct? "
                   "(n/a if not a connective, or if NoSense)"),
            toggle("manual_full_sense_correct", ["yes", "no", "n/a"],
                   "Is the classifier's FULL sense reasonable?"),
            toggle("manual_expected_top_level", TOP_LEVEL_CATEGORIES,
                   "Expected top-level category (if it is a connective)"),
            text("manual_expected_full_sense",
                 "Expected full sense (optional)", "e.g. Contingency.Cause"),
            text("manual_notes", "Notes (optional)"),
        ]
        self.controls.children = tuple(rows)


def progress_summary(cases: List[dict], store: ForcedSpanStore) -> pd.DataFrame:
    rows = []
    for key, label in (("form", "form"),):
        pass
    frame = pd.DataFrame([{
        "form": c["form"],
        "prediction": "accepted" if c.get("accepted") else "NoSense",
        "answered": store.is_answered(c),
    } for c in cases])
    return (frame.groupby(["form", "prediction"], observed=True)
            .agg(n=("answered", "size"), answered=("answered", "sum"))
            .assign(remaining=lambda t: t["n"] - t["answered"])
            .reset_index())
