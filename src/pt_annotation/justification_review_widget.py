"""
Click-through reviewer for justification annotations.

One justification on screen at a time: the text with evidence spans
highlighted, and one verdict control per annotation. Keyboard-free, so it
works the same in JupyterLab and in the notebook interface.

Every control writes straight through to the VerdictStore, which saves on
each change. There is no "submit" step to forget, and closing the kernel
mid-review loses nothing.
"""

import ipywidgets as widgets
from IPython.display import HTML, display

from src.pt_annotation.justification_review_tools import VERDICTS, render_justification


class ReviewApp:
    def __init__(self, records, store, schema, sample=None):
        self.records = records
        self.store = store
        self.schema = schema
        self.sample = sample or {}
        self.position = 0

        # v2 dropped `use`, so the use dropdown and the "wrong use" verdict
        # are removed rather than left on screen as dead controls.
        self.verdicts = [
            v for v in VERDICTS if schema.has_use or v != "wrong use"
        ]
        self.categories = list(schema.categories)

        self.total_annotations = sum(
            len(s.get("annotations", []))
            for r in records
            for s in (r.get("annotation") or {}).get("sentences", [])
        )

        self.progress = widgets.HTML()
        self.header = widgets.HTML()
        self.body = widgets.HTML()
        self.controls = widgets.VBox()

        self.previous_button = widgets.Button(description="< Prev", layout=widgets.Layout(width="90px"))
        self.next_button = widgets.Button(description="Next >", layout=widgets.Layout(width="90px"))
        self.accept_all_button = widgets.Button(
            description="Mark all ok",
            button_style="success",
            tooltip="Set every unreviewed annotation on this justification to ok",
            layout=widgets.Layout(width="120px"),
        )
        self.jump = widgets.IntText(value=1, layout=widgets.Layout(width="60px"))
        self.jump_button = widgets.Button(description="Go", layout=widgets.Layout(width="50px"))

        self.previous_button.on_click(lambda _: self.move(-1))
        self.next_button.on_click(lambda _: self.move(1))
        self.accept_all_button.on_click(lambda _: self.accept_all())
        self.jump_button.on_click(lambda _: self.go_to(self.jump.value - 1))

        self.navigation = widgets.HBox([
            self.previous_button, self.next_button,
            widgets.HTML("&nbsp;&nbsp;"),
            self.accept_all_button,
            widgets.HTML("&nbsp;&nbsp;go to"), self.jump, self.jump_button,
        ])

    # ------------------------------------------------------------

    def display(self):
        display(widgets.VBox([
            self.progress, self.navigation, self.header, self.body, self.controls,
        ]))
        self.render()

    def move(self, step):
        self.go_to(self.position + step)

    def go_to(self, position):
        self.position = max(0, min(position, len(self.records) - 1))
        self.render()

    # ------------------------------------------------------------

    def current(self):
        return self.records[self.position]

    def render(self):
        record = self.current()
        metadata = record["metadata"]
        justification_id = metadata["justification_id"]

        judged, total = self.store.progress(self.total_annotations)
        self.progress.value = (
            f'<div style="font-family:system-ui,sans-serif">'
            f'<b>{judged}/{total}</b> annotations reviewed &nbsp;&middot;&nbsp; '
            f'justification <b>{self.position + 1}</b> of {len(self.records)}</div>'
        )

        self.header.value = (
            f'<div style="font-family:ui-monospace,monospace;font-size:0.75em;'
            f'color:#888;margin-top:6px">{justification_id}</div>'
        )

        self.body.value = render_justification(record, self.sample.get(justification_id))
        self.controls.children = self.build_controls(record)

    # ------------------------------------------------------------

    def build_controls(self, record):
        metadata = record["metadata"]
        justification_id = metadata["justification_id"]
        annotation = record.get("annotation") or {}

        blocks = []

        for sentence in annotation.get("sentences", []):
            sentence_id = sentence.get("sentence_id")
            items = sentence.get("annotations", [])
            rows = []

            for index, item in enumerate(items):
                rows.append(
                    self.annotation_row(justification_id, sentence_id, index, item)
                )

            rows.append(self.missed_row(justification_id, sentence_id, bool(items)))

            blocks.append(widgets.VBox(
                [widgets.HTML(
                    f'<div style="font-weight:600;font-size:0.8em;color:#555;'
                    f'margin-top:6px">sentence {sentence_id}</div>'
                )] + rows,
                layout=widgets.Layout(border="1px solid #eee", padding="4px", margin="2px 0"),
            ))

        return blocks

    def annotation_row(self, justification_id, sentence_id, index, item):
        existing = self.store.get(justification_id, sentence_id, index)

        use_text = (
            f'<span style="color:#777">{item.get("use")}</span>'
            if self.schema.has_use else ""
        )
        label = widgets.HTML(
            f'<div style="width:170px;font-size:0.85em">'
            f'<b>{item.get("category")}</b> {use_text}</div>',
            layout=widgets.Layout(width="180px"),
        )

        verdict = widgets.Dropdown(
            options=self.verdicts,
            value=existing.get("verdict", "") or "",
            layout=widgets.Layout(width="130px"),
        )
        corrected_category = widgets.Dropdown(
            options=[""] + self.categories,
            value=existing.get("corrected_category", "") or "",
            layout=widgets.Layout(width="150px"),
        )
        corrected_use = widgets.Dropdown(
            options=[""] + list(self.schema.uses),
            value=existing.get("corrected_use", "") or "",
            layout=widgets.Layout(width="110px"),
        )
        note = widgets.Text(
            value=str(existing.get("note", "") or ""),
            placeholder="note",
            layout=widgets.Layout(width="240px"),
        )

        def persist(_=None):
            self.store.set({
                "justification_id": justification_id,
                "sentence_id": sentence_id,
                "annotation_index": index,
                "category": item.get("category"),
                "use": item.get("use"),
                "evidence_span": item.get("evidence_span"),
                "verdict": verdict.value,
                "corrected_category": corrected_category.value,
                "corrected_use": corrected_use.value,
                "note": note.value,
            })
            judged, total = self.store.progress(self.total_annotations)
            self.progress.value = (
                f'<div style="font-family:system-ui,sans-serif">'
                f'<b>{judged}/{total}</b> annotations reviewed &nbsp;&middot;&nbsp; '
                f'justification <b>{self.position + 1}</b> of {len(self.records)}</div>'
            )

        for control in (verdict, corrected_category, corrected_use, note):
            control.observe(persist, names="value")

        row = [label, verdict, corrected_category]
        if self.schema.has_use:
            row.append(corrected_use)
        row.append(note)
        return widgets.HBox(row)

    def missed_row(self, justification_id, sentence_id, has_annotations):
        existing = self.store.get_missed(justification_id, sentence_id)

        prompt = "missed a label?" if has_annotations else "should this be labelled?"
        label = widgets.HTML(
            f'<div style="width:150px;font-size:0.8em;color:#999">{prompt}</div>',
            layout=widgets.Layout(width="160px"),
        )
        missed_category = widgets.Dropdown(
            options=[""] + self.categories,
            value=existing.get("missed_category", "") or "",
            layout=widgets.Layout(width="130px"),
        )
        missed_note = widgets.Text(
            value=str(existing.get("missed_note", "") or ""),
            placeholder="what was missed",
            layout=widgets.Layout(width="240px"),
        )

        def persist(_=None):
            self.store.set_missed(
                justification_id, sentence_id,
                missed_category.value, missed_note.value,
            )

        for control in (missed_category, missed_note):
            control.observe(persist, names="value")

        return widgets.HBox([label, missed_category, missed_note])

    # ------------------------------------------------------------

    def accept_all(self):
        """Mark every not-yet-judged annotation on this justification ok.

        For the common case where the whole justification reads correctly.
        It deliberately does not overwrite verdicts you already set, so a
        stray click cannot erase a considered judgement.
        """
        record = self.current()
        metadata = record["metadata"]
        justification_id = metadata["justification_id"]

        for sentence in (record.get("annotation") or {}).get("sentences", []):
            sentence_id = sentence.get("sentence_id")
            for index, item in enumerate(sentence.get("annotations", [])):
                existing = self.store.get(justification_id, sentence_id, index)
                if existing.get("verdict"):
                    continue
                self.store.set({
                    "justification_id": justification_id,
                    "sentence_id": sentence_id,
                    "annotation_index": index,
                    "category": item.get("category"),
                    "use": item.get("use"),
                    "evidence_span": item.get("evidence_span"),
                    "verdict": "ok",
                    "corrected_category": "",
                    "corrected_use": "",
                    "note": "",
                })
        self.render()
