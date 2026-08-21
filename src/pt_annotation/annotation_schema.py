"""
The annotation schemas, as named versions.

The v2 prompt changed the scheme, not just its wording: categories were
renamed and added, and two fields were dropped. Rather than hard-switching
and orphaning the v1 pilot output, each version is described here and the
runner, validator and reviewer read the description.

What actually changed:

  Deduction   -> Mechanical        (semantic content invoked, not inference form)
  Consistency -> ClaimComparison
  Social      -> SocialJudgment
  (new)       -> Uncertainty       (was scattered across Other in v1)
  `use`             dropped        (used / discounted / mentioned)
  `rule_mentioned`  dropped        (subsumed by Mechanical)

v2 stops coding the *form* of the inference and codes the *type of
information* the sentence reasons with, which is why the inference-flavoured
names went away.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Schema:
    name: str
    prompt_filename: str
    categories: tuple
    has_use: bool
    has_rule_mentioned: bool
    uses: tuple = ()

    def allowed_categories(self):
        return set(self.categories)


V1 = Schema(
    name="v1",
    prompt_filename="justification_annotation.txt",
    categories=(
        "Deduction", "Consistency", "Payoff",
        "Testimony", "Social", "Behavioral", "Other",
    ),
    has_use=True,
    has_rule_mentioned=True,
    uses=("used", "discounted", "mentioned"),
)

V2 = Schema(
    name="v2",
    prompt_filename="justification_annotation_v2.txt",
    categories=(
        "Mechanical", "Testimony", "SocialJudgment", "Behavioral",
        "ClaimComparison", "Payoff", "Uncertainty", "Other",
    ),
    has_use=False,
    has_rule_mentioned=False,
)

SCHEMAS = {"v1": V1, "v2": V2}

DEFAULT_SCHEMA = "v2"


def get_schema(name):
    if name not in SCHEMAS:
        raise ValueError(f"Unknown schema {name!r}. Known: {sorted(SCHEMAS)}")
    return SCHEMAS[name]


# Light backgrounds with forced dark text, so highlighting survives a dark
# Jupyter theme. Both vocabularies are present: the reviewer must be able to
# open either pilot.
CATEGORY_COLORS = {
    # v2
    "Mechanical":      "#cfe4ff",
    "Testimony":       "#e6dcff",
    "SocialJudgment":  "#ffd4e0",
    "Behavioral":      "#fff2b0",
    "ClaimComparison": "#d4f0d4",
    "Payoff":          "#ffe4c0",
    "Uncertainty":     "#d9d9f3",
    "Other":           "#e2e2e2",
    # v1
    "Deduction":       "#cfe4ff",
    "Consistency":     "#d4f0d4",
    "Social":          "#ffd4e0",
}


def detect_schema(records):
    """Infer the schema from saved output, so a reviewer opened on a folder
    does not need to be told which pilot it is looking at."""
    for record in records:
        name = (record.get("metadata") or {}).get("schema")
        if name in SCHEMAS:
            return SCHEMAS[name]

    seen = {
        item.get("category")
        for record in records
        for sentence in (record.get("annotation") or {}).get("sentences", [])
        for item in sentence.get("annotations", [])
    }
    if seen & set(V2.categories) - set(V1.categories):
        return V2
    if seen & set(V1.categories) - set(V2.categories):
        return V1
    return get_schema(DEFAULT_SCHEMA)
