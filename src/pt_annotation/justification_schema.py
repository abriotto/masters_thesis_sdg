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

# v3 is v2's scheme with rewritten guidance: the categories, the output shape
# and the absent `use`/`rule_mentioned` fields are all identical, verified
# against the prompt's OUTPUT FORMAT section. Only the wording that leads the
# annotator to a category changed, which is why this is a new prompt file
# rather than a new vocabulary.
#
# Note the filename is "annotations" (plural) where v1 and v2 are
# "annotation" -- matching the file on disk, not the pattern.
V3 = Schema(
    name="v3",
    prompt_filename="justification_annotations_v3.txt",
    categories=V2.categories,
    has_use=False,
    has_rule_mentioned=False,
)

# The candidate for the full run. Same taxonomy and output shape as v2/v3 --
# verified against the prompt's OUTPUT FORMAT section -- with the category
# guidance clarified further.
#
# "frozen" is a claim about the scheme, not a version bump: it is the wording
# intended to be fixed for the full corpus, which is why it gets a validation
# run on justifications the earlier pilots never touched. v1-v3 were developed
# against the same 40, so those 40 can no longer measure anything: any prompt
# tuned on them will look good on them.
FROZEN = Schema(
    name="frozen",
    prompt_filename="justification_annotations_frozen.txt",
    categories=V2.categories,
    has_use=False,
    has_rule_mentioned=False,
)

SCHEMAS = {"v1": V1, "v2": V2, "v3": V3, "frozen": FROZEN}

DEFAULT_SCHEMA = "frozen"


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
