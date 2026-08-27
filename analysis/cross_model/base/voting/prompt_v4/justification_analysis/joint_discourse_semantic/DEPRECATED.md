# DEPRECATED - do not use for thesis results

**Status: superseded 2026-08-27.** Kept for provenance and debugging only.

Everything in this directory defines a discourse x semantic association as
**same-sentence co-occurrence**. That definition is wrong for this corpus:

1. **It misses relations that cross a sentence boundary.** In
   *"He swapped Paul with Mike. Therefore Mike is the Werewolf."* the
   Mechanical content and the Contingency marker sit in different sentences,
   and both plainly belong to the same short argument. The sentence-level
   analysis records nothing.

2. **It is a cross-model confound.** A model that packs a thought into one
   sentence scores a co-occurrence; a model that splits the same thought over
   two does not. These three models differ systematically in justification
   length (3.16 / 3.89 / 3.48 mean sentences), so the metric was partly
   measuring packaging rather than association.

## What this changed, concretely

The replacement analysis is not a cosmetic re-cut. At least one headline
finding here does not survive it:

| pairing | sentence-level lift (this directory) | justification-level lift | verdict |
|---|---|---|---|
| E2B Payoff x Contingency | 2.42 | **1.01** | an artefact of packaging - E2B states payoff and cause in one clause, but at justification level the two are independent |
| 31B Mechanical x Contingency | 2.42 | 1.25 | attenuates but survives |
| ClaimComparison x Temporal | 6.56 / 4.82 / 2.34 | 2.98 / 2.24 / 1.30 | survives in all three models |

The E2B row is the reason this directory is deprecated rather than merely
superseded: read on its own it supports a conclusion the corrected analysis
contradicts.

## Replacement

* analysis: `../joint_discourse_semantic_justification/`
* notebook: `notebooks/04_justification_analysis/10_final_joint_justification_level.ipynb`
* module: `src/justification_analysis/joint_justification/`

The sentence-level machinery itself is still used underneath the replacement -
loading, the exact relation-to-sentence alignment, and the same-sentence
co-presence check that became the **localization diagnostic**. What is
deprecated is treating same-sentence co-occurrence as the *definition* of
association, not the sentence-level records.

`qualitative/` here is also superseded: its examples were selected from
same-sentence pairings, so they cannot characterise a justification-level
association. See `../joint_discourse_semantic_justification/qualitative/`.
