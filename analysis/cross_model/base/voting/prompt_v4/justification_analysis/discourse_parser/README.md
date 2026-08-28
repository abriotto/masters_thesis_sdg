# Explicit discourse relations (RQ2)

Everything here comes from one parser run over the 2,292 vote justifications.
The pipeline is frozen: **native standard discopy, explicit relations only** —
see `DISCOURSE_PIPELINE_FINAL_HANDOFF.md` at the repository root.

## Where the thesis results are

```
thesis_tables/final_discourse/   F0-F5   the six final tables   (+ README)
figures/final_discourse/         F1-F5   the five final figures (+ README)
```

**Only artifacts derived from `discopy_explicit_candidates.csv` filtered to
`is_connective == True` may go in those two directories.** No DiMLex,
forced-span or hybrid-derived output.

Everything else in this folder is diagnostic or superseded.
**`ARTIFACT_STATUS.md` classifies every file** as PRODUCTION,
VALIDATION/DIAGNOSTIC or DEPRECATED/EXPLORATORY. Read it before citing anything.

## Regenerating

```bash
"C:/Users/annab/miniconda3/envs/sdglogs/python.exe" -m nbconvert --to notebook --execute --inplace notebooks/04_justification_analysis/7_final_discourse_analysis.ipynb
```

Notebook 7 is the canonical generator for every final table and figure. It
asserts the frozen invariants before computing anything.

Re-running the **parser** itself takes ~74 min on CPU in the separate
`discopy-env` and is only necessary if the corpus or the checkpoint changes.

## Validation

All four manual rounds are complete and closed — `VALIDATION_README.md`. No
further manual validation is planned.
