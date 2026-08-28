# Artifact status manifest — RQ2 discourse

Every file under `discourse_parser/` (and the sibling `dimlex_analysis/`) is
classified below. **Nothing is deleted**; the classification decides only what
may be cited as a thesis result.

| status | meaning |
|---|---|
| **PRODUCTION** | a final thesis result. Derived only from `discopy_explicit_candidates.csv` filtered to `is_connective == True`. |
| **VALIDATION/DIAGNOSTIC** | evidence *about the method* — manual validation, DiMLex coverage, forced-span probe, rejected hybrid. Cite in the Method, never as a model result. |
| **DEPRECATED/EXPLORATORY** | superseded or out of the final scope. Kept for the record; not cited. |

Frozen pipeline: **native standard discopy, explicit relations only**. DiMLex is
a coverage/sensitivity diagnostic. The DiMLex-expanded hybrid is rejected. See
`DISCOURSE_PIPELINE_FINAL_HANDOFF.md` at the repository root.

---

## PRODUCTION

| file | note |
|---|---|
| `discopy_explicit_candidates.csv` | **the single source of truth.** 14,209 candidates, 5,504 accepted, 8,705 NoSense. Rejected candidates are retained so the accept/reject behaviour stays inspectable. |
| `thesis_tables/final_discourse/F0_run_level` | per-run counts, words and densities — the base every other final table aggregates |
| `thesis_tables/final_discourse/F1_overall_density` | **A** overall explicit-relation density |
| `thesis_tables/final_discourse/F2_top_level_density` | **B** top-level PDTB density per 100 words |
| `thesis_tables/final_discourse/F3_top_level_composition` | **C** relative four-class composition |
| `thesis_tables/final_discourse/F4_fine_grained_senses` | **D** level-2 senses (secondary) |
| `thesis_tables/final_discourse/F5_bootstrap_pairwise` | **E** paired game-level bootstrap, 95% percentile CIs |
| `figures/final_discourse/F1_top_level_density.png` | from F2 |
| `figures/final_discourse/F2_composition.png` | from F3 |
| `figures/final_discourse/F3_fine_grained_senses.png` | from F4 |
| `figures/final_discourse/F4_bootstrap_forest_stochastic.png` | from F5 — the main inferential figure |
| `figures/final_discourse/F5_bootstrap_forest_greedy.png` | from F5 — greedy, separate figure |

Generator: **`notebooks/04_justification_analysis/7_final_discourse_analysis.ipynb`**
(canonical), over `src/justification_analysis/comparison/discourse_final.py` and
`discourse_figures.py`. Re-running the notebook reproduces all six tables and
all five figures; it asserts the frozen invariants first and fails rather than
silently reporting different numbers.

Frozen invariants:

```
14,209 candidates · 5,504 accepted · 8,705 NoSense
2,292 justifications · 191 games · 191 justifications per run
169,748 WORD_PATTERN tokens · 9 observed level-2 senses
relation_type == "Explicit" throughout
```

## VALIDATION/DIAGNOSTIC

**Manual validation** — all four rounds complete and closed, details in
`VALIDATION_README.md`:
`manual_validation_sample_50.csv`, `manual_validation_completed.csv`,
`manual_validation_report.txt`, `manual_error_table.csv`,
`coverage_inspection_sample_30.csv`, `coverage_inspection_completed.csv`,
`coverage_inspection_flags.csv`,
`forced_span_probe/forced_span_validation_*.csv`,
`experimental_hybrid/hybrid_validation_*.csv`.

**DiMLex coverage machinery** — lexical coverage only, never extraction:
`dimlex_occurrences.csv`, `alignment_dimlex_side.csv`,
`alignment_discopy_side.csv`, `coverage_vs_classification_summary.csv`,
`connective_inventory_comparison.csv`, `coverage_gap_triage.csv`,
`coverage_gap_summary_by_form.csv`, `sense_change_crosstab.csv`,
`conditional_marker_diagnostic.csv`, and thesis tables `06`, `07`, `08`,
`10b`, `11`, `12`, `13`, `14`.

**Forced-span probe** — `forced_span_probe/` (see its `EXPERIMENTAL.md`) and
tables `15`–`18`.

**Rejected hybrid** — `experimental_hybrid/` (see its `EXPERIMENTAL.md`) and
tables `19`–`23`.

**Method evidence about the parser, not the models:**
`thesis_tables/02c_connective_form_acceptance` and
`figures/method/fig4_contextual_filtering.png` — candidate acceptance rate per
connective form (`for`: 1,632 candidates, 0 accepted; `and`: 18.5%). Regenerated
by the final section of notebook 7.

## DEPRECATED/EXPLORATORY

| file | why |
|---|---|
| `thesis_tables/01`–`05`, `09` | superseded by `final_discourse/F1`–`F3`, `F0`. Values verified to match the recomputation exactly; `F3` differs from `05` on purpose (shares are now computed per run then averaged, not pooled). |
| `thesis_tables/10_cooccurrence_*` (6 files) | no discourse-only co-occurrence analysis is planned; co-occurrence moves to the later joint discourse × semantic work |
| `discopy_run_level_statistics.csv`, `discopy_model_summary.csv`, `discopy_category_rates_per100w.csv` | superseded by `final_discourse/F0`–`F2` |
| `side_by_side_dimlex_vs_discopy.csv`, `temporal_rate_both_tools.csv` | early ad-hoc comparisons, superseded by tables `06`/`07` |
| `../dimlex_analysis/` (6 files) | DiMLex lexical counts stratified by vote correctness / swap — DiMLex counts are not model results, and stratification is out of the final scope |

## Note on diagnostic table 18

`thesis_tables/18_hybrid_actual_vs_sensitivity.csv` reports `hybrid_gain_pct`
3.9 for 31B stochastic where an earlier throwaway version reported 3.8. **3.9 is
correct and needs no restoration:** it is a direct recomputation from the
unrounded rates, whereas the original rounded the rates to three decimals before
dividing. The table is a diagnostic about the rejected hybrid; the change is
immaterial to its conclusion and every other cell is unchanged.

## Notebook roles

| notebook | role |
|---|---|
| `1_dimlex_justification_analysis.ipynb` | DiMLex lexical diagnostic |
| `2_discopy_discourse_analysis.ipynb` | parser diagnostics and inspection |
| `3`–`6` | manual validation, complete and closed |
| **`7_final_discourse_analysis.ipynb`** | **canonical thesis-result generation** |

`src/justification_analysis/comparison/run_diagnostic_tables.py` regenerates the
diagnostic tables `10b`–`23`; it exists for provenance and is not part of the
final workflow.
