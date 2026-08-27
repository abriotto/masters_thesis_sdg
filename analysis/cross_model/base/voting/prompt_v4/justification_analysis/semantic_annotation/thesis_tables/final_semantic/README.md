# Final semantic tables

Produced by `notebooks/04_justification_analysis/8_final_semantic_analysis.ipynb`
over `results/justification_annotation/full_frozen/annotations.jsonl`
(frozen schema, 2,292 justifications, 191 games, 3 models x 4 runs).

Do not edit by hand: re-run the notebook.

| table | contents |
|---|---|
| `S0_integrity_summary` | every corpus count recomputed against the frozen invariants |
| `S0b_repaired_sentences` | sentence texts repaired from the input shards (1 row) |
| `S0c_multilabel_distribution` | labels per sentence, with both denominators |
| `S1_annotation_summary` | corpus descriptives by model and decoding group |
| `S1b_sentence_length` | justification length per model and run |
| `S2_run_level_prevalence` | prevalence per model x decoding x run x category (incl. `Other`) |
| `S3_model_semantic_prevalence` | **primary**: prevalence mean +/- SD across runs |
| `S3b_density_sensitivity` | SENSITIVITY ONLY: assignments per 100 sentences |
| `S4_prevalence_bootstrap_differences` | paired game-level bootstrap, pairwise model differences |
| `S5_cooccurrence_joint_prevalence` | justification-level joint prevalence + support |
| `S5b_cooccurrence_run_level` | the same, per run, before averaging |
| `S6_cooccurrence_lift` | lift (secondary/diagnostic), support retained |
| `S6b_cooccurrence_ranked_pairs` | unordered pairs ranked, thin support flagged |
| `S7_correctness_presence_association` | P(correct \| present) - P(correct \| absent), cluster bootstrap |
| `S8_correctness_stability_groups` | games per 0/3, 1/3, 2/3, 3/3 correctness group |
| `S9_correctness_stability_semantics` | mean Q per category within each K group |
| `S10_within_game_correctness_contrasts` | within mixed games, correct minus incorrect |

Conventions: prevalence is the share of justifications invoking a category at
least once (justification level, not sentence level). Stochastic rows are the
mean of runs 1-3 with SD; greedy is a single run and its SD is NaN. Stochastic
and greedy are never pooled. All bootstraps: 10,000 replicates, seed 20260826,
95% percentile intervals, resampling GAMES with all their runs attached.
`ci_excludes_zero` describes the interval; it is not a significance test.

`Other` appears in S1, S2 and S3 only. It is 0.3% of labels and is excluded
from co-occurrence, the bootstrap comparisons and the correctness analyses.
