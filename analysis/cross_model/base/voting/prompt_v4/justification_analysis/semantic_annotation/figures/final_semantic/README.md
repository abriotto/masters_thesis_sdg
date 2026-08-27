# Final semantic figures

Produced by `notebooks/04_justification_analysis/8_final_semantic_analysis.ipynb`.
300 dpi PNG, drawn directly from the tables in `../../thesis_tables/final_semantic/`.

| figure | contents |
|---|---|
| `F1_semantic_prevalence.png` | category prevalence by model; stochastic bars +/- SD, greedy as open diamonds |
| `F2_semantic_cooccurrence.png` | joint prevalence (top) and lift (bottom) per model, stochastic |
| `F2b_semantic_cooccurrence_greedy.png` | the same for greedy decoding |
| `F3_correctness_presence_association.png` | present-minus-absent correctness deltas with bootstrap CIs |
| `F3b_correctness_presence_greedy.png` | the same for greedy decoding |
| `F4_correctness_stability.png` | mean Q per category across the 0/3 - 3/3 groups |
| `F5_within_game_correctness.png` | within mixed games, correct minus incorrect, with CIs |

Model colours are viridis at 0.15 / 0.45 / 0.72 for E2B / E4B / 31B, matching
the discourse figures. Greedy never carries an error bar: it is one run.

These are analysis artifacts. Not all of them belong in the thesis body.
