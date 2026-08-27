# Final semantic figures

Produced by `notebooks/04_justification_analysis/8_final_semantic_analysis.ipynb`.
300 dpi PNG, drawn directly from the tables in `../../thesis_tables/final_semantic/`.

| figure | contents |
|---|---|
| `F1_semantic_prevalence.png` | category prevalence by model; stochastic bars +/- SD, greedy as open diamonds |
| `F2_semantic_cooccurrence_prevalence.png` | joint prevalence, three model panels, stochastic |
| `F2b_semantic_cooccurrence_lift.png` | lift, three model panels, stochastic |
| `F2c_semantic_cooccurrence_prevalence_greedy.png` | joint prevalence, greedy decoding |
| `F2d_semantic_cooccurrence_lift_greedy.png` | lift, greedy decoding |
| `F3_correctness_presence_association.png` | present-minus-absent correctness deltas with bootstrap CIs |
| `F3b_correctness_presence_greedy.png` | the same for greedy decoding |
| `F4_correctness_stability.png` | mean Q per category across the 0/3 - 3/3 groups |
| `F5_within_game_correctness.png` | within mixed games, correct minus incorrect, with CIs |

Model colours are viridis at 0.15 / 0.45 / 0.72 for E2B / E4B / 31B, matching
the discourse figures. Greedy never carries an error bar: it is one run.

## Co-occurrence figures

The four `F2*` figures replace an earlier pair of six-panel figures that stacked
joint prevalence over lift; those were too tall and dense for a single column,
and their shared title wrongly described the greedy panels as a "mean of 3
runs". Each figure is now one row of three model panels, authored at 6.5 in
wide so it prints at its authored point sizes under
`\includegraphics[width=\linewidth]`.

* joint-prevalence panels share one 0-100% viridis scale; the diagonal is a
  category with itself, i.e. its marginal prevalence, and is outlined in white;
* lift panels share one scale centred on 1.0, red above and blue below, with a
  rule on the colorbar at 1.0. The scale is symmetric about 1, so its low end
  is 1 - span; where that is below zero (greedy) the tail is unreachable and
  the tick list starts at 0 so no impossible lift is printed. The diagonal is
  undefined and drawn in grey so it cannot be read as lift = 1, which is
  nearly the same white;
* cell values are printed at the precision of the underlying table and nothing
  is thresholded, masked or reordered.

`F2` is the main-thesis candidate. `F2b` is appendix material; `F2c` and `F2d`
are analysis artifacts.
