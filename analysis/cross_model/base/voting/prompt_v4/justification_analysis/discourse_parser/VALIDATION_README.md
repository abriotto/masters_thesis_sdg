# Manual validation — all rounds complete

Four review rounds were run, all in `notebooks/04_justification_analysis/`.
**All four are closed and no further manual validation is planned.**

Every sample was purposively stratified, so results are reported as **raw
counts**: no confidence intervals, no population precision, and **no recall**
(the justifications are not exhaustively gold-annotated).

---

## 1. 50-case validation of standard discopy — DONE

`3_manual_discourse_validation.ipynb` — the **primary** validation of the
production pipeline.

- sample : `manual_validation_sample_50.csv`
- answers: `manual_validation_completed.csv` (50/50)
- report : `manual_validation_report.txt`, errors in `manual_error_table.csv`

| | result |
|---|---|
| connective identification | 30/30 |
| top-level sense | 29/30 |
| `rejected_nosense` judged valid (false rejections) | 1/10 |
| `not_enumerated` judged valid (coverage misses) | 3/10 |

Only error: `#26 further` → Temporal, should be Expansion. `#7 when` was a
level-2 error only.

Regenerate the report:

```bash
"C:/Users/annab/miniconda3/envs/sdglogs/python.exe" src/justification_analysis/validation/evaluate_manual_validation.py --csv analysis/cross_model/base/voting/prompt_v4/justification_analysis/discourse_parser/manual_validation_completed.csv
```

## 2. 30-case coverage inspection — DONE

`4_coverage_gap_inspection.ipynb`

- sample : `coverage_inspection_sample_30.csv`
- answers: `coverage_inspection_completed.csv` (30/30)
- flags  : `coverage_inspection_flags.csv` (3 internally inconsistent rows,
  deliberately **not** corrected)
- summary: `thesis_tables/10b_coverage_inspection_by_form.csv`

| form | valid / reviewed |
|---|---|
| given | 10/10 |
| given that | 6/6 |
| despite | 1/7 |
| eventually | 0/4 |
| with | 0/3 |

Targeted diagnostic over out-of-inventory forms, drawn from the plausible tail
of a heuristic. **Not** a second accuracy estimate.

## 3. 30-case forced `given` / `given that` validation — DONE

`5_forced_given_validation.ipynb`

- sample : `forced_span_probe/forced_span_validation_sample_30.csv`
- answers: `forced_span_probe/forced_span_validation_completed.csv`

`given` accepted 10/10 valid, `given` NoSense 0/10 (all correctly rejected),
`given that` accepted 9/9, one false rejection. Top-level sense correct 19/20.

## 4. 25-case hybrid validation — DONE, hybrid rejected

`6_hybrid_candidate_validation.ipynb`

- sample : `experimental_hybrid/hybrid_validation_sample.csv`
- answers: `experimental_hybrid/hybrid_validation_completed.csv` (23/25
  answered; `probe_id` 1525 and 1003 left blank)

Only **4/18** accepted forced spans were genuine connectives — `eventually`
0/5, `with` 0/4, `upon` 0/2, `without` 0/2 — while 0/5 NoSense controls were
wrongly rejected. This is the evidence on which the DiMLex-expanded hybrid was
rejected.

---

## Notes

- No sheet is pre-annotated, and each round writes to its own answer file; the
  50-case results are untouched by rounds 2–4.
- There is no HTML validation tool. Validation is Jupyter only.
