# Runbook: derivation-trace finetuning

Repo path on the cluster: `/home/abriotto/Tesi/masters_thesis_sdg`.

**One placeholder left in every script: `#SBATCH --mem=FILL_IN_MEM`.** Partition,
node, wall clock and GPU are filled in. Nothing uses `$TMPDIR` or node-local
scratch: every path is repo-relative on the shared filesystem, so all outputs
survive job exit.

## Submission order

```bash
# 1. Diagnostics - two nodes, in parallel, independent of each other
sbatch slurm_files/dv_diag_A_night.slurm
sbatch slurm_files/dv_diag_B_discussion.slurm

# 2. Training - owner1, one node, sequential. afterany, not afterok, so a
#    failure in one does not block the rest.
JID=$(sbatch --parsable slurm_files/dv_train_E2B.slurm)
JID=$(sbatch --parsable --dependency=afterany:$JID slurm_files/dv_train_E4B.slurm)
sbatch --dependency=afterany:$JID slurm_files/dv_train_31B.slurm
```

## Job table

| Job | Partition / node | Wall | Writes | Check in the log |
|---|---|---|---|---|
| `dv_diag_A_night` | testing / vgpu8-0 | 6h | `results/finetuning/base_diagnostic_{E2B,E4B,31B}_night.jsonl` | `non-terminating : 0`. If not, raise `--max_new_tokens` and rerun; those rows are unscored, not wrong. |
| `dv_diag_B_discussion` | testing / vgpu9-0 | 6h | `results/finetuning/base_diagnostic_{E2B,E4B,31B}_discussion.jsonl` | Same. Low accuracy here is the expected result, not a failure. |
| `dv_train_E2B` | owner1 | 4h | `models/finetuned/gemma-4-E2B-derivation-v1/` | STEP 1: supervised span starts at `Night actions, in call order:` and contains no `Dealt cards:`. |
| `dv_train_E4B` | owner1 | 4h | `models/finetuned/gemma-4-E4B-derivation-v1/` | Same. |
| `dv_train_31B` | owner1 | 8h | `models/finetuned/gemma-4-31B-derivation-v1/` | Same. |

Three distinct adapter directories, one per variant, all repo-relative. One epoch
checkpoint each, `save_strategy="epoch"` hardcoded in the trainer.

Optional, no GPU: `sbatch slurm_files/dv_00_token_stats.slurm` writes
`results/finetuning/token_stats_{E2B,E4B,31B}.json`. The identical guard runs as
STEP 0 inside every training job, so this is only useful for seeing the numbers
before queueing.

## The length guard

`MAX_SEQ=8192` in all three training scripts. Sequences are ~3k tokens; 8192 is
real headroom over a chars-per-token *estimate* that had only ~36% margin at 4096.
Not higher: training does no generation, so extra context is wasted memory, and
that matters for 31B on a 48GB card.

STEP 0 of every training job runs `token_stats --limit 8192`, which exits non-zero
if any example exceeds it. With `set -euo pipefail` the job aborts there, before
the model loads. **Silent truncation of the completion is the one failure that
cannot be detected afterwards** - the adapter would have been trained on a
derivation with its tail cut off - so it is turned into a hard failure. On failure:

```
*** N game(s) exceed max_seq_length=8192. RAISE IT - do not filter them out.
```

## Reading the diagnostic

```
night  (30 generations)
   scored          : 30
   non-terminating : 0   (hit max_new_tokens=14000)
   unparseable     : 0   (finished, no Final configuration block)
   per-player      : 128/150 = 85.3%
   exact-match     : 21/30 = 70.0%
```

Accuracy is over **scored** generations only. A generation that hit the token cap
never finished, and one that finished without a `Final configuration` block said
nothing about the roles. Neither is a wrong answer, and scoring them 0/5 would
understate accuracy while hiding a decoding-budget problem. Both are recorded per
row as `non_terminating`, `parsed_ok` and `scored`, with the raw text kept.

`max_new_tokens=14000`, `max_seq_length=24576`: base models with thinking enabled
overrun, and 31B produced non-terminations at 16000 on the previous role-inference
eval.

**A high `night` score is a real finding, not a null result.** It would mean the
base model can already do the bookkeeping when handed the night actions, and the
headline comparison has to be read that way. That is why the diagnostics run first.

## The data

```
data/processed/jin2024_onuw/sft_derivation_v1/split.json    90 train / 30 val ids
data/processed/jin2024_onuw/sft_derivation_v1/train.jsonl   90 games
data/processed/jin2024_onuw/sft_derivation_v1/val.jsonl     30 games
data/processed/jin2024_onuw/sft_derivation_v1/all.jsonl     all 120
```

Stratified by end-game Werewolf count, seed 1234, no overlap:

| Werewolves | corpus | train | val | val share |
|---|---|---|---|---|
| 0 | 3 | 2 | 1 | 33.3% |
| 1 | 77 | 58 | 19 | 24.7% |
| 2 | 40 | 30 | 10 | 25.0% |
| **total** | **120** | **90** | **30** | **25.0%** |

`split.json` is the authority; the JSONLs are materialised from its id lists, not
from a fresh shuffle. Regenerate with
`python -m src.finetuning.derivation.split --write`. Deterministic.

**Note:** episode_031, the corpus's only Robber decline, is in *validation*. The
model never sees that pattern in training, so a decline-shaped error at eval is
unsurprising and not a general failure.

## What the target is

The prompt carries the instruction, the rules, the player list, the **initial deal
and centre**, and the full transcript **including the private Moderator night
messages**. The completion is the night actions plus the final configuration.

The `Dealt cards:` block is in the prompt and not in the target: supervising a
verbatim copy of prompt text trains copying. The complete three-section rendering
is kept per row as `full_trace` for the appendix.

Loss covers the whole completion. The anchor is the assistant turn marker, not
`<channel|>`. Verify without a GPU:

```bash
python -m src.finetuning.derivation.inspect_loss_span --game_id episode_002
```

## Local checks, no GPU

| Command | Expect |
|---|---|
| `python -m src.finetuning.derivation.test_simulation` | 7/7 test groups pass |
| `python -m src.finetuning.derivation.gate` | 120/120, gate passed |
| `python -m src.finetuning.derivation.render` | five sample traces |
| `python -m src.finetuning.derivation.build_dataset --write` | rewrites `all.jsonl` |

`build_dataset` refuses to run if any game fails the gate.

## Decoding config

The diagnostic uses `temperature=1.0, top_p=0.95, top_k=64`, matching
`src/voting/run_llm_votes.py` and `generate_traces.py`, so the numbers are
comparable with the voting experiments.
