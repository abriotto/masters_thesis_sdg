# Runbook: derivation-trace finetuning

Everything here runs on the Slurm cluster. Fill in `FILL_IN_PARTITION`,
`FILL_IN_TIME`, `FILL_IN_GPU` and `FILL_IN_MEM` in each script before submitting;
they are the only placeholders.

Repo path assumed on the cluster: `/home/abriotto/Tesi/masters_thesis_sdg`.

## Prerequisite: the split does not exist yet

`slurm_files/dv_train_*.slurm` and `dv_base_diagnostic.slurm` read

```
data/processed/jin2024_onuw/sft_derivation_v1/train.jsonl
data/processed/jin2024_onuw/sft_derivation_v1/val.jsonl
data/processed/jin2024_onuw/sft_derivation_v1/split.json
```

Only `all.jsonl` (120 games) exists so far. **The 90/30 stratified split is step 7
and has not been built** — do not submit anything below until it has, or the jobs
will fail on a missing file.

## Order

### 0. Token lengths — no GPU

```bash
sbatch slurm_files/dv_00_token_stats.slurm
```

Writes `results/finetuning/token_stats_{E2B,E4B,31B}.json`.

**Check:** every variant reports `games over 4096 : 0`. If any reports more, raise
`MAX_SEQ` in all three `dv_train_*.slurm` to clear the reported max. Do not filter
the long games out — that is what silently shrank the previous dataset.

Estimated from characters at ~4 chars/token, the max total is ~3,015 tokens, so
4096 should hold with room. This job replaces that estimate with real counts.

### 1. Base diagnostic — GPU

```bash
sbatch slurm_files/dv_base_diagnostic.slurm
```

Writes `results/finetuning/base_diagnostic_{E2B,E4B,31B}.jsonl`, one row per game
per condition, with the raw generation kept.

**Run this before the finetunes.** It fixes the interpretation in advance instead
of after seeing the training result.

**Check** the summary table at the end of the log:

```
night        per-player NNN/150 = NN.N%   exact-match NN/30   unparsed N   truncated N
discussion   per-player NNN/150 = NN.N%   exact-match NN/30   unparsed N   truncated N
```

- `night` high (say >80% per-player) means the base model can already do the
  bookkeeping when handed the night actions. The finetune is then teaching format
  and reliability, not the deduction — say so in the writeup.
- `discussion` near chance is expected and is the honest baseline for the voting
  task, which never shows night actions.
- `unparsed` should be low. If it is high, the base model is ignoring the output
  format rather than getting the answer wrong; those two failures mean different
  things and the raw responses are in the JSONL to tell them apart.
- `truncated` should be 0. If not, raise `--max_new_tokens`.

### 2. Finetune, one job per variant — GPU

```bash
sbatch slurm_files/dv_train_E2B.slurm
sbatch slurm_files/dv_train_E4B.slurm
sbatch slurm_files/dv_train_31B.slurm
```

Writes adapters to `models/finetuned/gemma-4-{E2B,E4B,31B}-derivation-v1`, one per
epoch (`save_strategy="epoch"` is hardcoded in the trainer). `models/` is
gitignored — the adapters stay on the cluster.

Each script runs `--inspect_only` first, then trains, in the same job.

**Check, in the `inspect_only` section, before trusting the run:**

- the supervised span starts at `Dealt cards:`, not at `Final configuration:`.
  This is the whole point of the rebuild. If the loss starts at the answer, the
  `--response_part` did not take.
- the masked region ends with the end of the transcript.
- token count is under `max_seq_length` with no truncation warning.

Then in the training section: loss decreasing, and three adapter directories at
the end.

## Verifying the loss span locally

The same check runs on CPU with only a tokeniser:

```bash
python -m src.finetuning.derivation.inspect_loss_span --model_name unsloth/gemma-4-E2B-it --game_id episode_002
```

Without `--model_name` it prints a marker-level view that needs no tokeniser, which
is all that runs on the laptop (HuggingFace is unreachable there — TLS interception).

## What produces the data

| Command | Writes |
|---|---|
| `python -m src.finetuning.derivation.test_simulation` | nothing; 7 test groups must pass |
| `python -m src.finetuning.derivation.gate` | nothing; must report 120/120 |
| `python -m src.finetuning.derivation.render` | nothing; prints five sample traces |
| `python -m src.finetuning.derivation.build_dataset --write` | `sft_derivation_v1/all.jsonl` |

`build_dataset` refuses to run if any game fails the gate.

## Decoding config

The diagnostic uses `temperature=1.0, top_p=0.95, top_k=64`, matching
`src/voting/run_llm_votes.py` and `generate_traces.py`, so the numbers are
comparable with the voting experiments.
