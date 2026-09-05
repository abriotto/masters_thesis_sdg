#!/bin/bash
#SBATCH --job-name=dv_train_E4B
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_train_E4B_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_train_E4B_%j.err
#SBATCH --account=thesis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=owner1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Derivation-trace QLoRA finetune, E4B.
#
# --response_part is the ASSISTANT TURN MARKER, not '<channel|>'. The completion is
# derivation + answer and nothing is masked; only the prompt is. This is the change
# from the previous run, which supervised the ~30-token answer alone. Verified with
#   python -m src.finetuning.derivation.inspect_loss_span --model_name <model>
# the supervised span starts at "Night actions, in call order:". The Dealt cards
# block is in the PROMPT, not the target - supervising it would train copying.
#
# $'...' is bash ANSI-C quoting: it turns \n into a real newline, which is what the
# marker contains. Plain '<|turn>model\n' would pass a literal backslash-n and the
# marker would never be found.
#
# MAX_SEQ=8192. Sequences are ~3k tokens; 8192 is roughly 2.7x headroom over the
# chars-per-token ESTIMATE, which is the point - the estimate had only ~36% margin
# at 4096 and silent truncation of the completion is undetectable afterwards.
# Not set higher: training does no generation, so extra context is wasted memory,
# and that matters for 31B on a 48GB card.
# STEP 0 below turns a bad guess into a hard failure instead of a quiet truncation.

set -euo pipefail
cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

MODEL="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
DATA="data/processed/jin2024_onuw/sft_derivation_v1"
OUT="models/finetuned/gemma-4-E4B-derivation-v1"
MAX_SEQ=8192

mkdir -p results/finetuning logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi

echo
echo "=== STEP 0: length guard - abort rather than truncate ==="
# token_stats exits non-zero if ANY example exceeds --limit. With `set -e` that
# aborts the job here, before training. Silent truncation of the completion is the
# one failure that cannot be detected after the fact: the adapter would simply have
# been trained on a derivation with its tail cut off.
python -m src.finetuning.derivation.token_stats \
  --model_name "$MODEL" \
  --limit "$MAX_SEQ" \
  --out "results/finetuning/token_stats_E4B.json"

echo
echo "=== STEP 1: inspect_only - what does the loss actually cover? ==="
python -m src.finetuning.train_role_inference \
  --model_name "$MODEL" \
  --train_path "$DATA/train.jsonl" \
  --val_path "$DATA/val.jsonl" \
  --response_part $'<|turn>model\n' \
  --max_seq_length "$MAX_SEQ" \
  --inspect_only

echo
echo "=== STEP 2: train ==="
python -m src.finetuning.train_role_inference \
  --model_name "$MODEL" \
  --train_path "$DATA/train.jsonl" \
  --val_path "$DATA/val.jsonl" \
  --output_dir "$OUT" \
  --response_part $'<|turn>model\n' \
  --max_seq_length "$MAX_SEQ" \
  --num_train_epochs 3 \
  --seed 3407

echo
echo "=== done. adapters (one per epoch) in $OUT ==="
ls -la "$OUT"
