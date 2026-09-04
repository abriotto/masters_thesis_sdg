#!/bin/bash
#SBATCH --job-name=dv_train_E2B
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_train_E2B_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_train_E2B_%j.err
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# ---------------- FILL THESE IN ----------------
#SBATCH --partition=FILL_IN_PARTITION
#SBATCH --time=FILL_IN_TIME
#SBATCH --gres=FILL_IN_GPU
#SBATCH --mem=FILL_IN_MEM
# -----------------------------------------------

# Derivation-trace QLoRA finetune, E2B.
#
# --response_part is the ASSISTANT TURN MARKER, not '<channel|>'. The completion is
# derivation + answer and nothing is masked; only the prompt is. This is the change
# from the previous run, which supervised the ~30-token answer alone. Verified with
#   python -m src.finetuning.derivation.inspect_loss_span --model_name <model>
# the supervised span starts at "Dealt cards:".
#
# $'...' is bash ANSI-C quoting: it turns \n into a real newline, which is what the
# marker contains. Plain '<|turn>model\n' would pass a literal backslash-n and the
# marker would never be found.
#
# max_seq_length is set from the measured token distribution (dv_00_token_stats),
# not carried over from the old runs. Run that first.

set -euo pipefail
cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

MODEL="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
DATA="data/processed/jin2024_onuw/sft_derivation_v1"
OUT="models/finetuned/gemma-4-E2B-derivation-v1"
MAX_SEQ=4096

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi

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
