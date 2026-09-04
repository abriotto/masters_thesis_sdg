#!/bin/bash
#SBATCH --job-name=dv_base_diag
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_base_diag_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_base_diag_%j.err
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# ---------------- FILL THESE IN ----------------
#SBATCH --partition=FILL_IN_PARTITION
#SBATCH --time=FILL_IN_TIME
#SBATCH --gres=FILL_IN_GPU
#SBATCH --mem=FILL_IN_MEM
# -----------------------------------------------

# Base-model diagnostic, all three variants, on the 30 validation games, in two
# prompt conditions:
#
#   night       full prompt, private Moderator night messages included. This is the
#               finetuning condition; the derivation is deducible from it.
#   discussion  the same prompt with those messages removed. This is what an
#               external observer sees, and what the voting task shows.
#
# One generation per game per condition, decoding identical to the voting runs
# (temperature 1.0, top_p 0.95, top_k 64).
#
# Why it matters: if the base model already scores high in `night`, the finetune is
# teaching bookkeeping the model can already do, and the headline result must be
# read with that in mind. Run this BEFORE the finetunes so the interpretation is
# fixed in advance rather than after seeing the numbers.
#
# All three variants run sequentially in one job. Split into three jobs if your
# queue prefers shorter ones.

set -euo pipefail
cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

SPLIT="data/processed/jin2024_onuw/sft_derivation_v1/split.json"
mkdir -p results/finetuning

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi

for SPEC in \
  "E2B:unsloth/gemma-4-E2B-it-unsloth-bnb-4bit" \
  "E4B:unsloth/gemma-4-E4B-it-unsloth-bnb-4bit" \
  "31B:unsloth/gemma-4-31B-it-unsloth-bnb-4bit" ; do
  TAG="${SPEC%%:*}"
  MODEL="${SPEC#*:}"
  echo
  echo "############ $TAG ############"
  python -m src.finetuning.derivation.base_diagnostic \
    --model_name "$MODEL" \
    --split_path "$SPLIT" \
    --conditions night,discussion \
    --max_seq_length 8192 \
    --max_new_tokens 4096 \
    --output_path "results/finetuning/base_diagnostic_${TAG}.jsonl"
done

echo
echo "=== done. One JSONL per variant in results/finetuning/ ==="
