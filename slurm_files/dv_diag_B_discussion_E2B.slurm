#!/bin/bash
#SBATCH --job-name=dv_diag_B_discussion_E2B
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_diag_B_discussion_E2B_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_diag_B_discussion_E2B_%j.err
#SBATCH --account=thesis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=testing
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Base-model diagnostic. ONE variant, ONE condition, one job.
#
#   condition B = night actions removed, INITIAL DEAL KEPT. The model is still
#                 given the starting configuration and only has to apply what the
#                 players announce openly. This is NOT the external-observer
#                 setting - condition C is. Kept as the middle rung of the ladder.
#   variant     = E2B
#
# Split six ways, by variant as well as by condition, so 31B can be pinned to a
# large card without dragging E2B and E4B down with it if it OOMs. The previous
# single job ran all three variants sequentially and buffered every result to the
# end, so an OOM in the last variant would have destroyed the first two.
#
# NO --nodelist here: pass it at submit time, e.g.
#   sbatch --nodelist=vgpu8-0 slurm_files/dv_diag_B_discussion_E2B.slurm
#
# Results are appended and flushed PER GAME, so a walltime kill or an OOM costs
# only the game in flight.
#
# Prompts, decoding (temperature 1.0, top_p 0.95, top_k 64) and output paths are
# unchanged from the previous version.

set -euo pipefail
cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

TAG="E2B"
MODEL="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
CONDITION="discussion"
SPLIT="data/processed/jin2024_onuw/sft_derivation_v1/split.json"
OUT="results/finetuning/base_diagnostic_${TAG}_${CONDITION}.jsonl"

mkdir -p results/finetuning logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "=== variant $TAG, condition $CONDITION ==="
nvidia-smi

python -m src.finetuning.derivation.base_diagnostic \
  --model_name "$MODEL" \
  --split_path "$SPLIT" \
  --conditions "$CONDITION" \
  --max_seq_length 24576 \
  --max_new_tokens 14000 \
  --output_path "$OUT"

echo
echo "=== done. $OUT ==="
wc -l "$OUT"
