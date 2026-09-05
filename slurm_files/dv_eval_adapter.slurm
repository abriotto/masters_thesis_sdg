#!/bin/bash
#SBATCH --job-name=dv_eval_adapter
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_eval_adapter_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_eval_adapter_%j.err
#SBATCH --account=thesis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=testing
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=FILL_IN_MEM

# Evaluate ONE trained adapter on the 30 validation games, condition A.
#
#   sbatch --nodelist=<NODE> slurm_files/dv_eval_adapter.slurm <TAG> <ADAPTER_PATH>
#
# e.g.
#   sbatch --nodelist=vgpu9-0 slurm_files/dv_eval_adapter.slurm \
#     31B models/finetuned/gemma-4-31B-derivation-v1/final_adapter
#
# TAG is E2B, E4B or 31B and selects the base checkpoint the adapter attaches to.
# ADAPTER_PATH is any directory the trainer wrote - final or a per-epoch checkpoint.
#
# Comparability with the base diagnostic is the whole point, so all three of the
# things that could move are pinned to the diagnostic's values:
#
#   prompt    built by base_diagnostic.build_condition_prompt from the val.jsonl
#             row, the same function and the same input the diagnostic used, so
#             the prompts are byte-identical per condition.
#   decoding  temperature 1.0, top_p 0.95, top_k 64, thinking enabled - hardcoded
#             in evaluate_adapter.generate, same as base_diagnostic.
#   budget    max_new_tokens 14000 and max_seq_length 24576, matching the
#             diagnostic rather than evaluate_adapter's own smaller defaults.
#
# CONDITION defaults to night. `public` strips the private Moderator night
# messages AND the "## Initial deal" section, via the same build_condition_prompt
# that base_diagnostic uses, so the number is comparable with the base public
# results. The condition is in the output filename, so runs never overwrite.
#
# Base per-player accuracies to compare against:
#   night    E2B 75.3%   E4B 84.7%   31B 100%
#   public               E4B 55.3%
#
# 31B 4-bit plus a 24576 KV cache will not fit the 23.5 GB A5000 on vgpu8-0.
# Put 31B on the A6000 or vgpu10.

set -euo pipefail
cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: sbatch $0 <TAG: E2B|E4B|31B> <ADAPTER_PATH> [CONDITION: night|public]" >&2
  exit 2
fi

TAG="$1"
ADAPTER="$2"
CONDITION="${3:-night}"

case "$CONDITION" in
  night|public) ;;
  *) echo "unknown CONDITION '$CONDITION' - expected night or public" >&2; exit 2 ;;
esac

case "$TAG" in
  E2B) MODEL="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit" ;;
  E4B) MODEL="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit" ;;
  31B) MODEL="unsloth/gemma-4-31B-it-unsloth-bnb-4bit" ;;
  *) echo "unknown TAG '$TAG' - expected E2B, E4B or 31B" >&2; exit 2 ;;
esac

if [ ! -d "$ADAPTER" ]; then
  echo "adapter path does not exist: $ADAPTER" >&2
  exit 2
fi

VAL="data/processed/jin2024_onuw/sft_derivation_v1/val.jsonl"
STAMP=$(basename "$ADAPTER")
# CONDITION is in the filename so a public run cannot overwrite a night run.
OUT="results/finetuning/adapter_eval_${TAG}_${CONDITION}_${STAMP}.json"

mkdir -p results/finetuning logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "=== variant $TAG, condition $CONDITION, adapter $ADAPTER ==="
nvidia-smi

python -m src.finetuning.evaluate_adapter \
  --model_name "$MODEL" \
  --adapter_path "$ADAPTER" \
  --val_path "$VAL" \
  --condition "$CONDITION" \
  --max_seq_length 24576 \
  --max_new_tokens 14000 \
  --limit -1 \
  --output_path "$OUT"

echo
echo "=== done. $OUT ==="
echo "=== compare the per-player line against the base for THIS condition ==="
echo "===   night   E2B 75.3%  E4B 84.7%  31B 100% ==="
echo "===   public             E4B 55.3% ==="
