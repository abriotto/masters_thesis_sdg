#!/bin/bash
#SBATCH --job-name=dv_diag_B_discussion
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_diag_B_discussion_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/dv_diag_B_discussion_%j.err
#SBATCH --account=thesis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=testing
#SBATCH --nodelist=vgpu9-0
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=FILL_IN_MEM

# Base-model diagnostic, CONDITION B: night actions REMOVED from the prompt.
# This is what an external observer sees, and what the voting task shows. The
# derivation is NOT deducible from it; the final configuration can only be
# inferred from what players say. Expect this to be near chance - that is the
# honest baseline, not a failure.
#
# Condition A (night actions included) is a separate submission, dv_diag_A_night.slurm,
# so the two can run on different nodes and a failure in one does not lose the other.
#
# All three variants run in this job, sequentially, on one GPU.
#
# max_new_tokens=14000: base models with thinking enabled overrun, and 31B produced
# non-terminations at 16000 on the previous role-inference eval. A generation that
# hits the cap is an unparseable row, so the budget is deliberately generous.
# max_seq_length=24576 leaves room for prompt (~3k) plus that whole generation.
#
# Non-terminating and unparseable generations are recorded and reported on their own
# lines, and EXCLUDED from the accuracy denominator. They are not scored as wrong.

set -euo pipefail
cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

SPLIT="data/processed/jin2024_onuw/sft_derivation_v1/split.json"
mkdir -p results/finetuning logs

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi

for SPEC in \
  "E2B:unsloth/gemma-4-E2B-it-unsloth-bnb-4bit" \
  "E4B:unsloth/gemma-4-E4B-it-unsloth-bnb-4bit" \
  "31B:unsloth/gemma-4-31B-it-unsloth-bnb-4bit" ; do
  TAG="${SPEC%%:*}"
  MODEL="${SPEC#*:}"
  echo
  echo "############ $TAG - condition discussion ############"
  python -m src.finetuning.derivation.base_diagnostic \
    --model_name "$MODEL" \
    --split_path "$SPLIT" \
    --conditions discussion \
    --max_seq_length 24576 \
    --max_new_tokens 14000 \
    --output_path "results/finetuning/base_diagnostic_${TAG}_discussion.jsonl"
done

echo
echo "=== done. results/finetuning/base_diagnostic_{E2B,E4B,31B}_discussion.jsonl ==="
