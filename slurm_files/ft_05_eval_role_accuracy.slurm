#!/bin/bash
#SBATCH --job-name=ft_eval_role
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_eval_role_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_eval_role_%j.err
#SBATCH --partition=owner1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --account=thesis

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# This card is shared. A 31B eval needs the whole 47 GiB; on 2026-08-30 another
# process held 25.8 GiB and the run OOMed after two full weight loads. Fail in
# seconds instead, so the job can simply be resubmitted onto a free GPU.
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free memory: ${FREE_MIB} MiB (need >= ${MIN_FREE_MIB})"
if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
  echo "ABORT: another process is using this GPU. Resubmit; do not lower the eval"
  echo "settings, since that would break comparability with the base arm."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi


# Role accuracy on the 21 held-out validation episodes, BASE vs finetuned.
#
# Submit one job per model:
#   for M in E2B E4B 31B; do sbatch --export=ALL,MODEL=$M slurm_files/ft_05_eval_role_accuracy.slurm; done
#
# Both arms of a model run in the same job, on the same GPU, with identical
# --val_path / --max_seq_length / --max_new_tokens, so the only difference
# between them is the adapter. The UNTRACED val set is used throughout, so
# every condition sees identical prompts and gold labels.
#
# Budget: at 10000 new tokens the 31B base arm put two episodes exactly on the
# cap, and those same two were its only parse failures -- truncated mid-thought,
# never reaching the closing marker or the answer. E2B and E4B never came close
# (longest output 3271). 16000 gives 31B room to terminate on its own; the
# smaller models are unaffected. max_seq_length has to rise with it: the val
# prompts run about 3100 tokens, so 12000 could not have held prompt plus
# generation, and 24576 leaves headroom over 3100 + 16000.

MODEL="${MODEL:-E2B}"
# ARM=base | ft | both. The base arms are already done, so a rerun after an
# OOM should pass ARM=ft rather than repeating a three-hour base pass.
ARM="${ARM:-both}"

case "$MODEL" in
  31B) BASE=unsloth/gemma-4-31B-it-unsloth-bnb-4bit ;;
  E4B) BASE=unsloth/gemma-4-E4B-it-unsloth-bnb-4bit ;;
  E2B) BASE=unsloth/gemma-4-E2B-it-unsloth-bnb-4bit ;;
  *)   echo "unknown MODEL=$MODEL (expected 31B, E4B or E2B)"; exit 1 ;;
esac

ADAPTER="models/finetuned/gemma-4-${MODEL}-role-inference-traced/final_adapter"
VAL="data/processed/jin2024_onuw/sft_role_inference/val.jsonl"

if [ ! -d "$ADAPTER" ]; then
  echo "adapter not found: $ADAPTER"; exit 1
fi

mkdir -p results/finetuning

if [ "$ARM" = "base" ] || [ "$ARM" = "both" ]; then
echo "############ $MODEL BASE (no adapter) ############"
python -m src.finetuning.evaluate_adapter \
  --model_name "$BASE" \
  --val_path "$VAL" \
  --max_seq_length 24576 \
  --max_new_tokens 16000 \
  --output_path "results/finetuning/eval_${MODEL}_base_val.json"
fi

if [ "$ARM" = "ft" ] || [ "$ARM" = "both" ]; then
echo "############ $MODEL FINETUNED (traced adapter) ############"
python -m src.finetuning.evaluate_adapter \
  --model_name "$BASE" \
  --adapter_path "$ADAPTER" \
  --val_path "$VAL" \
  --max_seq_length 24576 \
  --max_new_tokens 16000 \
  --output_path "results/finetuning/eval_${MODEL}_traced_val.json"
fi

echo "=== done: $MODEL ==="
