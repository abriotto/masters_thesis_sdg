#!/bin/bash
#SBATCH --job-name=ft_eval_role
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_eval_role_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_eval_role_%j.err
#SBATCH --partition=owner1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=thesis

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p results/finetuning

# Post-finetuning role accuracy on the held-out 21 val episodes. The UNTRACED val
# set is used for every condition so base and finetuned are scored on identical
# prompts and gold labels. Base numbers to beat: 31B 69.2% / 52.9% werewolf recall,
# E4B 54.3% / 14.6%. Majority-class baseline is 26.2%.

echo "############ 31B traced adapter ############"
python -m src.finetuning.evaluate_adapter \
  --model_name unsloth/gemma-4-31B-it-unsloth-bnb-4bit \
  --adapter_path models/finetuned/gemma-4-31B-role-inference-traced/final_adapter \
  --val_path data/processed/jin2024_onuw/sft_role_inference/val.jsonl \
  --max_seq_length 12000 --max_new_tokens 10000 \
  --output_path results/finetuning/eval_31B_traced_val.json

echo "############ E4B traced adapter ############"
python -m src.finetuning.evaluate_adapter \
  --model_name unsloth/gemma-4-E4B-it-unsloth-bnb-4bit \
  --adapter_path models/finetuned/gemma-4-E4B-role-inference-traced/final_adapter \
  --val_path data/processed/jin2024_onuw/sft_role_inference/val.jsonl \
  --max_seq_length 12000 --max_new_tokens 10000 \
  --output_path results/finetuning/eval_E4B_traced_val.json

echo "=== done ==="
