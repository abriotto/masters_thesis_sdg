#!/bin/bash
#SBATCH --job-name=ft_role
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_role_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_role_%j.err
#SBATCH --partition=owner1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=thesis

set -euo pipefail

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

# conda activate <your_env>   # uncomment if the cluster needs it

nvidia-smi
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m src.finetuning.train_role_inference --train_path data/processed/jin2024_onuw/sft_role_inference_traced_31B/train.jsonl --val_path data/processed/jin2024_onuw/sft_role_inference_traced_31B/val.jsonl --output_dir models/finetuned/gemma-4-31B-role-inference-traced --response_part '<channel|>' --enable_thinking_in_prompt --max_seq_length 12288 --num_train_epochs 3 --smoke_test_samples 3