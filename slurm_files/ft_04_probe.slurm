#!/bin/bash
#SBATCH --job-name=ft_role
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_role_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft_role_%j.err
#SBATCH --partition=testing
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



python -m src.voting.run_llm_votes --prompt_path src/prompts/voting_prompt_v4.txt --prompt_version prompt_v4 --model_name unsloth/gemma-4-E4B-it-unsloth-bnb-4bit --adapter_path models/finetuned/gemma-4-E4B-role-inference-traced/final_adapter --gemma_enable_thinking --max_seq_length 24576 --max_new_tokens 14000 --max_games 5 --save_internal_thoughts --save_prompt