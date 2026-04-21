#!/bin/bash
#SBATCH --job-name=vote_qwen
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/vote_qwen_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/vote_qwen_%j.err
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=thesis

nvidia-smi

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1

python -m src.run_llm_votes\
  --backend hf_local \
  --model_name Qwen/Qwen2.5-7B-Instruct
  --index_path data/processed/lai2023/onuw_transcripts_ready/index_cleaned.json \
  --prompt_path src/prompts/vote_prompt_v4.txt \
  --prompt_version v4 \
  --task_name voting \
  --max_games 5 \
  --overwrite