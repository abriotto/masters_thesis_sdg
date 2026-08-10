#!/bin/bash
#SBATCH --job-name=ft01_gemma
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft01_gemma_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/ft01_gemma_%j.err
#SBATCH --partition=owner1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --account=thesis

cd /home/abriotto/Tesi/masters_thesis_sdg || exit 1



echo "=== Job $SLURM_JOB_ID on $(hostname) ==="

echo
echo "=== STEP 1: dependency check ==="
python - <<'PY'
import importlib
for name in ["torch", "transformers", "trl", "peft", "datasets", "unsloth"]:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:14s} {getattr(mod, '__version__', 'unknown')}")
    except ImportError as exc:
        print(f"  {name:14s} MISSING  ({exc})")
PY

echo
echo "=== STEP 2: dry run (tokenizer only) ==="
python -m src.finetuning.train_role_inference --dry_run

echo
echo "=== dry run finished ==="
echo "Check above: chat template resolved, response marker found,"
echo "token count under max_seq_length, and whether thought markers appear."
