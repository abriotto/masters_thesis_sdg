#!/bin/bash
#SBATCH --job-name=just_annot
#SBATCH --output=/home/abriotto/Tesi/masters_thesis_sdg/logs/just_annot_%j.out
#SBATCH --error=/home/abriotto/Tesi/masters_thesis_sdg/logs/just_annot_%j.err
#SBATCH --partition=owner1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --account=thesis

# No --gres=gpu here on purpose. This job only makes HTTPS calls to the
# DeepSeek API: it spends its whole wall time waiting on the network, and
# holding a GPU idle for hours would be wasted allocation. If owner1 is
# GPU-only on this cluster, switch --partition to a CPU queue; nothing else
# in this script needs to change.

set -euo pipefail

REPO=/home/abriotto/Tesi/masters_thesis_sdg
cd "$REPO" || exit 1

mkdir -p "$REPO/logs"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "started: $(date --iso-8601=seconds)"

# ------------------------------------------------------------
# API key
#
# The key must never be written into this file -- it is in git. Keep it in a
# private file on the cluster instead:
#
#   printf 'export DEEPSEEK_API_KEY=sk-...\n' > ~/.deepseek_env
#   chmod 600 ~/.deepseek_env
#
# Then this script picks it up automatically. Fail before submitting work
# rather than burning a queue slot to discover the key is missing.
# ------------------------------------------------------------

if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "$HOME/.deepseek_env" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/.deepseek_env"
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set and ~/.deepseek_env was not found." >&2
    exit 1
fi

echo "API key: present (${#DEEPSEEK_API_KEY} chars)"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

# conda activate <your_env>   # uncomment if the cluster needs it
PYTHON=${PYTHON:-python}

$PYTHON -c "import openai, pandas; print('openai', openai.__version__, '| pandas', pandas.__version__)"

# ------------------------------------------------------------
# Sample and annotate
#
# --resume makes the annotation step idempotent: it appends only
# justifications that are not already in the output file with a successful
# annotation. So if the job hits the time limit or is requeued, resubmitting
# picks up where it stopped instead of paying for every call again.
# ------------------------------------------------------------

# SCHEMA selects the prompt, the allowed categories and the output folder.
# Override at submit time without editing this file:
#   sbatch --export=ALL,SCHEMA=v1 slurm_files/justification_annotation_pilot.slurm
SCHEMA=${SCHEMA:-v2}

PILOT_DIR=results/justification_annotation/pilot_${SCHEMA}
SAMPLE=$PILOT_DIR/pilot_sample.jsonl

echo "Schema: $SCHEMA  ->  $PILOT_DIR"
echo

if [[ ! -f "$SAMPLE" ]]; then
    echo "=== Drawing pilot sample ==="
    $PYTHON src/pt_annotation/sample_justification_pilot.py --schema "$SCHEMA"
else
    echo "=== Pilot sample already drawn, reusing it ==="
    echo "    (delete $SAMPLE to redraw; the seed makes it reproducible, and"
    echo "     identical across schemas, so v1 and v2 annotate the same 40)"
fi

echo
echo "=== Annotating with DeepSeek ==="
$PYTHON src/pt_annotation/justification_annotation_deepseek.py --schema "$SCHEMA" --resume

echo
echo "Review the result in notebooks/05_annotation_val/."

echo
echo "finished: $(date --iso-8601=seconds)"
