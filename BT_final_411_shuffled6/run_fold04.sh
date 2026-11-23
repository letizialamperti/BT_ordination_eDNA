#!/bin/bash
#OAR -n BT411-final-shuf6-f04
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout BT411-final-shuf6-f04-%jobid%.out
#OAR --stderr BT411-final-shuf6-f04-%jobid%.err
#OAR --project pr-qiepb

set -euo pipefail
BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BT_REPO=/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA
SAMPLES_DIR=/bettik/PROJECTS/pr-qiepb/lampertl/all
OUTPUT_ROOT="$BT_REPO/runs_bt_final_411_shuffled6"
FOLD=4
VIEWS=final-411-fold04-shuffled6

source "$BUNDLE/job_logging.sh"
source /applis/environments/conda.sh
conda activate zioboia
export PYTHONPATH="$BT_REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export BT_SHUFFLE_SCRATCH=/var/tmp

echo "Starting fold $FOLD: shuffled 6 epochs, seed 0, frozen 411 cohort"
python -u "$BUNDLE/train_final.py" \
    --fold "$FOLD" \
    --samples-dir "$SAMPLES_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --scratch-dir /var/tmp \
    --workers 12 --accelerator gpu

echo "Training finished; exporting only final embeddings for train and outer samples"
python -u "$BUNDLE/export_final.py" \
    --run-dir "$OUTPUT_ROOT/fold_04/seed_0" \
    --samples-dir "$SAMPLES_DIR" \
    --batch-size 64

echo "Fold $FOLD pipeline completed at $(date)"
