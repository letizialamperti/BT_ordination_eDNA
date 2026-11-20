#!/bin/bash
#OAR -n BT411-i03-shuffled6-export
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout BT411-i03-shuffled6-export-%jobid%.out
#OAR --stderr BT411-i03-shuffled6-export-%jobid%.err
#OAR --project pr-qiepb
set -euo pipefail
BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BT_REPO=/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA
VIEWS=shuffled
CHECKPOINTS=(initial epoch_06)
VIEWS_LOG="$VIEWS"
VIEWS="export-411-i03-$VIEWS_LOG"
source "$BUNDLE/job_logging.sh"
source /applis/environments/conda.sh
conda activate zioboia
export PYTHONPATH="$BT_REPO${PYTHONPATH:+:$PYTHONPATH}"
VIEWS="$VIEWS_LOG"
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    python -u "$BUNDLE/export_inner_embeddings.py" \
        --run-dir "$BT_REPO/runs_bt_inner_411/outer_01_inner_03/$VIEWS/seed_0" \
        --samples-dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
        --roles "$BUNDLE/splits/outer_01_inner_03/classifier_roles.csv" \
        --checkpoint "$CHECKPOINT"
done
