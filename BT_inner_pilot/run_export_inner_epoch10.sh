#!/bin/bash
#OAR -n BT-inner-export-epoch10
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout BT-inner-export-epoch10-%jobid%.out
#OAR --stderr BT-inner-export-epoch10-%jobid%.err
#OAR --project pr-qiepb

set -euo pipefail
BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BT_REPO=/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA
CHECKPOINT=epoch_10
VIEWS=export-epoch10-both

source "$BUNDLE/job_logging.sh"
source /applis/environments/conda.sh
conda activate zioboia
export PYTHONPATH="$BT_REPO${PYTHONPATH:+:$PYTHONPATH}"

status=0
for MODE in fixed shuffled; do
    RUN_DIR="$BT_REPO/runs_bt_inner_pilot_io_retry/outer_01_inner_02/$MODE/seed_0"
    echo "Starting export: $MODE / $CHECKPOINT at $(date)"
    if python -u "$BUNDLE/export_inner_embeddings.py" \
        --run-dir "$RUN_DIR" \
        --samples-dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
        --checkpoint "$CHECKPOINT"; then
        echo "Completed export: $MODE / $CHECKPOINT at $(date)"
    else
        export_status=$?
        echo "Export failed: $MODE / $CHECKPOINT (exit $export_status)" >&2
        status=$export_status
    fi
done
exit "$status"
