#!/bin/bash
#OAR -n BT-inner-export
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout BT-inner-export-%jobid%.out
#OAR --stderr BT-inner-export-%jobid%.err
#OAR --project pr-qiepb
set -euo pipefail
BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BT_REPO=/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA
VIEWS=fixed
CHECKPOINT=all
VIEWS_LOG="$VIEWS"
VIEWS="export-$VIEWS_LOG"
source "$BUNDLE/job_logging.sh"
source /applis/environments/conda.sh
conda activate zioboia
export PYTHONPATH="$BT_REPO${PYTHONPATH:+:$PYTHONPATH}"
VIEWS="$VIEWS_LOG"
python -u "$BUNDLE/export_inner_embeddings.py" \
    --run-dir "$BT_REPO/runs_bt_inner_pilot_io_retry/outer_01_inner_02/$VIEWS/seed_0" \
    --samples-dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
    --checkpoint "$CHECKPOINT"
