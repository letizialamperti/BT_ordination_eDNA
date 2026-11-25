#!/bin/bash
#OAR -n BT411-fixed2-export-f02
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout BT411-fixed2-export-f02-%jobid%.out
#OAR --stderr BT411-fixed2-export-f02-%jobid%.err
#OAR --project pr-qiepb
set -euo pipefail
BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BT_REPO=/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA
FOLD=2
VIEWS=export-411-fold02-fixed2
source "$BUNDLE/job_logging.sh"
source /applis/environments/conda.sh
conda activate zioboia
export PYTHONPATH="$BT_REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export BT_SHUFFLE_SCRATCH=/var/tmp
python -u "$BUNDLE/export_final.py" \
    --run-dir "$BT_REPO/runs_bt_final_411_fixed2/fold_02/seed_0" \
    --samples-dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
    --batch-size 64 --scratch-dir /var/tmp
