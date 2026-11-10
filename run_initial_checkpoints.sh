#!/bin/bash
#OAR -n BT432-initial-checkpoints
#OAR -l /nodes=1,walltime=00:30:00
#OAR --stdout BT432-initial-%jobid%.out
#OAR --stderr BT432-initial-%jobid%.err
#OAR --project pr-qiepb

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source /applis/environments/conda.sh
conda activate zioboia

BT_REPO="${BT_REPO:-/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA}"
export PYTHONPATH="$BT_REPO${PYTHONPATH:+:$PYTHONPATH}"
python generate_initial_checkpoints.py \
    --trained_root "$BT_REPO/runs_bt_432samples" \
    --output_root "$BT_REPO/runs_bt_432samples_initial" \
    "$@"
