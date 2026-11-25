#!/bin/bash
#OAR -n BT411-test_fold02
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout BT411-test_fold02-%jobid%.out
#OAR --stderr BT411-test_fold02-%jobid%.err
#OAR --project pr-qiepb
set -euo pipefail
BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BT_REPO=/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA
VIEWS=ablation-test_fold02
source "$BUNDLE/job_logging.sh"
source /applis/environments/conda.sh
conda activate zioboia
export PYTHONPATH="$BT_REPO${PYTHONPATH:+:$PYTHONPATH}"
export BT_SHUFFLE_SCRATCH=/var/tmp
export WANDB_MODE=disabled
python -u "$BUNDLE/export_ablation_test.py" \
    --kind trained-test --fold 2 \
    --runs-root "$BT_REPO/runs_bt_final_411_shuffled6" \
    --samples-dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
    --output-root "$BT_REPO/embeddings_ablation_test_411" \
    --scratch-dir /var/tmp
