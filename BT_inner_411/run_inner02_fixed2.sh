#!/bin/bash
#OAR -n BT411-i02-fixed2
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout BT411-i02-fixed2-%jobid%.out
#OAR --stderr BT411-i02-fixed2-%jobid%.err
#OAR --project pr-qiepb

set -euo pipefail
BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BT_REPO=/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA
VIEWS=fixed
MAX_EPOCHS=2

source "$BUNDLE/job_logging.sh"
source /applis/environments/conda.sh
conda activate zioboia
export PYTHONPATH="$BT_REPO${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export BT_SHUFFLE_SCRATCH=/var/tmp

python -u "$BUNDLE/train_bt_inner_pilot.py" \
    --roles "$BUNDLE/splits/outer_01_inner_02/sample_roles.csv" \
    --samples_dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
    --output_root "$BT_REPO/runs_bt_inner_411" \
    --views "$VIEWS" \
    --max_epochs "$MAX_EPOCHS" \
    --sequence_length 300 --sample_subset_size 500 --batch_size 64 \
    --token_emb_dim 8 --sample_repr_dim 256 --sample_emb_dim 64 \
    --barlow_twins_lambda 0.005 --initial_learning_rate 0.0003 \
    --weight_decay 0.0001 --validation_chunks_per_sample 4 \
    --num_workers 12 --accelerator gpu --seed 0
