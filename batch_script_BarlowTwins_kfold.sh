#!/bin/bash
#OAR -n barlow-twins-kfold
#OAR -l /nodes=1/gpu=1/core=12,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout barlow-twins-kfold-%j.out
#OAR --stderr barlow-twins-kfold-%j.err
#OAR --project pr-qiepb
set -euo pipefail
# Run from the repository root. Supply the fold as the first argument.
FOLD="${1:?Usage: bash batch_script_BarlowTwins_kfold.sh 1}"
: "${BT_SAMPLES_DIR:?Set BT_SAMPLES_DIR}"
: "${BT_REPR_DIM:?Set the verified representation dimension}"
: "${BT_EMB_DIM:?Set the verified output embedding dimension}"
: "${BT_LAMBDA:?Set the verified BT lambda}"
: "${BT_LEARNING_RATE:?Set the verified learning rate}"
: "${BT_MAX_EPOCHS:?Set the epoch budget before inspecting outer results}"
source /applis/environments/conda.sh
conda activate zioboia
python training_BarlowTwins_kfold.py \
    --outer_fold "$FOLD" \
    --samples_dir "$BT_SAMPLES_DIR" \
    --sample_repr_dim "$BT_REPR_DIM" \
    --sample_emb_dim "$BT_EMB_DIM" \
    --barlow_twins_lambda "$BT_LAMBDA" \
    --initial_learning_rate "$BT_LEARNING_RATE" \
    --max_epochs "$BT_MAX_EPOCHS" \
    --output_root "${BT_OUTPUT_ROOT:-runs_bt_kfold}" \
    --seed "${BT_SEED:-0}" \
    --sequence_length 300 --sample_subset_size 500 --batch_size 8 --num_workers 12
