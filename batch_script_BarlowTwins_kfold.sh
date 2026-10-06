#!/bin/bash
#OAR -n barlow-twins-kfold
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout barlow-twins-kfold-%j.out
#OAR --stderr barlow-twins-kfold-%j.err
#OAR --project pr-qiepb
set -e
# Defaults to fold 1; pass 2, 3 or 4 to run another fold.
FOLD="${1:-1}"
case "$FOLD" in 1|2|3|4) ;; *) echo 'Fold must be 1, 2, 3 or 4' >&2; exit 1;; esac
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /applis/environments/conda.sh
conda activate zioboia
python training_BarlowTwins_kfold.py \
    --outer_fold "$FOLD" \
    --samples_dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
    --split_dir k_cross \
    --output_root runs_bt_fixed \
    --sequence_length 300 \
    --sample_subset_size 500 \
    --batch_size 32 \
    --token_emb_dim 8 \
    --sample_repr_dim 256 \
    --sample_emb_dim 64 \
    --barlow_twins_lambda 0.005 \
    --initial_learning_rate 0.001 \
    --weight_decay 0.0001 \
    --max_epochs 1 \
    --num_workers 12 \
    --accelerator gpu \
    --seed 0
