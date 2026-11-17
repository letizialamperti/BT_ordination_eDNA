#!/bin/bash
#OAR -n barlow-twins-duration-bs64
#OAR -l /nodes=1/gpu=1,walltime=48:00:00
#OAR --stdout barlow-twins-duration-%jobid%.out
#OAR --stderr barlow-twins-duration-%jobid%.err
#OAR --project pr-qiepb

set -e
# Set the fold here before submitting the job.
FOLD=1
# Run both continuations, or select fixed or shuffled.
VIEWS=both
case "$FOLD" in
    1|2|3|4) ;;
    *) echo 'Fold must be 1, 2, 3 or 4' >&2; exit 1 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Directory containing the completed one-epoch training runs.
CHECKPOINT_ROOT=runs_bt_432samples/cap_0
FOLD_NAME=$(printf 'fold_%02d' "$FOLD")
RESUME_CHECKPOINT="$CHECKPOINT_ROOT/$FOLD_NAME/seed_0/final.ckpt"

source /applis/environments/conda.sh
conda activate zioboia

case "$VIEWS" in
    both) VIEW_MODES=(fixed shuffled) ;;
    fixed|shuffled) VIEW_MODES=("$VIEWS") ;;
    *) echo 'VIEWS must be fixed, shuffled or both' >&2; exit 1 ;;
esac

for VIEW_MODE in "${VIEW_MODES[@]}"; do
    python training_BarlowTwins_duration.py \
        --views "$VIEW_MODE" \
        --resume_checkpoint "$RESUME_CHECKPOINT" \
        --outer_fold "$FOLD" \
        --samples_dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
        --split_dir k_cross_4 \
        --output_root "runs_bt_duration_432samples/$VIEW_MODE" \
        --sequence_length 300 \
        --sample_subset_size 500 \
        --batch_size 64 \
        --token_emb_dim 8 \
        --sample_repr_dim 256 \
        --sample_emb_dim 64 \
        --max_chunks_per_sample 0 \
        --barlow_twins_lambda 0.005 \
        --initial_learning_rate 0.0003 \
        --weight_decay 0.0001 \
        --max_epochs 5 \
        --num_workers 12 \
        --accelerator gpu \
        --seed 0
done
