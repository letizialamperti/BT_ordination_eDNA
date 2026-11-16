#!/bi n/bash
#OAR -n barlow-twins-duration-bs64
#OAR -l /nodes=1/gpu=1,walltime=20:00:00
#OAR --stdout barlow-twins-duration-%jobid%.out
#OAR --stderr barlow-twins-duration-%jobid%.err
#OAR --project pr-qiepb

set -e
# Set the fold here before submitting the job.
FOLD=1
case "$FOLD" in
    1|2|3|4) ;;
    *) echo 'Fold must be 1, 2, 3 or 4' >&2; exit 1 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /applis/environments/conda.sh
conda activate zioboia

python training_BarlowTwins_duration.py \
    --outer_fold "$FOLD" \
    --samples_dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
    --split_dir k_cross_4 \
    --output_root runs_bt_duration_432samples \
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
