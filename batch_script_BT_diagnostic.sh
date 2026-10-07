#!/bin/bash
#OAR -n bt-diagnostic
#OAR -l /nodes=1/gpu=1,walltime=00:10:00
#OAR -t devel
#OAR -p gpumodel='A100'
#OAR --stdout bt-diagnostic-%j.out
#OAR --stderr bt-diagnostic-%j.err
#OAR --project pr-qiepb
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /applis/environments/conda.sh
conda activate zioboia
python -u diagnostic_BarlowTwins.py \
    --samples_dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
    --outer_fold 1 --max_steps 20 --sample_count 8 --num_workers 12
