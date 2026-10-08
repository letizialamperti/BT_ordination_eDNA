#!/bin/bash
#OAR -n bt-count-groups
#OAR -l /nodes=1/gpu=1,walltime=1:00:00
#OAR -p gpumodel='A100'
#OAR --stdout bt-count-groups-%jobid%.out
#OAR --stderr bt-count-groups-%jobid%.err
#OAR --project pr-qiepb
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /applis/environments/conda.sh
conda activate zioboia
python -u count_sequence_groups.py \
    --samples_dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
    --subset_sizes 250 500 1000 \
    --sequence_length 300 \
    --output_dir "sequence_group_counts_$(date +%Y%m%d_%H%M%S)_${OAR_JOB_ID:-local}"
