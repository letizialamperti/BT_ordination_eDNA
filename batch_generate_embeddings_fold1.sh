#!/bin/bash
#OAR -n bt-embeddings-fold1
#OAR -l /nodes=1/gpu=1,walltime=08:00:00
#OAR -p gpumodel='A100'
#OAR --stdout bt-embeddings-fold1-%jobid%.out
#OAR --stderr bt-embeddings-fold1-%jobid%.err
#OAR --project pr-qiepb

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source /applis/environments/conda.sh
conda activate zioboia

python generate_embeddings_fold.py
