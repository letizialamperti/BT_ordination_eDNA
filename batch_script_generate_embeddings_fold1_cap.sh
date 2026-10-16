#!/bin/bash
#OAR -n bt-embeddings-fold1
#OAR -l /nodes=1/gpu=1,walltime=08:00:00
#OAR --stdout bt-embeddings-fold1-%jobid%.out
#OAR --stderr bt-embeddings-fold1-%jobid%.err
#OAR --project pr-qiepb

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source /applis/environments/conda.sh
conda activate zioboia

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"

nvidia-smi

python generate_embeddings_fold.py

echo "Job finished at: $(date)"
