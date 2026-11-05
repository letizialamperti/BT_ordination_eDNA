#!/bin/bash
#OAR -n BT411-test-embeddings
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR --stdout BT411-test28-%jobid%.out
#OAR --stderr BT411-test28-%jobid%.err
#OAR --project pr-qiepb

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source /applis/environments/conda.sh
conda activate zioboia

python generate_test_embeddings.py "$@"
