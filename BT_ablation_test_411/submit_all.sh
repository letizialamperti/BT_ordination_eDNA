#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
oarsub -S ./run_initial_shared.sh
oarsub -S ./run_test_fold01.sh
oarsub -S ./run_test_fold02.sh
oarsub -S ./run_test_fold03.sh
oarsub -S ./run_test_fold04.sh
