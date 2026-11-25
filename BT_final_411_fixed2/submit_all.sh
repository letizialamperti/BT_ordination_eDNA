#!/bin/bash
# Run this on the login node: submit four independent jobs to OAR.
set -euo pipefail
BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$BUNDLE"
chmod +x run_fold01.sh run_fold02.sh run_fold03.sh run_fold04.sh
oarsub -S ./run_fold01.sh
oarsub -S ./run_fold02.sh
oarsub -S ./run_fold03.sh
oarsub -S ./run_fold04.sh
