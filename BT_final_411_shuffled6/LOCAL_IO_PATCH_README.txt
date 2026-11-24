Local I/O export patch — 24 November 2026

The fold-4 log confirms OSError Errno 70 while reading a CSV on Bettik.
This patch leaves BT weights, training code and numerical inference unchanged.
It stages the checkpoint and one raw sample at a time on node-local scratch,
performs CSV generation and validation locally, then publishes verified files
atomically to shared storage. Transfers and metadata operations use bounded
retries (five attempts; backoff 1/2/4/8 seconds). Permanent failures still stop.

Before GPU inference, the whole existing sample cache is copied/hash-checked.
Stable corrupted cache entries are moved to _quarantine and regenerated.
An unreachable cache due to persistent communication errors is not called
corruption. Existing files with matching hashes are reused. The two previous
exporter protocols are accepted only when all numerical/data fields agree.

Copy/extract all three Python files into the EXISTING BT_final_411_shuffled6
folder: export_final.py, export_staged.py, export_io.py. Do not nest another
BT_final_411_shuffled6 folder inside it. Checkpoints and data are not replaced.
The ZIP can be extracted from the BT repository root:
  unzip -o BT_final_411_local_io_patch.zip
  cd BT_final_411_shuffled6
  chmod +x resume_export_fold04.sh
  oarsub -S ./resume_export_fold04.sh

Do not submit duplicate jobs for a still-running fold. No BT retraining occurs.
The launchers already set BT_SHUFFLE_SCRATCH=/var/tmp. Manual invocations may
also pass --scratch-dir /var/tmp. Only a checkpoint, one raw CSV at a time and
small per-sample outputs occupy scratch. Successful runs clean their scratch;
failed runs print/retain the local directory for diagnostics.

Progress may first show checking_cache, then export, then complete. Combined
CSV paths are unchanged; the root manifest.json marks a complete 411-sample
export. The models and pooling procedure are unchanged.

Passed tests: original real-BT CPU regression with inference equivalence,
simulated transient Errno 70 recovery, persistent transport error rejection,
corrupted-cache quarantine/recompute with identical values, detection/retry of
a simulated zero-byte published file, and strict protocol migration.

Shared storage remains necessary for raw-data transfer and publication. If
persistent communication failures remain after retries, HPC storage/network
support is needed; local staging cannot repair the cluster infrastructure.
