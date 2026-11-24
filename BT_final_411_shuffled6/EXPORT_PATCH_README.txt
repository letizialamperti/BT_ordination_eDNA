Export diagnostic/resume patch — 24 November 2026

Confirmed: BT completed all six epochs. Export rejected SPY202495 projector CSV.
The old error did not identify the exact mismatch, and temporary files were
removed on exception. Root cause is not established without the failing CSV.

This patch changes validation/diagnostics and publication order only. The model,
generator, inference math, batch size and pooling are unchanged. It accepts the
original exporter d991a0cb... only when all other protocol fields match, archives
the old export protocol, and reuses hash-verified completed samples.

Extract the patch ZIP in the BT repository root (not the output root):
  unzip -o BT_final_411_export_patch.zip
Then:
  cd BT_final_411_shuffled6
  chmod +x resume_export_fold03.sh
  oarsub -S ./resume_export_fold03.sh

Run after the previous job has ended. The resume job is export-only; it does not
change training checkpoints. Completed samples are verified/skipped. The sample
without a manifest is recalculated. If it fails again, read train.err and preserve:
  runs_bt_final_411_shuffled6/fold_03/seed_0/embeddings/epoch_06/
    _samples/SPY202495/_diagnostics/<attempt>/failure.json
The same directory contains all three generated CSVs (when available).
No failed sample is marked complete or included in the final combined CSV.

Equivalent export-only launchers for folds 1, 2 and 4 are included, but inspect
the corresponding job logs before starting them. Do not delete _samples.

Verification passed: valid schemas, intentionally malformed projector CSV,
persistent failure diagnostics, strict legacy-protocol migration, rejected data/
checkpoint/config/generator changes, export-only SHs, and real BT synthetic CPU
regression including six epochs, resume and inference equivalence.
