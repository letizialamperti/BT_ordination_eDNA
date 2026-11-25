# BT fixed views for two epochs on the common 411 samples

This folder submits four independent OAR GPU jobs. Each job trains a fresh BT
encoder for exactly two epochs with fixed within-sample views, then exports
sample embeddings for all 411 development samples using its final checkpoint.
This is a prespecified sensitivity experiment, not a configuration selected as
the best model. No classifier is trained here and the 28 coralligenous test
samples are excluded from both this training and this export.

The training core, architecture and hyperparameters match BT_final_411_shuffled6;
only the view mode and training duration change. Fixed means that the same
within-sample read subsets are used in both epochs. Chunk/minibatch order still
changes deterministically between epochs. Inference uses the existing frozen
shuffle seed zero and pools over all complete 500-pair inference subsets.

## Upload and launch

Upload the entire BT_final_411_fixed2 folder directly inside:

    /bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA

The existing ORDNA modules from that repository and conda environment zioboia
are required. No installation command or repository update is performed by the
jobs. For reproducibility, the launcher checks the dataset/model/representation/
sequence-mapper source hashes against those used by the previous pilot. If those
files differ, the training stops rather than silently changing the model.

On the login node:

```bash
cd /bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/BT_final_411_fixed2
chmod +x submit_all.sh run_fold*.sh resume_export_fold*.sh
./submit_all.sh
```

This submits four jobs; OAR decides when resources permit each one to start.
Each requests one GPU, one node, 24 hours and project pr-qiepb. The fold number
is written directly inside each SH, so no positional fold argument is required.
To submit just one fold instead, run:

```bash
oarsub -S ./run_fold01.sh
```

Do not run submit_all.sh after separately submitting those same jobs.

## Cohort and training

Each encoder sees only the train IDs in its shipped split. The held-out spatial
block is not opened, indexed or passed to a validation loader during training.
There is no inner validation, early stopping, or checkpoint choice by a score.
The training duration is fixed before submission.

| Fold | Training samples | Outer samples | Exported samples |
|---|---:|---:|---:|
| 01 | 316 | 95 | 411 |
| 02 | 304 | 107 | 411 |
| 03 | 289 | 122 | 411 |
| 04 | 324 | 87 | 411 |

Settings: fixed views, two epochs, seed 0, sequence length 300, 500 read pairs per
view, training batch size 64, token dimension 8, pre-projector dimension 256,
projector dimension 64, lambda 0.005, learning rate 0.0003, weight decay 0.0001,
uncapped training chunks, 12 DataLoader workers. A final singleton batch is
merged into the preceding batch; all eligible chunks are retained.

## Checkpoints and exports

The new output root is separate from all previous runs:

    /bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/runs_bt_final_411_fixed2

Each fold has fold_01/seed_0, fold_02/seed_0, etc. It saves initial.ckpt,
epoch_01.ckpt, epoch_02.ckpt, last.ckpt and final.ckpt, plus config.json,
sample_roles.csv, train_qc.csv, metrics, training_complete.json and progress.json.
The initial checkpoint is saved before training for a later initialization
ablation. final.ckpt is saved after the second epoch; its tensor state should
match epoch_02.ckpt, although checkpoint file bytes need not be identical.

After training, that same job automatically creates:

    fold_01/seed_0/embeddings/epoch_02/embeddings_preprojector_256D.csv
    fold_01/seed_0/embeddings/epoch_02/embeddings_projector_64D.csv
    fold_01/seed_0/embeddings/epoch_02/embedding_qc.csv
    fold_01/seed_0/embeddings/epoch_02/export_protocol.json
    fold_01/seed_0/embeddings/epoch_02/manifest.json

Each unified CSV contains all 411 samples with train/outer_eval roles for that
fold. Means and within-sample standard deviations are retained in both embedding
CSVs. Subsequent classification will use 256D means, consistent with the existing
comparison; these scripts do not select a different representation.

## Interrupted jobs and export resume

The 24-hour request does not guarantee completion. The pipeline is resumable:
relaunching a run SH resumes an unfinished training from last.ckpt (or from the
saved initial checkpoint when no epoch has completed). Completed training is
hash-verified and skipped. Changed inputs/configuration/code are rejected instead
of overwriting an incompatible run.

If training has finished and only export remains, submit the corresponding
export-only job, for example:

```bash
oarsub -S ./resume_export_fold03.sh
```

Export uses node-local /var/tmp for the checkpoint, one raw sample at a time,
inference and validation. Shared-storage transfers use bounded retries and hash
verification. Each completed sample is cached in embeddings/epoch_02/_samples.
A corrupted cache entry is preserved under _quarantine before regenerating it.
The unified CSVs and final manifest are produced only after all 411 samples pass
validation. A directory containing only _samples is not a complete export.

Once complete, download the unified CSVs, config.json, sample_roles.csv,
training_complete.json, export_protocol.json and manifest.json for each fold.
Keep the per-sample cache on the server for recovery; it need not be downloaded
for classification.

## Logs and progress

OAR output names start BT411-final-fixed2-f01, etc. The .out file prints the
node-local log directory and the final log destination inside this folder:

    job_logs/JOBID-final-411-fold01-fixed2/train.out
    job_logs/JOBID-final-411-fold01-fixed2/train.err
    job_logs/JOBID-final-411-fold01-fixed2/exit_status.txt

Logs are written locally and copied back on exit. A forced kill/node failure can
prevent the exit trap from copying them. progress.json in the run directory is
updated during training and export. To check fold 1:

```bash
cat ../runs_bt_final_411_fixed2/fold_01/seed_0/progress.json
```

## Local verification

verification.json records the synthetic real-model regression test. It checks
fixed-view behavior, exactly two epochs, no training reads from the absent outer
files, deterministic checkpoint resume, export/recovery, and agreement with the
single-pass numerical generator. shell_verification.json covers Bash syntax and
mock submission of four jobs; no actual server job is submitted by these checks.
The test uses small synthetic dimensions and does not establish runtime or GPU
memory use at study scale. Source versions and hashes are recorded for auditing.
