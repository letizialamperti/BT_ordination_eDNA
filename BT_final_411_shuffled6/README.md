# Four final BT refits on the 411 common samples

Frozen recipe: shuffled paired-read views each epoch, six total epochs, seed 0,
300 nucleotides, 500 paired reads per view, batch size 64, token embedding 8,
preprojector 256D, projector 64D, lambda 0.005, AdamW lr 0.0003 and wd 0.0001.
No inner validation, early stopping or further model selection is performed.

| Outer fold | Training IDs | Held-out IDs |
| --- | ---: | ---: |
| 1 | 316 | 95 |
| 2 | 304 | 107 |
| 3 | 289 | 122 |
| 4 | 324 | 87 |

The 411 IDs and original spatial block assignments are frozen in cohort_411.csv.
The 21 extra IDs used in earlier 432-based experiments are absent. Counts precede
read QC. Training indexes/reads only train IDs. Held-out raw files are accessed
only by the frozen export after training finishes.

## Submit on the server

Copy/extract this entire directory into:

```text
/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/BT_final_411_shuffled6
```

From the directory:

```bash
chmod +x submit_all.sh run_fold01.sh run_fold02.sh run_fold03.sh run_fold04.sh
./submit_all.sh
```

The submission script issues four independent OAR submissions. Each requests
one GPU for 24 hours; OAR determines when GPUs are available. Each fold is written
directly in its SH. Do not submit the submission script itself as a GPU job.
Individual submission is also possible, for example:

```bash
oarsub -S ./run_fold01.sh
```

Do not launch both methods for the same fold. Runtime locks reject concurrent
writers, but duplicate submissions still waste queue slots.

Every job activates zioboia using /applis/environments/conda.sh and imports the
BT repository using PYTHONPATH. Training validates the repository model, dataset,
representation and mapper hashes against those used for candidate selection.
New runs initialize weights from seed 0; they do not load any pilot checkpoint.

## Same-job sequence and outputs

Each run_foldNN.sh runs train_final.py, then export_final.py. If training fails,
export is not attempted. The output root is separate from all pilot runs:

```text
runs_bt_final_411_shuffled6/fold_01/seed_0/
runs_bt_final_411_shuffled6/fold_02/seed_0/
runs_bt_final_411_shuffled6/fold_03/seed_0/
runs_bt_final_411_shuffled6/fold_04/seed_0/
```

Each run saves config.json, sample_roles.csv, train_qc.csv, initial.ckpt,
epoch_01.ckpt through epoch_06.ckpt, last.ckpt, final.ckpt, CSV metrics,
training_complete.json and progress.json. The final export uses final.ckpt,
which is saved after all six epochs have finished.

Only the final encoder is exported in this job, avoiding another pass for the
initial encoder or intermediate epochs. The initialized checkpoint is preserved
for a later ablation. The final outputs, including all 411 train/outer IDs, are:

```text
seed_0/embeddings/epoch_06/embeddings_preprojector_256D.csv
seed_0/embeddings/epoch_06/embeddings_projector_64D.csv
seed_0/embeddings/epoch_06/embedding_qc.csv
seed_0/embeddings/epoch_06/manifest.json
```

The 256D means are the chosen classifier input. The CSVs retain within-sample SD
for provenance, but it is not an input of the frozen classifier. Roles are train
and outer_eval; exporting held-out data does not update encoder weights. There is
no classifier training or scoring in these GPU jobs.

## Progress and interruption

Python logs use node-local /var/tmp and are copied to this bundle's job_logs at
exit. The OAR stdout reports the node and local log path. On shared storage, the
latest phase can be inspected without entering the GPU node:

```bash
cat ../runs_bt_final_411_shuffled6/fold_01/seed_0/progress.json
```

During training it reports initialization/indexing or finished epoch work.
Durable resume points are the actual checkpoints, not the progress counter.
During export it reports completed samples, total samples and the last sample.

Twenty-four hours is a requested limit, not a measured runtime guarantee. Full
training uses more samples than the inner pilot; four concurrent jobs can also
compete for shared I/O. A rerun of the same fold SH is safe after the first job
has ended: it resumes its own last.ckpt, or verifies/skips completed training,
then resumes export using per-sample manifests. It never resumes a pilot run.
Changed data or code are rejected instead of silently mixing artefacts.

Export writes an atomic result for each completed sample under _samples and
combines them into the three final CSVs only when all samples are finished.
Missing final manifest.json means the combined export is not complete. A sample
interrupted during computation is recalculated; earlier complete samples are
verified and reused. Preserve _samples until export is complete. Forced SIGKILL
or node loss can prevent final local-log copy-back, but saved shared-storage
checkpoints and per-sample transactions remain usable.

Shuffled temporary CSVs occupy a node-local copy of eligible training reads;
free capacity is checked before construction. Training originals, checkpoints
and sample export cache remain on Bettik, so shared filesystem failures can still
interrupt execution. The six-epoch numerical budget, mean pooling, 500-pair
inference subsets, inference batch 64 and per-sample random_state=0 are preserved.

## Verification

verify_pipeline.py uses the actual imported BT code on small synthetic CPU data.
It checks absent outer files during training, six-epoch completion, epoch-5 resume,
no validation metrics, same-job relaunch behavior, incremental export recovery,
and equivalence with original single-pass inference. Shell checks verify all four
OAR headers, fold arguments, train-before-export ordering and submission commands.
Reports are supplied in verification.json and shell_verification.json.
These tests do not measure full-sized H100 runtime or memory requirements.
