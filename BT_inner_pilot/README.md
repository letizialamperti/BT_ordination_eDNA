# BT pilot with spatial inner validation

This pilot fixes outer block 1 and inner validation block 2 before examining results.
Raw-read roles are 215 inner training, 109 inner validation and 108 outer samples.
After restricting the classifier to the shared cohort, 209 inner training and 107
inner validation samples remain. Outer samples and the 28 coralligenous samples
are excluded from fitting, embedding export and pilot scoring.

Copy the entire directory to the BT server. Set BT_REPO in the launchers and run:

```bash
chmod +x run_inner_fixed.sh run_inner_shuffled.sh run_export_inner.sh
oarsub -S ./run_inner_fixed.sh
oarsub -S ./run_inner_shuffled.sh
```

Both modes start from fresh weights with seed 0, not the old one-epoch checkpoint.
They save initial.ckpt and a full checkpoint after every epoch, up to ten epochs.
Output is runs_bt_inner_pilot_io_retry/outer_01_inner_02/MODE/seed_0.
This separate root preserves the earlier failed run.
The fixed and shuffled runs have the same architecture and training chunk counts.
Shuffled mode rebuilds indexes after rewriting paired rows. Temporary CSVs occupy
one copy of the eligible inner training reads on node-local /var/tmp. Original
reads, checkpoints and CSV metrics still use the shared filesystem. Free space is
checked before creating the shuffled copy. Fixed validation views use at most
four 1,000-pair chunks per eligible sample, with a fixed seed and fixed batch order.
This cap limits the diagnostic cost and does not cap the training reads. Validation
does not select or stop the BT model during training. CSV logs contain the BT loss
and its components. Each OAR job requests 24 hours. Re-submit the same launcher to
resume from last.ckpt if it ended after saving at least one complete epoch.

Python stdout/stderr are written unbuffered to node-local /var/tmp. At process
exit, the launcher copies train.out, train.err and exit_status.txt to
job_logs/JOBID-MODE in this bundle and preserves the Python exit status.
If that copy fails, the OAR log reports the local directory where logs remain.
A forced kill or loss of the node can prevent this final copy. Progress bars and
model summaries are disabled; epoch metrics and checkpoint saving remain enabled.

Before a full run, a raw-data preflight can be run in the same conda environment
with the BT repository on PYTHONPATH:

```bash
python train_bt_inner_pilot.py --views fixed --check-only \
  --samples_dir /bettik/PROJECTS/pr-qiepb/lampertl/all \
  --output_root /bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/runs_bt_inner_pilot_io_retry
```

This scans eligible training and validation reads and checks their first/last
indexed views; it can take substantial time but requires no GPU training. It
does not open outer-fold CSVs. Normal training performs the same checks during
setup. Exclusive filesystem locks reject simultaneous writers to the same run.
Configuration, source hashes and raw input size/mtime are compared on resume;
changed inputs require a new output root. Every batch contains at least two
chunks: a last singleton is merged into the preceding batch (at most BS+1).

The reproducible synthetic regression suite uses the actual imported BT model,
short sequences and small subsets on CPU. It creates no real study embeddings:

```bash
python verify_pipeline.py --workers 0
# On a node that allows multiprocessing:
python verify_pipeline.py --workers 2 --report audit_report_workers2.json
```

It checks batch boundaries, invalid splits, competing writers, both view modes,
fresh initialization, validation, checkpoint numbering, exact resumed/continuous
CPU weights, frozen export, cached output integrity and the three-target
classifier. The JSON report records checked script hashes. CPU smoke tests do
not establish CUDA memory requirements or shared filesystem availability.

After checkpoints are available, use run_export_inner.sh. Set VIEWS to fixed or
shuffled and CHECKPOINT to all, initial or epoch_01 etc. Exports include only the
316 classifier samples. The same mean pooling over complete 500-pair subsets is
used for every checkpoint. Completed exports are verified and skipped on relaunch.
Export checks each sample's role against the frozen training split and refuses
to overlap with an active training job. Input file size/mtime and output hashes
are recorded, so changed raw inputs cannot silently reuse cached embeddings.
Versioned epoch files (for example epoch_02-v1.ckpt after an interrupted save)
are rejected by --checkpoint all, preventing duplicate epoch comparisons.
Verify the files before selecting an explicit checkpoint in that situation.
Initial export can take appreciable time because it reads every usable subset.

Transfer the embeddings directory for each mode to the classifier workspace.
Run the local fixed-budget classifier comparison, choosing separate output folders:

```bash
python evaluate_inner_classification.py \
  --embeddings-root /path/to/fixed/seed_0/embeddings \
  --metadata /Users/Letizia/Desktop/PAPER_MEE/Metadata_AMP_460.csv \
  --output /path/to/pilot_classification/fixed
```

The classifier uses the same standardized softmax32, three seeds, inverse
class-frequency weights and AdamW settings as the current comparison. Its duration
is fixed at 20 epochs for every candidate. This is a prespecified diagnostic budget,
not a duration optimized separately on the same inner validation samples.
The evaluator verifies all sample IDs and roles against splits/classifier_roles.csv;
use --roles only when deliberately evaluating a separately frozen cohort.
Compare all checkpoint curves and targets; choosing duration and mode from this one
split is exploratory. Verify choices in the other inner rotations before refitting
on the full outer training set. The pilot has one encoder initialization.
