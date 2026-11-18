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
Output is runs_bt_inner_pilot/outer_01_inner_02/MODE/seed_0.
The fixed and shuffled runs have the same architecture and training chunk counts.
Shuffled mode rebuilds indexes after rewriting paired rows. Temporary CSVs occupy
one copy of the eligible inner training reads. Fixed validation views use at most
four 1,000-pair chunks per eligible sample, with a fixed seed and fixed batch order.
This cap limits the diagnostic cost and does not cap the training reads. Validation
does not select or stop the BT model during training. CSV logs contain the BT loss
and its components. Each OAR job requests 24 hours. Re-submit the same launcher to
resume from last.ckpt if it ended after saving at least one complete epoch.

After checkpoints are available, use run_export_inner.sh. Set VIEWS to fixed or
shuffled and CHECKPOINT to all, initial or epoch_01 etc. Exports include only the
316 classifier samples. The same mean pooling over complete 500-pair subsets is
used for every checkpoint. Completed exports are verified and skipped on relaunch.
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
Compare all checkpoint curves and targets; choosing duration and mode from this one
split is exploratory. Verify choices in the other inner rotations before refitting
on the full outer training set. The pilot has one encoder initialization.
