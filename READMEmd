# Barlow Twins: outer spatial CV with an inner holdout

This is four-fold outer CV with **one prespecified inner spatial holdout** per outer fold, not an exhaustive inner k-fold search. The existing CSV partitions are reused as groups; their spatial distances and class balance must still be checked with metadata. Default inner groups are 2, 3, 4, 1 for outer folds 1, 2, 3, 4. Never select the inner group from outer performance.

For each job, only `inner_train` updates weights and only `inner_valid` selects the checkpoint. `outer_eval` is recorded but its sequence files are never loaded during BT fitting. The external habitat test is absent from all four input CSVs. There is no automatic refit on all outer-training samples.

## Inspect splits (no torch or sequence files required)

```bash
for fold in 1 2 3 4; do
  python training_BarlowTwins_kfold.py --outer_fold "$fold" --dry_run
done
```

## Train

Use the original cluster environment. Confirm the hyperparameters from the actual experiment: the existing launcher and config disagree. Example command below uses shell variables which must first be set to the verified values:

```bash
python training_BarlowTwins_kfold.py \
  --outer_fold 1 \
  --samples_dir "$BT_SAMPLES_DIR" \
  --sample_repr_dim "$BT_REPR_DIM" \
  --sample_emb_dim "$BT_EMB_DIM" \
  --barlow_twins_lambda "$BT_LAMBDA" \
  --initial_learning_rate "$BT_LEARNING_RATE" \
  --max_epochs "$BT_MAX_EPOCHS"
```

Defaults retained from the old launcher: sequence length 300, subset size 500, batch size 8, token dimension 8. Verify those too. Set `--inner_fold` only if adopting a different prespecified protocol. `--accelerator cpu --num_workers 0` can be used for a small local smoke run.

`batch_script_BarlowTwins_kfold.sh` provides the original OAR/environment setup, with the fold as its first argument and explicit required `BT_*` environment variables. Run it from the repository root. Four separate jobs can use folds 1 through 4. Check how your scheduler forwards arguments/environment before submission.

Each job creates `runs_bt_kfold/fold_XX/seed_0/` with configuration, original split snapshots and SHA256 hashes, `sample_roles.csv`, metrics, best/last checkpoints and `best_checkpoint.txt`. Existing run directories are rejected rather than overwritten. Set a new output root for a new run.

## Downstream classifier and embeddings

Load the checkpoint named in `best_checkpoint.txt` in eval mode with no gradient updates. Use that same encoder for all sample embeddings within this outer fold. Do not pool embeddings across different encoders. The current `visualize_embeddings.py` has hard-coded paths and is not automatically adapted by this change.

Use `sample_roles.csv` to fit the classifier on `inner_train`, tune/checkpoint it on `inner_valid`, then report predictions on `outer_eval`. This simple protocol deliberately reserves the inner group throughout. Refitting on all outer-training samples is a separate protocol requiring fixed training budgets learned internally. The classifier code still needs adaptation. Keep the habitat test out of all development decisions.

## Numerical and data safeguards

The loss now clamps its standard-deviation denominator at 1e-6, retaining the existing sample-standard-deviation convention. A batch sampler merges final singleton batches into the preceding batch, retaining all chunks (maximum batch size is nominal size + 1). Missing sample files or samples dropped by the original Dataset cause an error. Raw-read parsing and the existing deterministic chunk construction remain unchanged and require their own data QC.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests verify sample isolation, rejected duplicate IDs, and complete chunk coverage with no singleton batches. Full numerical/GPU training must be checked in the project environment with the sequence data.
