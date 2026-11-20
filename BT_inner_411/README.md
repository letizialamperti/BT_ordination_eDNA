# BT candidate selection on the common 411 samples

This bundle starts a separate experiment. It never resumes the 432-based pilot.
All BT input role CSVs contain exactly the same 411 common IDs; the other 21 IDs
are absent, including from pretraining. The excluded outer block is always 1.
The audited trainer and exporter are reused unchanged.

| Rotation | BT/classifier training IDs | Inner validation IDs | Excluded outer IDs |
| --- | ---: | ---: | ---: |
| Outer 1, inner 2: training blocks 3+4 | 209 | 107 | 95 |
| Outer 1, inner 3: training blocks 2+4 | 194 | 122 | 95 |

Counts are before the trainer's read QC. Both candidates start from fresh seed-0
weights: fixed runs for 2 total epochs, shuffled for 6. Checkpoints are saved each
epoch; no early stopping or additional epoch selection is performed.

## Server training

Copy the entire BT_inner_411 directory into the BT repository on the server:
/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/BT_inner_411

From that directory:

```bash
chmod +x run_*.sh
oarsub -S ./run_inner02_fixed2.sh
oarsub -S ./run_inner02_shuffled6.sh
oarsub -S ./run_inner03_fixed2.sh
oarsub -S ./run_inner03_shuffled6.sh
```

Each job requests one GPU and 24 hours. This is four training jobs, 16 encoder
epochs in total. Input paths, epochs and modes are written directly into each SH.
The launchers activate zioboia and use the original BT repository via PYTHONPATH.
Shuffled temporary reads and process logs use node-local /var/tmp. Log copy-back
and shared-storage limitations are the same as the audited pilot.

Outputs are separate from previous experiments:

```text
runs_bt_inner_411/outer_01_inner_02/fixed/seed_0/
runs_bt_inner_411/outer_01_inner_02/shuffled/seed_0/
runs_bt_inner_411/outer_01_inner_03/fixed/seed_0/
runs_bt_inner_411/outer_01_inner_03/shuffled/seed_0/
```

## Export after the corresponding training job finishes

```bash
oarsub -S ./run_export_inner02_fixed2.sh
oarsub -S ./run_export_inner02_shuffled6.sh
oarsub -S ./run_export_inner03_fixed2.sh
oarsub -S ./run_export_inner03_shuffled6.sh
```

Each export job processes just initial.ckpt and its candidate checkpoint
(epoch_02.ckpt for fixed, epoch_06.ckpt for shuffled). A checkpoint is complete
when its embeddings directory contains manifest.json. Completed exports are
verified and skipped on relaunch; an interrupted checkpoint starts over.

The final classifier uses preprojector 256D means. The projector 64D and QC files
are also exported for integrity checks. No outer-fold IDs are exported here.

## Local comparison and selection

Download config.json and the embeddings directory from each seed_0 folder,
preserving the tree below runs_bt_inner_411. The checkpoints need not be copied
locally. For example, if the downloaded tree is /path/to/runs_bt_inner_411:

```bash
conda activate zioboia
python evaluate_selection.py \
  --runs-root /path/to/runs_bt_inner_411 \
  --metadata /Users/Letizia/Desktop/PAPER_MEE/Metadata_AMP_460.csv \
  --output /path/to/BT411_inner_selection
```

The classifier protocol is fixed: standardized softmax32, 20 training epochs,
three seeds 0/1/2, inverse-frequency training weights, and the same settings as
the first pilot. Standardization is fit on inner training samples only. Run
configs and manifests must agree with this bundle's 411-sample assignments.

The comparison rule is written before the new results in experiment_protocol.json:
average balanced accuracy across classifier seeds, then give equal weight to
the three targets and two rotations. An exact tie favors fixed2 for lower cost.
Target/rotation results and classifier-seed SD are reported separately. Seed SD
is not uncertainty across independent field samples or spatial folds.

Outputs include comparison_by_rotation_target.csv, candidate_selection.csv and
selection_result.json. The result is a recommendation to inspect before freezing
the final configuration. No outer scoring or automatic final training occurs.

After selection, fit four fresh encoders on the three outer-training blocks for
each outer fold, using the selected mode/duration and no inner holdout. Refit the
classifiers on the corresponding training cohort with a frozen training budget.
Their outer-scoring and final shuffled training launchers will be prepared once
the candidate is frozen. Previously explored folds remain a limitation of the
development process; this experiment is not a fully nested evaluation.
