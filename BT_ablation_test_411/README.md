# BT 411: initialization ablation and coralligenous test exports

This bundle exports embeddings only. It does not train BT or fit classifiers.
Use it with the four completed shuffled-six-epoch runs in:

    /bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/runs_bt_final_411_shuffled6/fold_01/seed_0

and the corresponding fold_02, fold_03 and fold_04 directories.
The BT repository and its ORDNA modules must remain available on the server.

## Upload and submit

Upload this entire folder as BT_ablation_test_411 directly inside
/bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA, then run:

```bash
cd /bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/BT_ablation_test_411
chmod +x submit_all.sh run_*.sh
./submit_all.sh
```

This submits five independent jobs (each one GPU, requested walltime 24 hours):

- run_initial_shared.sh: verify all four initial.ckpt state dictionaries, including
  buffers, are identical and have global_step zero. Export one shared encoder on
  411 cohort samples and the disjoint 28 test samples (439 samples total).
- run_test_fold01.sh through run_test_fold04.sh: export exactly the same 28 test
  samples using each fold's final.ckpt. Verify the six-epoch completion marker,
  checkpoint hash, training configuration and frozen spatial cohort.

The initial checkpoint file hashes may differ even when the tensor states match.
Only identical tensor states permit a shared initial export. If this check fails,
that job stops before inference; four separate initial exports would be needed.
The four trained-test jobs can run independently of the initial export.

To submit just one job instead of submit_all.sh, use, for example:

```bash
oarsub -S ./run_test_fold03.sh
```

Do not submit duplicate jobs for the same output; an exclusive lock protects it.

## Outputs

All output is under the new directory:

    /bettik/PROJECTS/pr-qiepb/lampertl/BT_ordination_eDNA/embeddings_ablation_test_411

It contains initial_shared and trained_fold01 through trained_fold04. Each has:

    progress.json
    embeddings/embeddings_preprojector_256D.csv
    embeddings/embeddings_projector_64D.csv
    embeddings/embedding_qc.csv
    embeddings/export_protocol.json
    embeddings/manifest.json
    embeddings/_samples/...

The unified CSVs and manifest are published after every expected sample passes
validation. A directory containing only _samples is an incomplete export.
The initial CSV uses role=cohort for 411 samples and role=test for 28 samples;
trained-test CSVs use role=test for all 28. Roles do not enter the encoder input.
The classification comparison will use pre-projector 256D means only, matching
the frozen trained-embedding analysis; 64D and SD are retained for QC.

The generator, read filtering, inference subset shuffling (seed zero), pooling
and batch size match the final trained export. All complete 500-pair inference
subsets are processed, with leftover reads excluded as in the existing protocol.
Inputs and checkpoints are staged locally; transfers use bounded retries and
hash checks. Invalid cached outputs are preserved in _quarantine. Re-running the
same SH resumes verified samples. The requested 24 hours is not a runtime
promise, especially for the shared 439-sample export.

Local job logs are copied on exit to this bundle's job_logs directory. The OAR
.out file prints that directory. A normal walltime termination should trigger
copying, but a forced kill or node failure may prevent the final log copy.

Download the three unified CSVs, export_protocol.json and manifest.json from
each completed embeddings directory. Keep _samples on the server for resuming;
it is not needed for the subsequent local classification analysis.

## Frozen downstream plan (local, after exports return)

The bundle does not yet implement the complete local ablation/test analysis.
The included evaluate_final_classification.py is an existing reference fitter;
it expects 411 samples with fold-specific train/outer_eval roles, not the shared
439-sample CSV directly. Do not run it directly on that shared export.

For the initial encoder, select the 411 cohort rows and restore each fold's
original roles from splits/fold_01.csv through fold_04.csv. Fit the same frozen
standardized 256D softmax32 classifier for 20 epochs, with no early stopping,
seeds 0/1/2, the same training-only scaler and class weighting. This gives 36
classifiers (four folds x three targets x three seeds). Score the outer samples
and compare with the already frozen trained-encoder results.

For coralligenous testing, apply each trained encoder's existing classifier and
its own training scaler to the matching trained_fold export. Apply each initial
encoder classifier and its own scaler to the 28 test rows of initial_shared.
Neither labels nor predictions of these 28 samples may change the frozen model,
scaler, epoch count or threshold. Report fold-level scores averaged over seeds;
a separately identified probability ensemble of the 12 classifiers per target
can also be reported, using the same rule for both encoder families.

Targets remain: unprotected_vs_protected, protection (fully protected versus
others), and Index_MPA_num (four classes). These 28 samples have already been
inspected in earlier experiments: this is a frozen comparison on an explored
holdout, not a claim of a completely untouched independent evaluation.

## Verification scope

verification.json records successful synthetic checkpoint-identity and job-routing
integration checks and Bash syntax validation. It does not claim that these jobs
have run on the server or that real-data inference has finished. Numerical
export/I/O code is copied unchanged from the previously verified final bundle.
The frozen_configuration.json file is the original decision record, including
historical status fields; its encoder/classifier parameters define this recipe.
