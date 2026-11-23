"""Fresh outer-fold BT refit: frozen shuffled-6 recipe, no inner validation."""
import argparse
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from types import SimpleNamespace

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch

from training_core import PilotData, EpochLabel, BarlowTwinsDataset, SelfAttentionBarlowTwinsEmbedder
from ORDNA.models.representation_module import SelfAttentionRepresentationModule
from ORDNA.utils.sequence_mapper import SequenceMapper
from final_checks import sha, read_roles, file_stats, run_lock, atomic_json


HERE = Path(__file__).resolve().parent


class OuterTrainingData(PilotData):
    def setup(self, stage=None):
        if self.training is not None:
            return
        # No stat(), open(), indexing or loader for the outer sample files.
        files = [self.args.samples_dir / f"{code}.csv"
                 for code in self.roles.loc[self.roles.role.eq("train"), "spygen_code"]]
        self.training = BarlowTwinsDataset(files, self.args.sample_subset_size,
                                          self.args.sequence_length, max_chunks_per_sample=0,
                                          seed=self.args.seed)
        if len(self.training) < 2:
            raise ValueError("Insufficient eligible training chunks.")
        self.original_files = list(self.training.files)
        self.training_counts = list(self.training.accumulated_num_subsets)
        pd.DataFrame(self.training.qc_records).to_csv(self.run_dir / "train_qc.csv", index=False)
        for index in {0, len(self.training)-1}:
            if any(tuple(v.shape) != (self.args.sample_subset_size, 2, self.args.sequence_length)
                   for v in self.training[index]):
                raise ValueError("Unexpected training view shape.")

    # Reuse the exact epoch-seeded shuffled views, sampler and batch construction
    # used in the selected pilot. Explicitly disable inherited validation loader.
    def val_dataloader(self):
        return []


class Progress(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        atomic_json(Path(trainer.default_root_dir) / "progress.json", dict(
            phase="training", epoch_work_completed=trainer.current_epoch+1,
            total_epochs=trainer.max_epochs, global_step=trainer.global_step))
        print(f"Finished epoch work {trainer.current_epoch+1}/{trainer.max_epochs}; "
              f"global step {trainer.global_step}", flush=True)


def source_hashes():
    return dict(trainer=sha(__file__), core=sha(HERE / "training_core.py"),
                checks=sha(HERE / "final_checks.py"), locks=sha(HERE / "pilot_checks.py"),
                dataset=sha(inspect.getfile(BarlowTwinsDataset)),
                model=sha(inspect.getfile(SelfAttentionBarlowTwinsEmbedder)),
                representation=sha(inspect.getfile(SelfAttentionRepresentationModule)),
                sequence_mapper=sha(inspect.getfile(SequenceMapper)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=[1,2,3,4], required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--roles", type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=os.environ.get("BT_SHUFFLE_SCRATCH"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--smoke-test", action="store_true", help="Synthetic-only small inputs; never a study refit.")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("workers must be non-negative")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        parser.error("CUDA GPU unavailable; check the OAR allocation and conda environment.")
    args.roles = args.roles or HERE / f"splits/fold_{args.fold:02d}.csv"
    roles = read_roles(args.roles, args.fold, args.smoke_test)
    recipe = json.loads((HERE / "frozen_configuration.json").read_text())
    params = dict(recipe["encoder"])
    if params["views"] != "shuffled" or params["max_epochs"] != 6:
        raise ValueError("Expected frozen shuffled-6 recipe.")
    if args.smoke_test:
        params.update(sequence_length=8, sample_subset_size=2, batch_size=3, token_emb_dim=2)
    sources = source_hashes()
    reference = json.loads((HERE / "reference_source_hashes.json").read_text())
    if not args.smoke_test:
        for name in ["dataset", "model", "representation", "sequence_mapper"]:
            if sources[name] != reference[name]:
                raise ValueError(f"BT repository {name} differs from the selected pilot source.")
    run = args.output_root / f"fold_{args.fold:02d}/seed_{params['seed']}"
    with run_lock(run):
        train(args, roles, params, sources, run)


def train(args, roles, params, sources, run):
    paths = [args.samples_dir / f"{c}.csv" for c in roles.loc[roles.role.eq("train"), "spygen_code"]]
    config = dict(params, outer_fold=args.fold, smoke_test=args.smoke_test,
                  roles_sha256=sha(args.roles), frozen_recipe_sha256=sha(HERE / "frozen_configuration.json"),
                  raw_input_stats=file_stats(paths), source_sha256=sources, inner_validation=False)
    path = run / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Run inputs, recipe or code changed. Use a distinct output root.")
    else:
        if list(run.glob("*.ckpt")):
            raise ValueError("Checkpoints exist without config.json; preserve this run and inspect it.")
        shutil.copy2(args.roles, run / "sample_roles.csv")
        atomic_json(path, config)
    if sha(run / "sample_roles.csv") != config["roles_sha256"]:
        raise ValueError("Frozen run roles changed.")
    final = run / "final.ckpt"
    marker = run / "training_complete.json"
    if marker.exists():
        completed = json.loads(marker.read_text())
        if completed["config_sha256"] != sha(path) or not final.is_file() or completed["checkpoint_sha256"] != sha(final):
            raise ValueError("Completed training artefacts changed.")
        print(f"Verified completed training; moving to export: {run}", flush=True)
        return
    resume = run / "last.ckpt"
    if not resume.exists():
        if list(run.glob("epoch_*.ckpt")):
            raise ValueError("Epoch checkpoints exist but last.ckpt is missing; inspect before resuming.")
        resume = run / "initial.ckpt"
    resume = str(resume) if resume.exists() else None
    print(json.dumps(dict(output=str(run), roles=roles.role.value_counts().to_dict(),
                          python=platform.python_version(), torch=torch.__version__, lightning=pl.__version__,
                          resume=resume, source_hashes=sources)), flush=True)
    atomic_json(run / "progress.json", dict(phase="initializing_training", total_epochs=6,
                                            resume_checkpoint=resume))
    pl.seed_everything(params["seed"], workers=True)
    model = SelfAttentionBarlowTwinsEmbedder(
        token_emb_dim=params["token_emb_dim"], seq_len=params["sequence_length"],
        sample_repr_dim=params["sample_repr_dim"], sample_emb_dim=params["sample_emb_dim"],
        lmbda=params["barlow_twins_lambda"], initial_learning_rate=params["initial_learning_rate"],
        weight_decay=params["weight_decay"])
    checkpoint = ModelCheckpoint(dirpath=str(run), filename="epoch_{completed_epochs:02.0f}",
                                 auto_insert_metric_name=False, save_top_k=-1, save_last=True,
                                 every_n_epochs=1, save_on_train_epoch_end=True,
                                 enable_version_counter=False)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=1, max_epochs=params["max_epochs"],
                         default_root_dir=str(run), callbacks=[EpochLabel(), Progress(), checkpoint],
                         logger=CSVLogger(str(run), name="metrics"), limit_val_batches=0,
                         num_sanity_val_steps=0, reload_dataloaders_every_n_epochs=1,
                         log_every_n_steps=10, enable_progress_bar=False, enable_model_summary=False)
    data_args = SimpleNamespace(**params, samples_dir=args.samples_dir, num_workers=args.workers)
    with tempfile.TemporaryDirectory(prefix="bt_final_", dir=args.scratch_dir) as temporary:
        print(f"Local shuffled scratch: {temporary}", flush=True)
        atomic_json(run / "progress.json", dict(phase="indexing_or_training", total_epochs=6,
                                                local_scratch=temporary, resume_checkpoint=resume))
        trainer.fit(model, datamodule=OuterTrainingData(data_args, roles, run, Path(temporary)), ckpt_path=resume)
        if trainer.current_epoch < params["max_epochs"]:
            raise RuntimeError("Training ended before the frozen epoch budget.")
        trainer.save_checkpoint(str(final))
    atomic_json(marker, dict(config_sha256=sha(path), checkpoint_sha256=sha(final), completed_epochs=6))
    atomic_json(run / "progress.json", dict(phase="training_complete", completed_epochs=6, next_phase="export"))
    print(f"TRAINING COMPLETE: {final}", flush=True)


if __name__ == "__main__":
    main()
