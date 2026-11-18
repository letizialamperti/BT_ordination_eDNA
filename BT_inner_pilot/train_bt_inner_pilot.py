"""Fit BT from initialization with an excluded inner spatial block."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader, RandomSampler

from ORDNA.data.barlow_twins_dataset import BarlowTwinsDataset
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CompleteBatches:
    """Keep a final singleton by merging it into the preceding batch."""
    def __init__(self, sampler, batch_size):
        if len(sampler) < 2 or batch_size < 2:
            raise ValueError("BT requires at least two chunks and a batch size >= 2.")
        self.sampler, self.batch_size = sampler, batch_size

    def __len__(self):
        q, r = divmod(len(self.sampler), self.batch_size)
        return q + int(r > 1)

    def __iter__(self):
        batch = []
        remaining = len(self.sampler)
        for index in self.sampler:
            batch.append(index)
            remaining -= 1
            if len(batch) >= self.batch_size and remaining != 1:
                yield batch
                batch = []
        if batch:
            yield batch


class PilotData(pl.LightningDataModule):
    def __init__(self, args, roles, run_dir):
        super().__init__()
        self.args, self.roles, self.run_dir = args, roles, run_dir
        self.training = None
        self.validation = None

    def setup(self, stage=None):
        if self.training is not None:
            return
        def make(role, cap, seed):
            files = [self.args.samples_dir / f"{code}.csv"
                     for code in self.roles.loc[self.roles.role.eq(role), "spygen_code"]]
            missing = [str(path) for path in files if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing)
            return BarlowTwinsDataset(files, self.args.sample_subset_size,
                                     self.args.sequence_length,
                                     max_chunks_per_sample=cap, seed=seed)
        self.training = make("inner_train", 0, self.args.seed)
        self.validation = make("inner_validation", self.args.validation_chunks_per_sample, 123)
        self.original_files = list(self.training.files)
        self.training_counts = list(self.training.accumulated_num_subsets)
        self.validation_order = np.random.default_rng(123).permutation(len(self.validation)).tolist()
        for name, dataset in [("train", self.training), ("validation", self.validation)]:
            if len(dataset) < 2:
                raise ValueError(f"Insufficient chunks in {name}.")
            pd.DataFrame(dataset.qc_records).to_csv(self.run_dir / f"{name}_qc.csv", index=False)

    def train_dataloader(self):
        epoch = self.trainer.current_epoch + 1
        if self.args.views == "shuffled":
            cache = self.run_dir / "shuffled_reads"
            if cache.exists():
                shutil.rmtree(cache)
            cache.mkdir()
            files = []
            for source in self.original_files:
                code_seed = int.from_bytes(hashlib.sha256(source.stem.encode()).digest()[:4], "little")
                rng = np.random.default_rng(np.random.SeedSequence([self.args.seed, epoch, code_seed]))
                frame = pd.read_csv(source)
                output = cache / source.name
                frame.iloc[rng.permutation(len(frame))].to_csv(output, index=False)
                files.append(output)
            training = BarlowTwinsDataset(files, self.args.sample_subset_size,
                                         self.args.sequence_length, max_chunks_per_sample=0,
                                         seed=self.args.seed)
            if training.accumulated_num_subsets != self.training_counts:
                raise ValueError("Shuffling changed the training chunks per sample.")
        else:
            training = self.training
        # Explicit epoch seeds make order reproducible when continuing a completed epoch.
        generator = torch.Generator().manual_seed(self.args.seed + 1000 * epoch)
        sampler = RandomSampler(training, generator=generator)
        return DataLoader(training, batch_sampler=CompleteBatches(sampler, self.args.batch_size),
                          num_workers=self.args.num_workers, persistent_workers=False,
                          generator=torch.Generator().manual_seed(self.args.seed + epoch),
                          pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.validation,
                          batch_sampler=CompleteBatches(self.validation_order, self.args.batch_size),
                          num_workers=self.args.num_workers, persistent_workers=False,
                          generator=torch.Generator().manual_seed(123),
                          pin_memory=torch.cuda.is_available())


class EpochLabel(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log("completed_epochs", float(trainer.current_epoch + 1), logger=True)

    def on_fit_start(self, trainer, pl_module):
        initial = Path(trainer.default_root_dir) / "initial.ckpt"
        if trainer.ckpt_path is None:
            trainer.save_checkpoint(str(initial))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", type=Path, default=Path(__file__).parent / "splits/sample_roles.csv")
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("runs_bt_inner_pilot"))
    parser.add_argument("--views", choices=["fixed", "shuffled"], required=True)
    parser.add_argument("--sequence_length", type=int, default=300)
    parser.add_argument("--sample_subset_size", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--token_emb_dim", type=int, default=8)
    parser.add_argument("--sample_repr_dim", type=int, default=256)
    parser.add_argument("--sample_emb_dim", type=int, default=64)
    parser.add_argument("--barlow_twins_lambda", type=float, default=.005)
    parser.add_argument("--initial_learning_rate", type=float, default=.0003)
    parser.add_argument("--weight_decay", type=float, default=.0001)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--validation_chunks_per_sample", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="gpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.sequence_length, args.sample_subset_size, args.token_emb_dim,
           args.sample_repr_dim, args.sample_emb_dim, args.max_epochs,
           args.validation_chunks_per_sample) <= 0 or args.batch_size < 2 or args.num_workers < 0:
        parser.error("Invalid dimensions, epoch count, batch size or worker count.")
    roles = pd.read_csv(args.roles)
    assert roles.spygen_code.is_unique
    assert set(roles.role) == {"inner_train", "inner_validation", "outer_eval"}
    groups = roles.groupby("role").spatial_block.unique().to_dict()
    assert len(groups["outer_eval"]) == len(groups["inner_validation"]) == 1
    assert len(set(groups["inner_train"])) == 2
    assert roles.spatial_block.nunique() == 4
    outer, inner = int(groups["outer_eval"][0]), int(groups["inner_validation"][0])
    run_dir = args.output_root / f"outer_{outer:02d}_inner_{inner:02d}" / args.views / f"seed_{args.seed}"
    print(roles.groupby("role").size().to_dict(), "output:", run_dir, flush=True)
    if args.check_only:
        return
    import inspect
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["roles_sha256"] = sha(args.roles)
    config["source_sha256"] = {
        "trainer": sha(__file__),
        "dataset": sha(inspect.getfile(BarlowTwinsDataset)),
        "model": sha(inspect.getfile(SelfAttentionBarlowTwinsEmbedder)),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    previous = run_dir / "config.json"
    resume = None
    if previous.exists():
        old = json.loads(previous.read_text())
        for key in config:
            if key not in {"max_epochs", "accelerator", "num_workers", "check_only"} and old.get(key) != config[key]:
                raise ValueError(f"Existing run differs in {key}.")
        resume = run_dir / "last.ckpt"
        if not resume.is_file():
            raise ValueError("Existing incomplete run has no last.ckpt. Select a fresh output_root.")
    else:
        previous.write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy2(args.roles, run_dir / "sample_roles.csv")
    pl.seed_everything(args.seed, workers=True)
    model = SelfAttentionBarlowTwinsEmbedder(
        token_emb_dim=args.token_emb_dim, seq_len=args.sequence_length,
        sample_repr_dim=args.sample_repr_dim, sample_emb_dim=args.sample_emb_dim,
        lmbda=args.barlow_twins_lambda, initial_learning_rate=args.initial_learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir), filename="epoch_{completed_epochs:02.0f}",
        auto_insert_metric_name=False, save_top_k=-1, save_last=True,
        every_n_epochs=1, save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=1, max_epochs=args.max_epochs,
        default_root_dir=str(run_dir), callbacks=[EpochLabel(), checkpoint],
        logger=CSVLogger(str(run_dir), name="metrics"), num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1, log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=PilotData(args, roles, run_dir), ckpt_path=resume)
    trainer.save_checkpoint(str(run_dir / "final.ckpt"))
    cache = run_dir / "shuffled_reads"
    if args.views == "shuffled" and cache.exists():
        shutil.rmtree(cache)


if __name__ == "__main__":
    main()
