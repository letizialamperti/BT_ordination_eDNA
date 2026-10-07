"""Bounded GPU smoke test; never saves a scientific training checkpoint."""
import argparse
import csv
import json
import random
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path

from training_BarlowTwins_kfold import load_folds, make_roles, MergeSingletonBatchSampler


def select_training_codes(folds, fold, seed, count):
    roles = make_roles(folds, fold)
    codes = [code for code, role in roles.items() if role == 'train']
    return random.Random(seed).sample(codes, min(count, len(codes)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples_dir', type=Path, required=True)
    p.add_argument('--split_dir', type=Path, default=Path(__file__).parent / 'k_cross')
    p.add_argument('--outer_fold', type=int, choices=range(1, 5), default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sample_count', type=int, default=8)
    p.add_argument('--max_steps', type=int, default=20)
    p.add_argument('--num_workers', type=int, default=12)
    p.add_argument('--output_root', type=Path, default=Path('runs_bt_diagnostic'))
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    if args.sample_count < 1 or args.max_steps < 1 or args.num_workers < 0:
        p.error('sample_count/max_steps must be positive; num_workers nonnegative.')
    _, folds = load_folds(args.split_dir)
    codes = select_training_codes(folds, args.outer_fold, args.seed, args.sample_count)
    print(f'DIAGNOSTIC ONLY: fold {args.outer_fold}; selected training samples: {codes}', flush=True)
    if args.dry_run:
        return

    import pandas as pd
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, RandomSampler
    from pytorch_lightning.loggers import CSVLogger
    from ORDNA.data.barlow_twins_dataset import BarlowTwinsDataset
    from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder

    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU visible in the allocated job.')
    run = args.output_root / f'fold_{args.outer_fold:02d}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(selected_training_samples=codes, diagnostic_only=True,
                  N=500, B=32, sample_repr_dim=256, sample_emb_dim=64,
                  lmbda=0.005, initial_learning_rate=0.001, weight_decay=0.0001,
                  max_read_pairs_per_sample=128000,
                  torch_version=torch.__version__, lightning_version=pl.__version__)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    pl.seed_everything(args.seed, workers=True)

    class Timing(pl.Callback):
        def __init__(self):
            self.times = []
            self.last = None

        def on_train_start(self, trainer, model):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.last = time.perf_counter()
            print(f'GPU: {torch.cuda.get_device_name()}', flush=True)

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = now - self.last
            self.last = now
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            if loss is None or not torch.isfinite(loss).all():
                raise RuntimeError('Non-finite or missing training loss.')
            self.times.append(elapsed)
            peak = torch.cuda.max_memory_allocated() / 2**30
            with (run / 'batch_timing.csv').open('a', newline='') as stream:
                writer = csv.writer(stream)
                if len(self.times) == 1:
                    writer.writerow(['batch', 'seconds', 'loss', 'peak_allocated_GiB'])
                writer.writerow([len(self.times), elapsed, float(loss.detach().cpu()), peak])
            print(f'Batch {len(self.times)}: {elapsed:.2f}s; peak allocated {peak:.2f} GiB', flush=True)

    timer = Timing()
    preparation_started = time.perf_counter()
    # Bounded copies avoid counting/scanning the full 324-sample dataset at startup.
    # Original files are opened read-only. Temporary inputs disappear after the test.
    with tempfile.TemporaryDirectory(prefix='inputs_', dir=run) as temp:
        files = []
        for code in codes:
            src = args.samples_dir / f'{code}.csv'
            print(f'Reading bounded input: {src.name}', flush=True)
            data = pd.read_csv(src, nrows=128000, usecols=['Forward', 'Reverse'])
            if len(data) < 1000:
                print(f'Warning: {code} has fewer than 1000 paired reads ({len(data)}).', flush=True)
            if data.isna().any().any() or data.apply(lambda col: col.astype(str).str.strip().eq('')).any().any():
                raise ValueError(f'{code}: missing or empty sequence in diagnostic input.')
            dest = Path(temp) / src.name
            data.to_csv(dest, index=False)
            files.append(dest)
        dataset = BarlowTwinsDataset(files, sample_subset_size=500, sequence_length=300)
        loader = DataLoader(dataset, batch_sampler=MergeSingletonBatchSampler(RandomSampler(dataset), 32),
                            num_workers=args.num_workers, pin_memory=True)
        if len(dataset) < 32:
            raise ValueError('Too few chunks to test the requested B=32.')
        print(f'Bounded dataset: {len(dataset)} chunks; {len(loader)} batches per pass.', flush=True)
        preparation_seconds = time.perf_counter() - preparation_started
        model = SelfAttentionBarlowTwinsEmbedder(token_emb_dim=8, seq_len=300,
                    sample_repr_dim=256, sample_emb_dim=64, lmbda=0.005,
                    initial_learning_rate=0.001, weight_decay=0.0001)
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_steps=args.max_steps,
                             max_epochs=-1, enable_checkpointing=False, limit_val_batches=0,
                             num_sanity_val_steps=0, logger=CSVLogger(str(run), name='metrics'),
                             callbacks=[timer], log_every_n_steps=1, enable_progress_bar=False)
        start = time.perf_counter()
        trainer.fit(model, train_dataloaders=loader)
        summary = dict(completed_steps=trainer.global_step,
                       preparation_seconds=preparation_seconds,
                       fit_seconds=time.perf_counter()-start,
                       median_batch_seconds_after_first=statistics.median(timer.times[1:] or timer.times),
                       peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                       gpu=torch.cuda.get_device_name(), diagnostic_only=True)
        (run / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    print(f'Diagnostic complete: {run}. No model checkpoint saved.', flush=True)


if __name__ == '__main__':
    main()
