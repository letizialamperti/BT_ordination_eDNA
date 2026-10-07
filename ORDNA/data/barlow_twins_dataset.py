"""Index complete pairs before splitting views; read indexed byte ranges on demand.

Input is a UTF-8 CSV with one paired-read record per physical line. Quoted
multiline records are rejected explicitly. Input files must not change after
indexing. Missing/blank/NA reads are removed as pairs, not replaced by padding.
"""
import bisect
import csv
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from ORDNA.utils.sequence_mapper import SequenceMapper


MISSING_READS = {'', 'na', 'nan', 'none', 'null', '<na>', 'n/a'}


def parse_record(raw, columns, path, line=None):
    try:
        row = next(csv.reader([raw.decode('utf-8')], strict=True))
    except (csv.Error, UnicodeError) as exc:
        raise ValueError(f'{path}, line {line}: malformed CSV or multiline record') from exc
    if not row:  # empty physical line
        return [''] * columns
    if len(row) != columns:
        raise ValueError(f'{path}, line {line}: expected {columns} columns, found {len(row)}')
    return row


def valid_pair(row, forward_idx, reverse_idx):
    forward, reverse = row[forward_idx].strip(), row[reverse_idx].strip()
    if forward.lower() in MISSING_READS or reverse.lower() in MISSING_READS:
        return None
    return forward, reverse


class BarlowTwinsDataset(Dataset):
    def __init__(self, sample_files: List[Path], sample_subset_size: int,
                 sequence_length: int) -> None:
        super().__init__()
        if sample_subset_size <= 0 or sequence_length <= 0:
            raise ValueError('Subset size and sequence length must be positive.')
        self.files = []
        self.accumulated_num_subsets = []
        self.sample_subset_size = sample_subset_size
        self.pad_seq_to_len = sequence_length
        self.sequence_mapper = SequenceMapper()
        self.qc_records = []
        self._ranges = []
        self._layouts = []
        self._file_stats = []
        required = 2 * sample_subset_size
        running = 0
        allowed = set(self.sequence_mapper.iupac_dict) - {'pad'}
        for source in sample_files:
            path = Path(source)
            before = path.stat()
            qc = dict(spygen_code=path.stem, total_rows=0, valid_paired_reads=0,
                      discarded_missing_pairs=0, minimum_required=required,
                      usable_chunks=0, unused_valid_pairs=0, reason='')
            ranges = []
            with path.open('rb') as stream:
                header_raw = stream.readline()
                if not header_raw:
                    qc['reason'] = 'empty_file'
                    self.qc_records.append(qc)
                    continue
                header = next(csv.reader([header_raw.decode('utf-8-sig')], strict=True))
                if header.count('Forward') != 1 or header.count('Reverse') != 1:
                    qc['reason'] = 'missing_or_duplicate_required_columns'
                    self.qc_records.append(qc)
                    continue
                fi, ri = header.index('Forward'), header.index('Reverse')
                count = 0
                start = None
                line = 1
                while True:
                    offset = stream.tell()
                    raw = stream.readline()
                    if not raw:
                        break
                    line += 1
                    qc['total_rows'] += 1
                    row = parse_record(raw, len(header), path, line)
                    pair = valid_pair(row, fi, ri)
                    if pair is None:
                        qc['discarded_missing_pairs'] += 1
                        continue
                    for read in pair:
                        if len(read) > sequence_length or set(read.lower()) - allowed:
                            raise ValueError(f'{path}, line {line}: unsupported nucleotide or read longer than {sequence_length}')
                    qc['valid_paired_reads'] += 1
                    if count == 0:
                        start = offset
                    count += 1
                    if count == required:
                        ranges.append((start, stream.tell()))
                        count = 0
                qc['unused_valid_pairs'] = count
                qc['usable_chunks'] = len(ranges)
                if not ranges:
                    qc['reason'] = 'insufficient_valid_paired_reads'
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Input changed during indexing: {path}')
            self.qc_records.append(qc)
            print(f'{path.name}: {qc["valid_paired_reads"]} valid pairs, '
                  f'{qc["discarded_missing_pairs"]} missing pairs discarded, '
                  f'{qc["usable_chunks"]} chunks', flush=True)
            if not ranges:
                continue
            self.files.append(path)
            self._ranges.append(np.asarray(ranges, dtype=np.int64))
            self._layouts.append((len(header), fi, ri))
            self._file_stats.append((after.st_size, after.st_mtime_ns))
            running += len(ranges)
            self.accumulated_num_subsets.append(running)

    def __len__(self):
        return self.accumulated_num_subsets[-1] if self.accumulated_num_subsets else 0

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        fidx = bisect.bisect_right(self.accumulated_num_subsets, index)
        previous = self.accumulated_num_subsets[fidx-1] if fidx else 0
        start, end = self._ranges[fidx][index-previous]
        path = self.files[fidx]
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self._file_stats[fidx]:
            raise RuntimeError(f'Input changed after indexing: {path}')
        columns, fi, ri = self._layouts[fidx]
        pairs = []
        with path.open('rb') as stream:
            stream.seek(int(start))
            while stream.tell() < end:
                row = parse_record(stream.readline(), columns, path)
                pair = valid_pair(row, fi, ri)
                if pair is not None:
                    pairs.append(pair)
        n = self.sample_subset_size
        if len(pairs) != 2*n:
            raise RuntimeError(f'{path}: indexed chunk does not contain {2*n} valid pairs')
        return self._to_tensor(pairs[:n]), self._to_tensor(pairs[n:])

    def _to_tensor(self, pairs):
        if len(pairs) != self.sample_subset_size:
            raise ValueError('Every view must contain exactly N valid paired reads.')
        forward, reverse = zip(*pairs)
        fwd = self.sequence_mapper.map_seq_list(forward, pad_to_len=self.pad_seq_to_len)
        rev = self.sequence_mapper.map_seq_list(reverse, pad_to_len=self.pad_seq_to_len)
        return torch.stack((torch.tensor(fwd, dtype=torch.long),
                            torch.tensor(rev, dtype=torch.long)), dim=1)
