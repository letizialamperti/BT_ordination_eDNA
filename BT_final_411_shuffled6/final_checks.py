"""Validate outer folds and hash large checkpoints without loading them into RAM."""
import hashlib
from pathlib import Path

import pandas as pd
from pilot_checks import atomic_json, run_lock


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_roles(path, fold, smoke_test=False):
    roles = pd.read_csv(path, dtype={"spygen_code": str})
    needed = ["spygen_code", "spatial_block", "role"]
    if not set(needed).issubset(roles) or roles[needed].isna().any().any():
        raise ValueError("Missing role columns or entries.")
    if not roles.spygen_code.is_unique or (not smoke_test and len(roles) != 411):
        raise ValueError("Expected exactly 411 unique cohort IDs.")
    if any(not c or c != c.strip() or c in {".", ".."} or "/" in c or "\\" in c for c in roles.spygen_code):
        raise ValueError("Invalid sample ID.")
    if set(roles.role) != {"train", "outer_eval"} or set(roles.spatial_block) != {1, 2, 3, 4}:
        raise ValueError("Expected four spatial blocks and train/outer_eval roles.")
    expected = roles.spatial_block.eq(fold)
    if not expected.any() or not roles.role.eq("outer_eval").equals(expected):
        raise ValueError("Outer roles do not match the requested spatial fold.")
    if not smoke_test:
        cohort = pd.read_csv(Path(__file__).parent / "cohort_411.csv")
        cohort["role"] = cohort.spatial_block.map(lambda b: "outer_eval" if b == fold else "train")
        if not roles[needed].sort_values("spygen_code").reset_index(drop=True).equals(
            cohort[needed].sort_values("spygen_code").reset_index(drop=True)
        ):
            raise ValueError("Assignments differ from the frozen common 411-sample cohort.")
    return roles


def file_stats(paths):
    return {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
