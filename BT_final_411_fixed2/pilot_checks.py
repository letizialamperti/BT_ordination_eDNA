"""Input and run-isolation checks shared by pilot training and export."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

import pandas as pd


def read_roles(path, include_outer=True):
    roles = pd.read_csv(path, dtype={"spygen_code": str})
    columns = {"spygen_code", "role", "spatial_block"}
    if not columns.issubset(roles.columns) or roles[list(columns)].isna().any().any():
        raise ValueError(f"Missing role columns or values: {path}")
    if not roles.spygen_code.is_unique:
        raise ValueError(f"Duplicate sample IDs: {path}")
    if any(not code or code in {".", ".."} or "/" in code or "\\" in code
           or code != code.strip() for code in roles.spygen_code):
        raise ValueError(f"Unsafe or whitespace-padded sample ID: {path}")
    expected = {"inner_train", "inner_validation"}
    if include_outer:
        expected.add("outer_eval")
    if set(roles.role) != expected:
        raise ValueError(f"Expected roles {sorted(expected)}: {path}")
    blocks = pd.to_numeric(roles.spatial_block, errors="raise")
    if not blocks.isin([1, 2, 3, 4]).all():
        raise ValueError("Spatial blocks must be integers 1 through 4.")
    roles["spatial_block"] = blocks.astype(int)
    if (roles.groupby("spatial_block").role.nunique() != 1).any():
        raise ValueError("A spatial block occurs in multiple roles.")
    groups = roles.groupby("role").spatial_block.nunique()
    if groups["inner_train"] != 2 or groups["inner_validation"] != 1:
        raise ValueError("Expected two training blocks and one inner validation block.")
    if include_outer and (groups["outer_eval"] != 1 or roles.spatial_block.nunique() != 4):
        raise ValueError("Expected one excluded outer block.")
    return roles


@contextmanager
def run_lock(directory, name=".training.lock"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {directory} ({name}).") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
