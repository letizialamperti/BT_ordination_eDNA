"""Bounded retries and verified atomic transfers for export on shared storage."""
import errno
import json
import os
from pathlib import Path
import time
import uuid

from final_checks import sha


class TransferIntegrityError(ValueError):
    pass


TRANSIENT = {5, 11, 70, 107, 110, 116, 121, errno.ECONNRESET, errno.EPIPE}


def retry(operation, description, attempts=5, delay=1., extra=()):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            recoverable = (isinstance(exc, OSError) and exc.errno in TRANSIENT) or isinstance(exc, extra)
            if not recoverable or attempt+1 == attempts:
                raise
            pause = min(delay * 2**attempt, 8.)
            print(f"I/O retry {attempt+1}/{attempts-1}: {description}: {exc}; waiting {pause:g}s", flush=True)
            time.sleep(pause)


def read_json(path, *, manifest=False):
    return retry(lambda: json.loads(Path(path).read_text()), f"read {path}",
                 extra=(json.JSONDecodeError,) if manifest else ())


def write_json(path, value):
    path = Path(path)
    payload = (json.dumps(value, indent=2)+"\n").encode()
    import hashlib
    expected = hashlib.sha256(payload).hexdigest()
    def operation():
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            if sha(temporary) != expected:
                raise TransferIntegrityError(f"JSON write hash mismatch: {temporary}")
            temporary.replace(path)
            if sha(path) != expected:
                raise TransferIntegrityError(f"JSON publication hash mismatch: {path}")
        finally:
            try:temporary.unlink(missing_ok=True)
            except OSError:pass
    retry(operation, f"write {path}", extra=(TransferIntegrityError,))


def copy_verified(source, target, expected_sha=None):
    """Check source stability, full byte count, and hash before/after rename."""
    source, target = Path(source), Path(target)
    def operation():
        import hashlib
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        before = source.stat();digest=hashlib.sha256();count=0
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, temporary.open("wb") as dst:
                for chunk in iter(lambda: src.read(8*1024*1024), b""):
                    digest.update(chunk);count+=len(chunk);dst.write(chunk)
                dst.flush();os.fsync(dst.fileno())
            after = source.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError(f"Source changed during transfer: {source}")
            value = digest.hexdigest()
            if count != before.st_size or (expected_sha is not None and value != expected_sha):
                raise TransferIntegrityError(f"Incomplete or changed source bytes: {source}")
            if sha(temporary) != value:
                raise TransferIntegrityError(f"Transferred bytes differ: {target}")
            temporary.replace(target)
            if sha(target) != value:
                raise TransferIntegrityError(f"Published bytes differ: {target}")
            return value
        finally:
            try:temporary.unlink(missing_ok=True)
            except OSError:pass
    return retry(operation, f"copy {source} -> {target}", extra=(TransferIntegrityError,))
