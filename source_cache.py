"""Reuse generated detector inputs across benchmark runs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def add_cache_options(parser):
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--source-cache", type=Path, default=os.environ.get("VOXEL_SOURCE_CACHE") or None,
                         help="directory for reusable generated detector inputs")
    options.add_argument("--no-source-cache", action="store_true", help="generate inputs in temporary scratch")


def prepare_source(scratch, cache, name, identity, build):
    if cache is None:
        build(scratch)
        return scratch, "disabled"

    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = cache / f"{name}-{digest}"
    with (cache / f".{destination.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_dir():
            return destination, "hit"
        with TemporaryDirectory(prefix=".building-", dir=cache) as working:
            stage = Path(working) / "source"
            stage.mkdir()
            build(stage)
            stage.rename(destination)
    return destination, "built"
