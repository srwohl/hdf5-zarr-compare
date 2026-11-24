"""Small, explicit NeXus/HDF5 versus sharded OME-Zarr experiment.

All volume indices and shapes are [z,y,x]. No MPI, GPU, or GPFS dependency.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import time

import h5py
import numpy as np
import zarr

from phantom import ELLIPSOIDS, make_sources

AXES = ("z", "y", "x")
DTYPE = np.dtype("<u2")
MIB = 1024 ** 2


def triple(value):
    try:
        result = tuple(int(x) for x in value.split(","))
        if len(result) != 3 or min(result) < 1:
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError("expected three positive integers: z,y,x")


def geometry(config, level):
    n = config["size"] // 2 ** level
    chunks = tuple(min(v, n) for v in config["chunks"])
    shards = tuple(min(v, n) for v in config["shards"])
    return (n,) * 3, chunks, shards


def regions(shape, shard):
    for origin in itertools.product(*(range(0, n, s) for n, s in zip(shape, shard))):
        yield tuple(slice(lo, min(lo + s, n)) for lo, s, n in zip(origin, shard, shape))


def nxgroup(parent, name, nxclass):
    group = parent.create_group(name)
    group.attrs["NX_class"] = nxclass
    return group


def nxdata(parent, name, array, start, spacing):
    """Attach NXdata axes to a real or virtual dataset; coordinates are centres."""
    group = nxgroup(parent, name, "NXdata")
    group["data"] = array
    group.attrs["signal"] = "data"
    group.attrs["axes"] = list(AXES)
    for axis, axis_name in enumerate(AXES):
        values = (np.arange(array.shape[axis]) + start[axis] + .5) * spacing
        coord = group.create_dataset(axis_name, data=values)
        coord.attrs["units"] = "um"
        group.attrs[f"{axis_name}_indices"] = axis
    array.attrs["units"] = "counts"
    array.attrs["description"] = "Synthetic relative intensity, 0..1 mapped to 0..65535"
    return group


def write_part(job):
    kind, output, level, bounds, config = job
    started = time.perf_counter()
    directory = Path(output)
    selection = tuple(slice(a, b) for a, b in bounds)
    source_path = Path(config.get("source_directory", directory / "source"))
    source = np.load(source_path / f"{level}.npy", mmap_mode="r")
    block = np.ascontiguousarray(source[selection])
    if kind == "zarr":
        concurrency = config.get("zarr_concurrency", zarr.config.get("async.concurrency"))
        with zarr.config.set({"async.concurrency": concurrency}):
            array = zarr.open_array(str(directory / "phantom.ome.zarr" / str(level)), mode="r+")
            array[selection] = block  # one complete shard; no shared-shard writers
    else:
        _, chunks, _ = geometry(config, level)
        stem = "_".join(str(a) for a, _ in bounds)
        path = directory / "nexus" / f"level_{level}" / f"part_{stem}.nxs"
        options = {} if config["compression"] == "none" else {"compression": "gzip", "compression_opts": 1}
        with h5py.File(path, "w") as f:
            f.attrs["default"] = "entry"
            entry = nxgroup(f, "entry", "NXentry")
            entry.attrs["default"] = "data"
            entry.create_dataset("title", data="Synthetic 3D ellipsoid phantom partition")
            dataset = f.create_dataset("voxels", data=block, chunks=chunks, **options)
            nxdata(entry, "data", dataset, [a for a, _ in bounds],
                   config["voxel_um"] * 2 ** level)
            del f["voxels"]  # retain one link at /entry/data/data
    return time.perf_counter() - started


def create_nexus_master(directory, config):
    with h5py.File(directory / "nexus" / "phantom.nxs", "w", libver="latest") as f:
        f.attrs["default"] = "entry"
        entry = nxgroup(f, "entry", "NXentry")
        entry.attrs["default"] = "data"
        entry.create_dataset("title", data=config.get("image_name", "Synthetic 3D ellipsoid phantom; no measured acquisition"))
        process = nxgroup(entry, config.get("process_name", "generation"), "NXprocess")
        process.create_dataset("program", data="voxel-io-demo")
        process.create_dataset("version", data="0.2.0")
        process.create_dataset("date", data=datetime.now(timezone.utc).isoformat())
        process.create_dataset("parameters", data=json.dumps({k: v for k, v in config.items() if k != "source_directory"}))
        levels = nxgroup(entry, "levels", "NXcollection")
        for level in range(config["levels"]):
            shape, _, shard = geometry(config, level)
            layout = h5py.VirtualLayout(shape=shape, dtype=DTYPE)
            for selection in regions(shape, shard):
                stem = "_".join(str(s.start) for s in selection)
                name = f"level_{level}/part_{stem}.nxs"
                part_shape = tuple(s.stop - s.start for s in selection)
                layout[selection] = h5py.VirtualSource(name, "/entry/data/data", shape=part_shape)
            ds = f.create_virtual_dataset(f"virtual_{level}", layout, fillvalue=0)
            nxdata(levels, str(level), ds, (0, 0, 0), config["voxel_um"] * 2 ** level)
            del f[f"virtual_{level}"]
        entry["data"] = levels["0"]
        entry.attrs["pyramid_levels"] = [f"levels/{i}" for i in range(config["levels"])]
        entry.attrs["pyramid_convention"] = "voxel-io-demo; NXdata per level, not an OME schema"


def sync_tree(directory):
    """Explicit fsync time is separate from construction; includes directories."""
    started = time.perf_counter()
    for base, _, filenames in os.walk(directory, topdown=False):
        for name in filenames:
            with open(Path(base) / name, "rb") as f:
                os.fsync(f.fileno())
        fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return time.perf_counter() - started


def build(args):
    config = {key: getattr(args, key) for key in
              ("size", "levels", "chunks", "shards", "noise", "seed", "voxel_um", "compression", "workers")}
    # Power-of-two dimensions keep edge geometry explicit and identical in both formats.
    power_two = lambda v: v > 0 and v & (v - 1) == 0
    if not power_two(args.size) or args.size < 16:
        raise ValueError("size must be a power of two, at least 16")
    if not 1 <= args.levels <= args.size.bit_length():
        raise ValueError("too many or too few levels")
    if not np.isfinite(args.noise) or not np.isfinite(args.voxel_um) or args.noise < 0 or args.voxel_um <= 0 or args.workers < 1 or args.seed < 0:
        raise ValueError("noise must be non-negative; voxel spacing and workers must be positive")
    for c, s in zip(args.chunks, args.shards):
        if not power_two(c) or not power_two(s) or c > s or s > args.size:
            raise ValueError("chunks and shards must be powers of two with chunk <= shard <= size")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)  # refuse to overwrite any existing dataset
    (directory / "nexus").mkdir()
    manifest = dict(config, complete=False, axes=AXES, dtype="uint16", gpfs_block_mib=8,
                    ellipsoids=ELLIPSOIDS, host=platform.node(),
                    versions={"numpy": np.__version__, "h5py": h5py.__version__, "zarr": zarr.__version__})
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    make_sources(directory / "source", args.size, args.levels, args.slab_z, args.noise, args.seed)
    timings = {"source_and_pyramids_seconds": time.perf_counter() - started}
    timings.update(write_formats(directory, config, args.write_order, args.sync))
    manifest.update(complete=True, write_order=args.write_order, sync=args.sync, timings=timings)
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Ready: {directory}")


def ome_multiscales(config, name, method):
    datasets = []
    for level in range(config["levels"]):
        step = config["voxel_um"] * 2 ** level
        datasets.append({"path": str(level), "coordinateTransformations": [
            {"type": "scale", "scale": [step] * 3},
            {"type": "translation", "translation": [step / 2] * 3}]})
    return {"name": name, "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in AXES],
            "datasets": datasets, "type": method}


def write_formats(directory, config, write_order="nexus,zarr", sync=False):
    """Export an existing source pyramid; one writer owns each physical part."""
    timings = {}
    for kind in write_order.split(","):
        started = time.perf_counter()
        if kind == "zarr":
            root = zarr.open_group(str(directory / "phantom.ome.zarr"), mode="w", zarr_format=3)
            for level in range(config["levels"]):
                shape, chunks, shards = geometry(config, level)
                root.create_array(str(level), shape=shape, dtype=DTYPE, chunks=chunks, shards=shards,
                                  compressors=None if config["compression"] == "none" else [zarr.codecs.GzipCodec(level=1)],
                                  dimension_names=AXES, config={"write_empty_chunks": True})
            root.attrs["ome"] = {"version": "0.5", "multiscales": [
                ome_multiscales(config, config.get("image_name", "3D ellipsoid phantom"), "local_mean")]}
            root.attrs["generation"] = {k: v for k, v in config.items() if k != "source_directory"}
        jobs = []
        for level in range(config["levels"]):
            shape, _, shard = geometry(config, level)
            if kind == "nexus":
                (directory / "nexus" / f"level_{level}").mkdir()
            for selection in regions(shape, shard):
                jobs.append((kind, str(directory), level, [(s.start, s.stop) for s in selection], config))
        if config["workers"] == 1:
            part_times = list(map(write_part, jobs))
        else:
            with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
                part_times = list(pool.map(write_part, jobs))
        if kind == "nexus":
            create_nexus_master(directory, config)
        timings[f"{kind}_construct_seconds"] = time.perf_counter() - started
        timings[f"{kind}_part_task_seconds_sum"] = sum(part_times)
        timings[f"{kind}_fsync_seconds"] = sync_tree(
            directory / ("nexus" if kind == "nexus" else "phantom.ome.zarr")) if sync else None
        print(f"{kind}: {len(jobs)} partitions, {timings[f'{kind}_construct_seconds']:.3f} s", flush=True)
    return timings


def load_manifest(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("construction did not complete; use a new output directory")
    return manifest


@contextmanager
def open_levels(directory, kind, count):
    if kind == "nexus":
        with h5py.File(directory / "nexus" / "phantom.nxs", "r", rdcc_nbytes=0) as f:
            yield [f[f"entry/levels/{i}/data"] for i in range(count)]
    else:
        root = zarr.open_group(str(directory / "phantom.ome.zarr"), mode="r")
        yield [root[str(i)] for i in range(count)]


def check(args):
    config = load_manifest(args.directory)
    for kind in ("nexus", "zarr"):
        with open_levels(args.directory, kind, config["levels"]) as arrays:
            for level, array in enumerate(arrays):
                source = np.load(args.directory / "source" / f"{level}.npy", mmap_mode="r")
                shape, _, shards = geometry(config, level)
                for selection in regions(shape, shards):
                    np.testing.assert_array_equal(array[selection], source[selection])
                print(f"{kind} level {level}: exact match", flush=True)


def touched(selection, tile):
    return math.prod((s.stop - 1) // t - s.start // t + 1 for s, t in zip(selection, tile))


def bench(args):
    config = load_manifest(args.directory)
    rows = []
    # Reopen at each level, but keep each handle open across reads. First != cold.
    for level in range(config["levels"]):
        shape, chunks, shards = geometry(config, level)
        n = shape[0]
        rng = np.random.default_rng(args.seed + level)
        trace = []
        for trial in range(args.reads):
            for axis, pattern in enumerate(("xy", "xz", "yz")):
                selection = [slice(0, n)] * 3
                index = int(rng.integers(n))
                selection[axis] = slice(index, index + 1)
                trace.append((trial, pattern, tuple(selection)))
            width = max(1, n // 8)
            starts = rng.integers(0, n - width + 1, size=3)
            trace.append((trial, "roi", tuple(slice(int(s), int(s) + width) for s in starts)))
        # Alternate which format reads first at each level; identical trace and no warm-up.
        order = ("nexus", "zarr") if level % 2 == 0 else ("zarr", "nexus")
        reference_sums = []
        for ki, kind in enumerate(order):
            started = time.perf_counter()
            with open_levels(args.directory, kind, config["levels"]) as arrays:
                open_seconds = time.perf_counter() - started
                sums = []
                for trial, pattern, selection in trace:
                    started_ns = time.perf_counter_ns()
                    data = np.asarray(arrays[level][selection])
                    ns = time.perf_counter_ns() - started_ns
                    checksum = int(data.sum(dtype=np.uint64))  # outside timed selection
                    sums.append(checksum)
                    count = touched(selection, chunks)
                    rows.append(dict(format=kind, level=level, pattern=pattern, trial=trial,
                        elapsed_ms=ns / 1e6, open_seconds=open_seconds, useful_bytes=data.nbytes,
                        chunks_intersected=count, full_chunk_model_bytes=count * math.prod(chunks) * 2,
                        files_or_shards_intersected=touched(selection, shards),
                        selection=json.dumps([[s.start, s.stop] for s in selection]), checksum=checksum,
                        cache_state="uncontrolled", host=platform.node()))
                if ki == 0:
                    reference_sums = sums
                elif reference_sums != sums:
                    raise ValueError("formats returned different data; run check")
    output = args.output or args.directory / "reads.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for level in range(config["levels"]):
        for pattern in ("xy", "xz", "yz", "roi"):
            values = []
            for kind in ("nexus", "zarr"):
                times = [r["elapsed_ms"] for r in rows if (r["format"], r["level"], r["pattern"]) == (kind, level, pattern)]
                values.append(f"{kind} {np.median(times):.2f} ms")
            print(f"level {level} {pattern}: " + ", ".join(values))
    print(f"Trace: {output}; full_chunk_model_bytes is a model, not measured GPFS traffic")


def view(args):
    import matplotlib
    matplotlib.use("TkAgg" if args.show else "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    config = load_manifest(args.directory)
    if not 0 <= args.level < config["levels"]:
        raise ValueError("level outside pyramid")
    with open_levels(args.directory, args.format, config["levels"]) as arrays:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
        fig.subplots_adjust(bottom=.32 if args.show else .13)
        artists = []
        def update(level, fractions):
            n = arrays[level].shape[0]
            indices = [min(n - 1, int(f * n)) for f in fractions]
            selections = [(indices[0], slice(None), slice(None)),
                          (slice(None), indices[1], slice(None)),
                          (slice(None), slice(None), indices[2])]
            extent = config["size"] * config["voxel_um"]
            for i, (ax, selection, label, labels) in enumerate(zip(
                    axes, selections, ("XY", "XZ", "YZ"), (("x", "y"), ("x", "z"), ("y", "z")))):
                image = arrays[level][selection]
                if len(artists) < 3:
                    artists.append(ax.imshow(image.reshape(n, n), cmap="gray", origin="lower", vmin=0, vmax=65535,
                                             extent=(0, extent, 0, extent)))
                else:
                    artists[i].set_data(image.reshape(n, n))
                ax.set_title(f"{label}: {AXES[i]} index {indices[i]}")
                ax.set_xlabel(f"{labels[0]} (µm)")
                ax.set_ylabel(f"{labels[1]} (µm)")
            fig.suptitle(f"{args.format} · level {level} · {n}³ voxels · {config.get('image_name', 'true 3D ellipsoids')}")
            fig.canvas.draw_idle()
        update(args.level, [.5] * 3)
        output = args.output or args.directory / f"preview-{args.format}.png"
        fig.savefig(output, dpi=150)
        print(output)
        if args.show:
            sliders = [Slider(fig.add_axes((.16, .23 - i * .045, .68, .022)), a, 0., 1., valinit=.5) for i, a in enumerate(AXES)]
            if config["levels"] > 1:
                scale = Slider(fig.add_axes((.16, .095, .68, .022)), "level", 0, config["levels"] - 1,
                               valinit=args.level, valstep=1)
                scale.on_changed(lambda _: update(int(scale.val), [s.val for s in sliders]))
            for slider in sliders:
                slider.on_changed(lambda _: update(int(scale.val) if config["levels"] > 1 else 0, [s.val for s in sliders]))
            plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build", help="generate sources, pyramids and both formats")
    p.add_argument("directory", type=Path, nargs="?", default=Path("data/demo"))
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--chunks", type=triple, default=(32, 64, 64))
    p.add_argument("--shards", type=triple, default=(128, 128, 128))
    p.add_argument("--slab-z", type=int, default=8)
    p.add_argument("--voxel-um", type=float, default=1.)
    p.add_argument("--noise", type=float, default=0.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compression", choices=("none", "gzip"), default="none")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--sync", action="store_true", help="time fsync separately for each format")
    p.add_argument("--write-order", choices=("nexus,zarr", "zarr,nexus"), default="nexus,zarr")
    p.set_defaults(function=build)
    for name, function in (("check", check), ("bench", bench), ("view", view)):
        p = commands.add_parser(name)
        p.add_argument("directory", type=Path, nargs="?", default=Path("data/demo"))
        p.set_defaults(function=function)
        if name == "bench":
            p.add_argument("--reads", type=int, default=8)
            p.add_argument("--seed", type=int, default=123)
            p.add_argument("--output", type=Path)
        if name == "view":
            p.add_argument("--format", choices=("nexus", "zarr"), default="zarr")
            p.add_argument("--level", type=int, default=0)
            p.add_argument("--show", action="store_true")
            p.add_argument("--output", type=Path)
    args = parser.parse_args()
    if getattr(args, "reads", 1) < 1 or getattr(args, "slab_z", 1) < 1:
        parser.error("reads and slab-z must be positive")
    try:
        args.function(args)
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
