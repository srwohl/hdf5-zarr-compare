"""Run a matched four-case beamline replay, reconstruction and ROI comparison."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from inspect import getsource
import itertools
import json
import math
import os
import resource
from pathlib import Path
import shlex
import socket
import subprocess
from tempfile import TemporaryDirectory
import time
import zlib

import h5py
import numpy as np
import skimage
import zarr

import demo
import feature
import replay
import tiled
from experiment_report import context_plot, make_report
from phantom import make_pyramid
from source_cache import add_cache_options, prepare_source

CASES = {"A": ("nexus", "acquisition"), "B": ("nexus", "bricks"),
         "C": ("zarr", "acquisition"), "D": ("zarr", "bricks")}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    if rows:
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=next(iter(rows)).keys())
            writer.writeheader()
            writer.writerows(rows)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def mount_info(path):
    try:
        result = subprocess.run(["findmnt", "--json", "--target", str(path),
                                 "--output", "FSTYPE,SOURCE,TARGET"], capture_output=True, text=True,
                                timeout=10, check=True)
        return json.loads(result.stdout)["filesystems"][0]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        return {}


class Timeline:
    def __init__(self, directory, command):
        self.directory = directory
        self.command = shlex.split(command) if command else []
        self.rows = []

    def sample(self):
        if self.command:
            started = time.perf_counter()
            try:
                output = subprocess.run(self.command, check=True, capture_output=True, text=True, timeout=10)
                counters = json.loads(output.stdout)
                if not isinstance(counters, dict) or not counters or not all(
                        type(value) in (int, float) and math.isfinite(value) for value in counters.values()):
                    raise ValueError("counter command must return a JSON object of finite numeric values")
                return dict(utc=utc_now(), counters=counters, sample_seconds=time.perf_counter() - started)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                details = getattr(error, "stderr", "") or ""
                if isinstance(details, bytes):
                    details = details.decode(errors="replace")
                return dict(utc=utc_now(), error=str(error), stderr=details.strip()[-2000:],
                            sample_seconds=time.perf_counter() - started)
        return {}

    @staticmethod
    def cpu_seconds():
        usage = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
        return sum(r.ru_utime for r in usage), sum(r.ru_stime for r in usage)

    @contextmanager
    def phase(self, case, phase, *, counters=True):
        before = self.sample() if counters else {}
        timestamp = utc_now()
        started = time.perf_counter()
        user_before, system_before = self.cpu_seconds()
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            user_after, system_after = self.cpu_seconds()
            end = utc_now()
            after = self.sample() if counters else {}
            delta = {}
            for key, value in before.get("counters", {}).items():
                last = after.get("counters", {}).get(key)
                delta[key] = last - value if last is not None and last >= value else None
            self.rows.append(dict(case=case, phase=phase, utc_start=timestamp, utc_end=end, seconds=seconds,
                                  cpu_user_seconds=user_after - user_before,
                                  cpu_system_seconds=system_after - system_before,
                                  before=before, after=after, counter_delta=delta))
            save_json(self.directory / "timeline.json", self.rows)


def configurations(args):
    n = args.size
    if n < 32 or n & (n - 1):
        raise ValueError("size must be a power of two, at least 32")
    if not 2 <= args.levels <= n.bit_length() - 4:
        raise ValueError("use at least two levels, with at least 16 voxels per overview axis")
    if not math.isfinite(args.fps) or args.fps <= 0 or min(args.workers, args.read_workers, args.write_workers,
                                                      args.zarr_concurrency, args.queue_frames, args.row_band) < 1:
        raise ValueError("fps, workers, queue capacity and row band must be positive")
    if (args.frame_chunk < 1 or args.angle_block < 1 or args.angles < 2 or
            args.angles % args.angle_block or args.angle_block % (4 * args.frame_chunk)):
        raise ValueError("angles must divide into angle blocks, each a multiple of four times frame-chunk")
    if args.partition_factor < 1 or args.partition_factor & (args.partition_factor - 1):
        raise ValueError("partition factor must be a positive power of two")
    if len(args.case_order) != len(CASES) or set(args.case_order) != set(CASES):
        raise ValueError("case order must contain A, B, C and D exactly once")
    byte_count = args.chunk_mib * demo.MIB
    if not math.isfinite(byte_count) or not byte_count.is_integer() or not 2 <= byte_count <= n**3 * 2:
        raise ValueError("chunk MiB must specify a whole power-of-two byte count that fits the volume")
    byte_count = int(byte_count)
    if byte_count & (byte_count - 1):
        raise ValueError("chunk bytes must be a power of two")
    exponent = (byte_count // 2).bit_length() - 1
    x = min(n, 2 ** ((exponent + 1) // 2))
    y = min(n, 2 ** exponent // x)
    slabs = (2 ** exponent // (x * y), y, x)
    bricks = tuple(2 ** ((exponent + axis) // 3) for axis in range(3))
    partitions = tuple(max(a, b) for a, b in zip(slabs, bricks))
    partitions = (min(n, partitions[0] * args.partition_factor), *partitions[1:])
    width, overlap = 5 * n // 8, n // 4
    starts = tiled.tile_starts(n, width, overlap)
    raw = dict(size=n, tile=width, overlap=overlap, tile_origins_yx_px=list(itertools.product(starts, starts)),
               angles_deg=np.linspace(0, 180, args.angles, endpoint=False).tolist(), angle_block=args.angle_block,
               row_band=args.row_band, photons=20000., dark=100., mu_per_voxel=2. / n,
               seed=args.seed, poisson=args.poisson, compression=args.compression, workers=args.write_workers,
               read_workers=args.read_workers, zarr_concurrency=args.zarr_concurrency,
               fps=args.fps, overflow=args.overflow, queue_frames=args.queue_frames, sync=args.sync,
               cache_state="uncontrolled", detector_bricks=args.detector_bricks)
    image = dict(size=n, levels=args.levels, shards=partitions, compression=args.compression,
                 voxel_um=1., workers=args.write_workers, zarr_concurrency=args.zarr_concurrency,
                 image_name="Reconstructed tiled phantom",
                 process_name="reconstruction",
                 reconstruction="Filtered backprojection from stored overlapping detector tiles")
    brick_chunks = (4 * args.frame_chunk, width // 2, width // 2)
    if args.detector_bricks == "sinogram":
        brick_chunks = tiled.sinogram_chunks(width, args.angle_block, args.frame_chunk)
    layouts = {"acquisition": dict(chunks=slabs, raw_chunks=(args.frame_chunk, width, width)),
               "bricks": dict(chunks=bricks, raw_chunks=brick_chunks)}
    return raw, image, layouts


def verify_raw(directory, raw, kind, sources, raw_source):
    with tiled.raw_reader(directory, raw, kind, []) as (read, calibration):
        width, block = raw["tile"], raw["angle_block"]
        for tile, reference in enumerate(sources):
            for name, values in zip(("flat", "dark"), calibration[tile]):
                np.testing.assert_array_equal(values, np.load(raw_source / f"{name}_{tile}.npy"))
            for start in range(0, len(reference), block):
                selection = (slice(start, start + block), slice(0, width), slice(0, width))
                np.testing.assert_array_equal(read(tile, selection, "verify"), reference[selection])


def prepare_pyramid(reconstruction, directory, levels):
    """Hash the floating-point reconstruction and quantize a staged pyramid."""
    directory.mkdir()
    source = np.load(reconstruction, mmap_mode="r")
    result = np.lib.format.open_memmap(directory / "0.npy", mode="w+", dtype="<u2", shape=source.shape)
    digest = hashlib.sha256()
    for start in range(0, len(source), 8):
        slab = source[start:start + 8]
        digest.update(memoryview(slab).cast("B"))
        result[start:start + 8] = np.rint(np.clip(slab, 0, 1) * 65535).astype("<u2")
    result.flush()
    del result
    make_pyramid(directory, levels)
    return digest.hexdigest()


def footprint(path):
    files = logical = allocated = 0
    for file in path.rglob("*"):
        if file.is_file():
            info = file.stat()
            files += 1
            logical += info.st_size
            allocated += info.st_blocks * 512
    return dict(files=files, logical_bytes=logical, allocated_bytes=allocated)


def probe_deflate(kind, image, center, repeats=3):
    """Time an in-memory codec reference using the chunk containing the ROI seed."""
    _, chunks, _ = demo.geometry(image, 0)
    selection = tuple(slice(i // c * c, (i // c + 1) * c) for i, c in zip(center, chunks))
    source = np.load(Path(image["source_directory"]) / "0.npy", mmap_mode="r")
    block = source[selection].tobytes()
    wrapper = 31 if kind == "zarr" else 15
    started = time.perf_counter()
    for _ in range(repeats):
        encoded = zlib.compress(block, level=1, wbits=wrapper)
    encode = (time.perf_counter() - started) / repeats
    started = time.perf_counter()
    for _ in range(repeats):
        decoded = zlib.decompress(encoded, wbits=wrapper)
    decode = (time.perf_counter() - started) / repeats
    if decoded != block:
        raise ValueError("codec probe did not round-trip")
    return dict(decoded_bytes=len(decoded), encoded_bytes=len(encoded),
                encode_seconds=encode, decode_seconds=decode, repeats=repeats)


def save_replay_trace(directory, partitions, times, tiles):
    save_json(directory / "publications.json", partitions)
    save_csv(directory / "frames.csv", [
        dict(frame_id=i, angle_index=i // tiles, tile=i % tiles, received_s=arrival, scheduled_lateness_s=late,
             blocked_s=max(0., blocked), dropped=blocked < 0, published_s=published if np.isfinite(published) else "",
             latency_ms=(published - arrival) * 1000 if np.isfinite(published) else "")
        for i, (arrival, late, blocked, published) in enumerate(times)])


def run_case(case, directory, scratch, raw_source, sources, raw, image, args, timeline):
    kind, layout = CASES[case]
    directory.mkdir()
    result = dict(case=case, format=kind, layout=layout, raw_chunks=raw["raw_chunks"],
                  image_chunks=image["chunks"], image_partitions=image["shards"])
    with timeline.phase(case, "raw_setup"):
        writer = replay.RawWriter(directory, raw, kind, raw_source)
    with timeline.phase(case, "raw_replay"):
        stats, partitions, times, origin = replay.run_replay(writer, sources, raw)
    result["replay"] = stats
    print(f"{case} {kind}/{layout}: {stats['published_frames']}/{stats['expected_frames']} frames published", flush=True)
    if stats["missing_frames"]:
        save_replay_trace(directory, partitions, times, len(sources))
        result["raw_storage"] = footprint(writer.root_path)
        result.update(verified=False, status="missing frames")
        return result

    raw_trace = []
    reconstruction = scratch / "reconstruction.npy"
    result["detector_reads"] = {}
    with timeline.phase(case, "reconstruct"):
        tiled.reconstruct(directory, raw, kind, raw_trace, source=raw_source, output=reconstruction,
                          read_metrics=result["detector_reads"])

    with timeline.phase(case, "pyramid", counters=False):
        result["reconstruction_sha256"] = prepare_pyramid(reconstruction, scratch / "pyramid", image["levels"])
    image = dict(image, source_directory=str(scratch / "pyramid"))
    with timeline.phase(case, "image_export"):
        if kind == "nexus":
            (directory / "nexus").mkdir()
        result["image_export"] = demo.write_formats(directory, image, kind, sync=args.sync)
        if args.sync:
            replay.sync_directory(directory)

    level = image["levels"] - 1
    step = 2 ** level
    trace = []
    with timeline.phase(case, "analyse"):
        analysis_started = open_started = time.perf_counter()
        with demo.open_levels(directory, kind, image["levels"]) as pyramid:
            result["analysis_open_seconds"] = time.perf_counter() - open_started
            overview_selection = (slice(0, raw["size"] // step),) * 3
            overview = feature.read_volume(pyramid[level], overview_selection, image, level, "overview", trace)
            overview_ready = time.perf_counter()
            bounds, seed, roi = feature.locate(overview, step, raw["size"])
            values = feature.read_volume(pyramid[0], roi, image, 0, "roi", trace)
            mask = feature.segment(values, seed, roi)
            analysed = time.perf_counter()
    result.update(first_overview_seconds=overview_ready - origin, analysed_roi_seconds=analysed - origin,
                  first_overview_after_analysis_start_seconds=overview_ready - analysis_started,
                  analysed_roi_after_analysis_start_seconds=analysed - analysis_started,
                  overview_level=level, specimen_bounds_overview=bounds, seed_zyx=seed.tolist(),
                  roi_zyx=[[s.start, s.stop] for s in roi], reads=trace)
    with timeline.phase(case, "publish_labels"):
        label_path = feature.write_labels(directory, kind, image, mask, roi)
        if args.sync:
            if kind == "zarr":
                demo.sync_tree(label_path)
            else:
                with label_path.open("rb") as handle:
                    os.fsync(handle.fileno())
            replay.sync_directory(label_path.parent)
        result["labels_ready_seconds"] = time.perf_counter() - origin

    with timeline.phase(case, "verify"):
        verify_raw(directory, raw, kind, sources, raw_source)
        feature.verify_volume(directory, kind, image, mask, roi)
        metrics = feature.accuracy(mask, raw["size"], roi, image["voxel_um"])
        passed = (metrics["dice"] >= args.min_dice and metrics["truth_coverage_fraction"] >= .98
                  and abs(metrics["volume_error_fraction"]) <= args.max_volume_error)
        result.update(accuracy=metrics, verified=passed, status="verified" if passed else "accuracy gate failed")
        result["verified_seconds"] = time.perf_counter() - origin

    save_replay_trace(directory, partitions, times, len(sources))
    save_csv(directory / "raw-reads.csv", raw_trace)
    save_csv(directory / "analysis-reads.csv", trace)
    result["raw_storage"] = footprint(writer.root_path)
    result["mask_sha256"] = hashlib.sha256(memoryview(mask).cast("B")).hexdigest()
    if args.compression == "gzip":
        result["deflate_probe"] = probe_deflate(kind, image, seed)
    context_plot(directory / "result.png", overview, step, bounds, values, mask, roi, seed, metrics)
    target = directory / ("nexus" if kind == "nexus" else "phantom.ome.zarr")
    result["image_and_label_storage"] = footprint(target)
    if kind == "zarr" and args.viewer_python:
        started = time.perf_counter()
        try:
            check = subprocess.run([args.viewer_python, str(Path(__file__).with_name("viewer_check.py")),
                                    str(target), json.dumps(result["roi_zyx"]), str(int(mask.sum()))],
                                   capture_output=True, text=True, timeout=180)
            result["viewer"] = dict(seconds=time.perf_counter() - started, passed=check.returncode == 0,
                                    output=(check.stdout + check.stderr)[-4000:])
        except (OSError, subprocess.SubprocessError) as error:
            result["viewer"] = dict(passed=False, output=str(error))
        if not result["viewer"]["passed"]:
            result.update(verified=False, status="viewer check failed")
    return result


def run(args):
    raw, image, layouts = configurations(args)
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(profile=args.profile, host=socket.gethostname(), mount=mount_info(directory),
                    cache_state="uncontrolled", raw=raw, image=image, layouts=layouts, case_order=args.case_order,
                    gpfs_counter_hook=bool(args.gpfs_counter_command),
                    accuracy_gate=dict(min_dice=args.min_dice, max_volume_error=args.max_volume_error,
                                       min_truth_coverage=.98),
                    versions=dict(numpy=np.__version__, h5py=h5py.__version__, hdf5=h5py.version.hdf5_version,
                                  zarr=zarr.__version__, skimage=skimage.__version__),
                    case_results=[], complete=False)
    manifest = directory / "experiment.json"
    save_json(manifest, metadata)
    timeline = Timeline(directory, args.gpfs_counter_command)
    if args.scratch:
        args.scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="voxel-replay-", dir=args.scratch) as temporary:
        scratch = Path(temporary)
        metadata["scratch_mount"] = mount_info(scratch)
        with timeline.phase("shared", "simulate", counters=False):
            identity = dict(
                parameters={key: raw[key] for key in ("size", "tile", "tile_origins_yx_px", "angles_deg",
                                                       "seed", "poisson", "photons", "dark", "mu_per_voxel")},
                versions=[np.__version__, skimage.__version__, version("scipy")],
                generator=getsource(tiled.make_acquisition),
                phantom=Path(__file__).with_name("phantom.py").read_text(),
            )
            cache = None if args.no_source_cache else args.source_cache
            staged, status = prepare_source(scratch, cache, "beamline", identity,
                                            lambda path: tiled.make_acquisition(path, raw, retain_reference=False))
            metadata["source_cache"] = status
            if cache is not None:
                metadata["source_cache_key"] = staged.name
            metadata["source_mount"] = mount_info(staged)
            raw_source = staged / "raw-source"
            sources = [np.load(raw_source / f"tile_{i}.npy", mmap_mode="r")
                       for i in range(len(raw["tile_origins_yx_px"]))]
        for case in args.case_order:
            _, layout = CASES[case]
            settings = layouts[layout]
            with TemporaryDirectory(prefix=f"case-{case}-", dir=scratch) as staged_case:
                result = run_case(case, directory / case, Path(staged_case), raw_source, sources,
                                  dict(raw, raw_chunks=settings["raw_chunks"]),
                                  dict(image, chunks=settings["chunks"]), args, timeline)
            metadata["case_results"].append(result)
            save_json(manifest, metadata)
    results = metadata["case_results"]
    comparable = [r for r in results if "reconstruction_sha256" in r]
    common = {r["reconstruction_sha256"] for r in comparable}
    rois = {(str(r["roi_zyx"]), r["mask_sha256"]) for r in comparable}
    metadata["identical_reconstruction_and_roi"] = (
        len(common) == len(rois) == 1 if len(comparable) == len(CASES) else None)
    metadata["verified"] = metadata["identical_reconstruction_and_roi"] and all(r["verified"] for r in results)
    metadata["verified"] = bool(metadata["verified"])
    make_report(directory, metadata, timeline.rows)
    metadata["complete"] = True
    save_json(manifest, metadata)
    print(f"Report: {directory / 'report.md'}", flush=True)
    return 0 if metadata["verified"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, nargs="?", help="new output directory")
    p.add_argument("--profile", choices=("local", "gpfs"), default="local")
    p.add_argument("--scratch", type=Path, help="staging filesystem for detector replay and reconstruction")
    add_cache_options(p)
    p.add_argument("--size", type=int)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--angles", type=int)
    p.add_argument("--fps", type=float, default=500., help="individual tile images per second")
    p.add_argument("--queue-frames", type=int, default=16)
    p.add_argument("--overflow", choices=("block", "drop"), default="block")
    p.add_argument("--workers", type=int, default=2, help="default reader and writer worker count")
    p.add_argument("--read-workers", type=int, help="override application detector readers")
    p.add_argument("--write-workers", type=int, help="override raw and image writers")
    p.add_argument("--zarr-concurrency", type=int, default=4, help="Zarr async.concurrency setting for chunk scheduling")
    p.add_argument("--angle-block", type=int, default=32, help="angles per raw file or shard")
    p.add_argument("--frame-chunk", type=int, default=4, help="angles per acquisition-oriented raw chunk")
    p.add_argument("--row-band", type=int, default=16)
    p.add_argument("--detector-bricks", choices=("spatial", "sinogram"), default="spatial",
                   help="raw detector chunks for cases B and D at the same chunk and shard byte budget")
    p.add_argument("--chunk-mib", type=float, help="full-resolution uncompressed image chunk size")
    p.add_argument("--partition-factor", type=int, default=1, help="multiply common image partition depth")
    p.add_argument("--compression", choices=("none", "gzip"), default="gzip")
    p.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--poisson", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--case-order", default="ABCD")
    p.add_argument("--min-dice", type=float, default=.80)
    p.add_argument("--max-volume-error", type=float, default=.25)
    p.add_argument("--gpfs-counter-command", default=os.environ.get("VOXEL_GPFS_COUNTER_COMMAND"),
                   help="JSON counter snapshot command, invoked without a shell")
    p.add_argument("--viewer-python", help="Python interpreter with napari-ome-zarr for a headless reader check")
    args = p.parse_args()
    if args.read_workers is None:
        args.read_workers = args.workers
    if args.write_workers is None:
        args.write_workers = args.workers
    gpfs = args.profile == "gpfs"
    if args.directory is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        args.directory = Path("data") / f"experiment-{args.profile}-{stamp}"
    if args.size is None:
        args.size = 256 if gpfs else 64
    if args.angles is None:
        args.angles = 256 if gpfs else 128
    if args.chunk_mib is None:
        args.chunk_mib = 8. if gpfs else .125
    if args.seed < 0 or not 0 <= args.min_dice <= 1 or not 0 <= args.max_volume_error <= 1:
        p.error("seed must be nonnegative and accuracy thresholds must be between 0 and 1")
    try:
        with zarr.config.set({"async.concurrency": args.zarr_concurrency}):
            return run(args)
    except (ValueError, FileExistsError) as error:
        p.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
