"""Simulate a parallel-beam detector mosaic, stitch stored tiles and reconstruct.

Known integer tile offsets; no registration estimation, phase contrast or GPU.
Raw axes are [angle, detector_y, detector_x]. Reconstructed axes are [z,y,x].
"""
import argparse
from contextlib import ExitStack, contextmanager
import csv
import itertools
import json
from math import gcd
from multiprocessing.util import Finalize
import os
from pathlib import Path
from threading import get_native_id, local
import time

import h5py
import numpy as np
from skimage.transform import iradon, radon
import zarr

import demo
from io_workers import ordered_batches, pool_for, start_workers
from phantom import make_pyramid, make_sources


def tile_starts(size, width, overlap):
    """Cover a detector axis; an edge-snapped final tile may overlap more."""
    if not 0 < width <= size or not 0 <= overlap < width:
        raise ValueError("require 0 <= overlap < tile <= size")
    starts = list(range(0, size - width + 1, width - overlap))
    if starts[-1] != size - width:
        starts.append(size - width)
    return starts


def feather(origin, width, size, overlap):
    """Positive linear overlap weights, without tapering exterior boundaries."""
    w = np.ones((width, width), dtype=np.float32)
    if overlap:
        ramp = np.arange(1, overlap + 1, dtype=np.float32) / (overlap + 1)
        for axis, start in enumerate(origin):
            a = np.ones(width, dtype=np.float32)
            if start > 0:
                a[:overlap] *= ramp
            if start + width < size:
                a[-overlap:] *= ramp[::-1]
            w *= a[:, None] if axis == 0 else a[None, :]
    return w


def make_acquisition(directory, config, *, retain_reference=True):
    n, width, angles = config["size"], config["tile"], np.asarray(config["angles_deg"])
    make_sources(directory / "truth", n, 1, 8, 0., config["seed"])
    truth = np.load(directory / "truth/0.npy", mmap_mode="r")
    ideal = np.lib.format.open_memmap(directory / "ideal-projections.npy", mode="w+",
                                    shape=(len(angles), n, n), dtype="<f4")
    for z in range(n):
        # A parallel-beam 3D volume is a stack of independent 2D Radon problems.
        ideal[:, z, :] = radon(truth[z].astype(np.float32) / 65535,
                               theta=angles, circle=True, preserve_range=True).T
    ideal.flush()
    source = directory / "raw-source"
    source.mkdir()
    for tile, (y, x) in enumerate(config["tile_origins_yx_px"]):
        raw = np.lib.format.open_memmap(source / f"tile_{tile}.npy", mode="w+",
                                       shape=(len(angles), width, width), dtype="<u2")
        flux = config["photons"] * (1 + .03 * tile)
        flat = np.full((width, width), flux + config["dark"], dtype="<f4")
        dark = np.full((width, width), config["dark"], dtype="<f4")
        np.save(source / f"flat_{tile}.npy", flat)
        np.save(source / f"dark_{tile}.npy", dark)
        for ai in range(len(angles)):
            expected = flux * np.exp(-config["mu_per_voxel"] * ideal[ai, y:y+width, x:x+width]) + config["dark"]
            if config["poisson"]:
                rng = np.random.default_rng(np.random.SeedSequence([config["seed"], tile, ai]))
                expected = rng.poisson(expected)
            raw[ai] = np.rint(np.clip(expected, 0, 65535)).astype("<u2")
        raw.flush()
    if not retain_reference:
        del ideal, truth
        (directory / "ideal-projections.npy").unlink()
        (directory / "truth/0.npy").unlink()
        (directory / "truth").rmdir()


def write_raw(directory, config, kind):
    """Identical angle blocks become HDF5 part files or Zarr shards."""
    started = time.perf_counter()
    count, width, block = len(config["angles_deg"]), config["tile"], config["angle_block"]
    chunks = (1, width, width) if config["layout"] == "frames" else (block, np.gcd(width, 8).item(), width)
    if kind == "zarr":
        root = zarr.open_group(str(directory / "raw.zarr"), mode="w", zarr_format=3)
        root.attrs.update(config)
        root.attrs["schema"] = "voxel-io-demo raw projection mosaic; NOT OME spatial-image metadata"
    else:
        (directory / "raw-nexus").mkdir()
    for tile, origin in enumerate(config["tile_origins_yx_px"]):
        raw = np.load(directory / "raw-source" / f"tile_{tile}.npy", mmap_mode="r")
        flat = np.load(directory / "raw-source" / f"flat_{tile}.npy")
        dark = np.load(directory / "raw-source" / f"dark_{tile}.npy")
        if kind == "zarr":
            group = root.create_group(f"tile_{tile}")
            group.attrs["origin_yx_px"] = origin
            array = group.create_array("data", shape=raw.shape, dtype="<u2", chunks=chunks,
                                       shards=(block, width, width), compressors=None,
                                       dimension_names=("angle", "detector_y", "detector_x"),
                                       config={"write_empty_chunks": True})
            array.attrs["units"] = "counts"
            group.create_array("flat", data=flat, compressors=None)
            group.create_array("dark", data=dark, compressors=None)
        else:
            (directory / "raw-nexus" / f"tile_{tile}").mkdir()
        for start in range(0, count, block):
            selection = (slice(start, start + block), slice(0, width), slice(0, width))
            if kind == "zarr":
                array[selection] = raw[selection]
            else:
                part = directory / "raw-nexus" / f"tile_{tile}" / f"part_{start}.nxs"
                with h5py.File(part, "w") as f:
                    f.attrs["default"] = "entry"
                    entry = demo.nxgroup(f, "entry", "NXentry")
                    entry.attrs["default"] = "data"
                    data = demo.nxgroup(entry, "data", "NXdata")
                    data.attrs["signal"] = "data"
                    data.create_dataset("data", data=raw[selection], chunks=chunks).attrs["units"] = "counts"
    if kind == "nexus":
        create_raw_master(directory, config, directory / "raw-source")
    return time.perf_counter() - started


def create_raw_master(directory, config, calibration_source):
    count, width, block = len(config["angles_deg"]), config["tile"], config["angle_block"]
    with h5py.File(directory / "raw-nexus" / "raw.nxs", "w", libver="latest") as f:
        f.attrs["default"] = "entry"
        entry = demo.nxgroup(f, "entry", "NXentry")
        entry.attrs["default"] = "tiles/tile_0"
        entry.create_dataset("title", data="Simulated tiled parallel-beam acquisition")
        entry.create_dataset("simulation_parameters", data=json.dumps(config))
        tiles = demo.nxgroup(entry, "tiles", "NXcollection")
        for tile, origin in enumerate(config["tile_origins_yx_px"]):
            group = demo.nxgroup(tiles, f"tile_{tile}", "NXdata")
            layout = h5py.VirtualLayout(shape=(count, width, width), dtype="<u2")
            for start in range(0, count, block):
                source = h5py.VirtualSource(f"tile_{tile}/part_{start}.nxs", "/entry/data/data",
                                            shape=(block, width, width))
                layout[start:start+block] = source
            group.create_virtual_dataset("data", layout).attrs["units"] = "counts"
            group.attrs["signal"] = "data"
            group.attrs["axes"] = ["angle", "detector_y", "detector_x"]
            for axis, values, units in (
                    ("angle", config["angles_deg"], "degrees"),
                    ("detector_y", np.arange(width) + origin[0] + .5, "um"),
                    ("detector_x", np.arange(width) + origin[1] - config["size"] // 2, "um")):
                group.create_dataset(axis, data=values).attrs["units"] = units
            group.attrs["origin_yx_px"] = origin
            group.create_dataset("flat", data=np.load(calibration_source / f"flat_{tile}.npy"))
            group.create_dataset("dark", data=np.load(calibration_source / f"dark_{tile}.npy"))


@contextmanager
def raw_reader(directory, config, kind, trace, *, load_calibration=True):
    if kind == "nexus":
        handle = h5py.File(directory / "raw-nexus/raw.nxs", "r", rdcc_nbytes=0)
        groups = [handle[f"entry/tiles/tile_{i}"] for i in range(len(config["tile_origins_yx_px"]))]
    else:
        handle = None
        root = zarr.open_group(str(directory / "raw.zarr"), mode="r")
        groups = [root[f"tile_{i}"] for i in range(len(config["tile_origins_yx_px"]))]
    try:
        calibration = ([(np.asarray(g["flat"][:]), np.asarray(g["dark"][:])) for g in groups]
                       if load_calibration else None)
        pid, thread_id = os.getpid(), get_native_id()

        def read(tile, selection, pattern):
            started = time.perf_counter_ns()
            array = groups[tile]["data"]
            lookup_finished = time.perf_counter_ns()
            values = np.asarray(array[selection])
            finished = time.perf_counter_ns()
            trace.append(dict(format=kind, tile=tile, pattern=pattern, elapsed_ms=(finished - started) / 1e6,
                              started_ns=started, finished_ns=finished, worker_pid=pid, worker_thread=thread_id,
                              array_lookup_ms=(lookup_finished - started) / 1e6,
                              returned_bytes=values.nbytes,
                              selection=json.dumps([[s.start, s.stop] for s in selection]),
                              cache_state="uncontrolled"))
            if "raw_chunks" in config:
                chunks = config["raw_chunks"]
                trace[-1]["full_chunk_model_bytes"] = demo.touched(selection, chunks) * int(np.prod(chunks)) * 2
                trace[-1]["parts_intersected"] = demo.touched(selection, (config["angle_block"], config["tile"], config["tile"]))
            return values
        yield read, calibration
    finally:
        if handle is not None:
            handle.close()


def sinogram_chunks(width, angle_block, frame_chunk):
    """Keep acquisition chunk bytes while spanning more angles and fewer rows."""
    factor = gcd(width, angle_block // frame_chunk)
    return (frame_chunk * factor, width // factor, width)


def band_requests(config, first, stop):
    width, count = config["tile"], len(config["angles_deg"])
    for tile, (y, _) in enumerate(config["tile_origins_yx_px"]):
        lo, hi = max(first, y), min(stop, y + width)
        if hi > lo:
            yield tile, (slice(0, count), slice(lo - y, hi - y), slice(0, width)), "sinogram_band"


def frame_requests(config):
    width = config["tile"]
    return [(tile, (slice(0, 1), slice(0, width), slice(0, width)), "detector_frame")
            for tile in range(len(config["tile_origins_yx_px"]))]


def request_batches(config):
    yield frame_requests(config)
    for first in range(0, config["size"], config["row_band"]):
        stop = min(first + config["row_band"], config["size"])
        yield list(band_requests(config, first, stop))


_raw_worker = local()


def init_raw_worker(directory, config, kind):
    trace = []
    reader = raw_reader(directory, config, kind, trace, load_calibration=False)
    _raw_worker.read, _ = reader.__enter__()
    _raw_worker.reader = reader
    _raw_worker.trace = trace
    if kind == "nexus":
        Finalize(None, reader.__exit__, (None, None, None), exitpriority=1)


def worker_read_batch(requests):
    trace = _raw_worker.trace
    trace.clear()
    arrays = [_raw_worker.read(*request) for request in requests]
    return arrays, trace.copy()


@contextmanager
def parallel_raw_batches(directory, config, kind, batches, workers):
    """Read bounded, ordered batches through independent handles."""
    with pool_for(kind, workers, initializer=init_raw_worker, initargs=(directory, config, kind)) as pool:
        start_workers(pool, workers)
        yield ordered_batches(pool, worker_read_batch, batches, workers)


def prefetched_frames(batches, trace):
    for arrays, rows in batches:
        trace.extend(rows)
        yield from arrays


def stitched_band(read, calibration, config, first, stop):
    """Read only intersecting tile rows, blend transmission, then take -log."""
    n, width = config["size"], config["tile"]
    count = len(config["angles_deg"])
    mosaic = np.zeros((count, stop-first, n), dtype=np.float32)
    weights = np.zeros((stop-first, n), dtype=np.float32)
    for tile, selection, pattern in band_requests(config, first, stop):
        origin = config["tile_origins_yx_px"][tile]
        y, x = origin
        local_y = selection[1]
        lo, hi = local_y.start + y, local_y.stop + y
        values = read(tile, selection, pattern).astype(np.float32)
        flat, dark = calibration[tile]
        transmission = (values - dark[local_y]) / (flat[local_y] - dark[local_y])
        w = feather(origin, width, n, config["overlap"])[local_y]
        mosaic[:, lo-first:hi-first, x:x+width] += transmission * w
        weights[lo-first:hi-first, x:x+width] += w
    if np.any(weights <= 0):
        raise ValueError("tile mosaic leaves uncovered detector pixels")
    transmission = mosaic / weights
    return -np.log(np.maximum(transmission, 1e-6)) / config["mu_per_voxel"]


def reconstruct(directory, config, kind, trace, *, source=None, output=None, read_metrics=None):
    n, count = config["size"], len(config["angles_deg"])
    source = source or directory / "raw-source"
    output = output or directory / f"reconstruction-{kind}.npy"
    result = np.lib.format.open_memmap(output, mode="w+",
                                     shape=(n,n,n), dtype="<f4")
    started = time.perf_counter()
    workers = min(config.get("read_workers", 1), 1 + (n + config["row_band"] - 1) // config["row_band"])
    read_wait = 0.
    with ExitStack() as stack:
        read, calibration = stack.enter_context(raw_reader(directory, config, kind, trace))
        if workers > 1:
            batches = stack.enter_context(parallel_raw_batches(directory, config, kind, request_batches(config), workers))
            frames = prefetched_frames(batches, trace)

            def read(*_):
                nonlocal read_wait
                waiting = time.perf_counter()
                frame = next(frames)
                read_wait += time.perf_counter() - waiting
                return frame

        setup = time.perf_counter() - started
        for tile, selection, pattern in frame_requests(config):
            frame = read(tile, selection, pattern)
            reference = np.load(source / f"tile_{tile}.npy", mmap_mode="r")
            np.testing.assert_array_equal(frame, reference[:1])
        for first in range(0, n, config["row_band"]):
            stop = min(n, first+config["row_band"])
            sinograms = stitched_band(read, calibration, config, first, stop)
            for zi in range(stop-first):
                result[first+zi] = iradon(sinograms[:,zi,:].T, theta=config["angles_deg"],
                                          output_size=n, circle=True, filter_name="ramp", preserve_range=True)
    result.flush()
    if read_metrics is not None:
        read_calls = sum(row["elapsed_ms"] for row in trace) / 1000
        read_metrics.update(workers=workers, setup_seconds=setup, read_calls_seconds_sum=read_calls,
                            foreground_wait_seconds=read_wait if workers > 1 else read_calls)
    return time.perf_counter() - started


def export_volume(directory, config):
    """The reconstructed volume gets valid spatial OME metadata, unlike raw angles."""
    target = directory / "volume"
    target.mkdir()
    (target / "source").mkdir()
    (target / "nexus").mkdir()
    n = config["size"]
    source = np.load(directory / "reconstruction-nexus.npy", mmap_mode="r")
    quantized = np.lib.format.open_memmap(target / "source/0.npy", mode="w+", shape=source.shape, dtype="<u2")
    for z in range(n):
        quantized[z] = np.rint(np.clip(source[z], 0, 1) * 65535).astype("<u2")
    quantized.flush()
    del quantized
    make_pyramid(target / "source", config["levels"])
    settings = dict(size=n, levels=config["levels"], chunks=[min(16,n),min(32,n),min(32,n)],
                    shards=[min(64,n)]*3, voxel_um=1., compression="none", workers=1,
                    image_name="Reconstructed tiled phantom", source_pipeline="../manifest.json",
                    process_name="reconstruction",
                    reconstruction="filtered backprojection from stored overlapping detector tiles")
    manifest = dict(settings, complete=False, axes=demo.AXES, dtype="uint16", gpfs_block_mib=8)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    timings = demo.write_formats(target, settings)
    manifest.update(complete=True, timings=timings)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))


def overview(directory, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    n, width = config["size"], config["tile"]
    truth = np.load(directory / "truth/0.npy", mmap_mode="r")[n//2].astype(np.float32)/65535
    ideal = np.load(directory / "ideal-projections.npy", mmap_mode="r")
    recon = np.load(directory / "reconstruction-nexus.npy", mmap_mode="r")[n//2]
    trace = []
    with raw_reader(directory, config, "zarr", trace) as (read, calibration):
        stitched = stitched_band(read, calibration, config, n//2, n//2+1)[:,0,:].T
    fig, axes = plt.subplots(2,3,figsize=(12,8),layout="constrained")
    axes[0,0].imshow(truth,cmap="gray",vmin=0,vmax=1)
    axes[0,0].set_title("Ground-truth XY slice")
    axes[0,1].imshow(ideal[0],cmap="gray")
    for tile, (y,x) in enumerate(config["tile_origins_yx_px"]):
        axes[0,1].add_patch(Rectangle((x-.5,y-.5),width,width,fill=False,edgecolor="tab:orange"))
        axes[0,1].text(x+3,y+12,str(tile),color="tab:orange")
    axes[0,1].set_title("Full projection: overlapping detector fields")
    tile = np.load(directory / "raw-source/tile_0.npy",mmap_mode="r")[0]
    axes[0,2].imshow(tile,cmap="gray")
    axes[0,2].set_title("Simulated raw tile 0: transmitted counts")
    axes[1,0].imshow(stitched,cmap="gray",aspect="auto",extent=(0,180,n,0))
    axes[1,0].set_title("Stitched sinogram at central z")
    axes[1,0].set_xlabel("Angle (degrees)")
    axes[1,0].set_ylabel("Detector coordinate (pixels)")
    axes[1,1].imshow(recon,cmap="gray",vmin=0,vmax=1)
    axes[1,1].set_title("Reconstructed from stored tiles")
    error = np.abs(recon-truth)
    image = axes[1,2].imshow(error,cmap="magma",vmin=0,vmax=max(.01,float(error.max())))
    axes[1,2].set_title("Absolute error against ground truth")
    fig.colorbar(image,ax=axes[1,2],shrink=.7)
    for ax in axes.flat:
        if ax is not axes[1,0]:
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Pixels")
    fig.suptitle(f"Tiled tomography · {n}³ phantom · {len(config['angles_deg'])} angles · {len(config['tile_origins_yx_px'])} tiles")
    fig.savefig(directory / "tiled-overview.png",dpi=150)
    plt.close(fig)


def run(args):
    n = args.size
    if n < 32 or n & (n-1):
        raise ValueError("size must be a power of two, at least 32")
    if args.angles < 2 or args.angle_block < 1 or args.angles % args.angle_block:
        raise ValueError("angles must be a positive multiple of angle-block, at least 2")
    if args.row_band < 1 or not 1 <= args.levels <= n.bit_length() or args.seed < 0:
        raise ValueError("invalid row-band, levels or seed")
    starts = tile_starts(n,args.tile,args.overlap)
    config = dict(size=n,tile=args.tile,overlap=args.overlap,
                  tile_origins_yx_px=list(itertools.product(starts,starts)),
                  angles_deg=np.linspace(0,180,args.angles,endpoint=False).tolist(),
                  angle_block=args.angle_block,row_band=args.row_band,levels=args.levels,
                  photons=20000.,dark=100.,mu_per_voxel=2./n,seed=args.seed,
                  poisson=args.poisson,layout=args.layout,known_integer_positions=True,
                  registration_estimated=False,raw_axes=["angle","detector_y","detector_x"],
                  volume_axes=["z","y","x"],cache_state="uncontrolled")
    directory = args.directory.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    manifest = dict(config,complete=False)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest,indent=2))
    timings = {}
    started = time.perf_counter()
    make_acquisition(directory,config)
    timings["simulate_seconds"] = time.perf_counter()-started
    print(f"Simulated {len(config['tile_origins_yx_px'])} tiles, {args.angles} angles, {args.tile}² pixels",flush=True)
    trace=[]
    for kind in ("nexus","zarr"):
        timings[f"{kind}_raw_write_seconds"] = write_raw(directory,config,kind)
        timings[f"{kind}_reconstruct_seconds"] = reconstruct(directory,config,kind,trace)
        print(f"Reconstructed from {kind} tiles",flush=True)
    truth = np.load(directory / "truth/0.npy",mmap_mode="r")
    a = np.load(directory / "reconstruction-nexus.npy",mmap_mode="r")
    b = np.load(directory / "reconstruction-zarr.npy",mmap_mode="r")
    squared_error = squared_truth = max_difference = 0.
    for z in range(n):
        expected = truth[z].astype(np.float32)/65535
        squared_error += float(np.sum((a[z]-expected)**2,dtype=np.float64))
        squared_truth += float(np.sum(expected**2,dtype=np.float64))
        max_difference = max(max_difference,float(np.max(np.abs(a[z]-b[z]))))
    if max_difference > 1e-6:
        raise ValueError("reconstructions from the two raw formats disagree")
    metrics = dict(nrmse=np.sqrt(squared_error/squared_truth),
                   max_abs_reconstruction_difference_between_formats=max_difference,
                   detector_pixels_recorded_per_angle=len(starts)**2*args.tile**2,
                   detector_pixels_in_panorama=n*n)
    ideal = np.load(directory / "ideal-projections.npy",mmap_mode="r")
    reference = iradon(ideal[:,n//2,:].T,theta=config["angles_deg"],output_size=n,
                       circle=True,filter_name="ramp",preserve_range=True)
    metrics["central_slice_max_abs_error_vs_full_detector_fbp"] = float(np.max(np.abs(a[n//2]-reference)))
    with (directory / "read-trace.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    export_volume(directory,config)
    overview(directory,config)
    manifest.update(complete=True,timings=timings,metrics=metrics)
    path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(metrics,indent=2))
    print(f"Ready: {directory}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path,nargs="?",default=Path("data/tiled"))
    p.add_argument("--size",type=int,default=128)
    p.add_argument("--tile",type=int,default=72)
    p.add_argument("--overlap",type=int,default=16)
    p.add_argument("--angles",type=int,default=256)
    p.add_argument("--angle-block",type=int,default=32)
    p.add_argument("--row-band",type=int,default=16)
    p.add_argument("--levels",type=int,default=3)
    p.add_argument("--layout",choices=("frames","rows"),default="frames")
    p.add_argument("--poisson",action="store_true",help="independent photon noise for each tile and angle")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    try:
        run(args)
    except (ValueError,FileExistsError) as exc:
        p.exit(2,f"{exc}\n")


if __name__=="__main__":
    main()
