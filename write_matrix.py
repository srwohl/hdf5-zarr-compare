"""Measure acquisition writes against the same stored or synthetic detector data."""
import argparse
import itertools
import json
import os
from pathlib import Path
from shutil import disk_usage
from statistics import median
from tempfile import TemporaryDirectory
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from experiment import CASES, Timeline, footprint, mount_info, save_csv, save_json, verify_raw
from profile_raw_reads import fresh_reader_pool as fresh_trial_pool
from replay import RawWriter, gb_per_second, run_replay
from tiled import raw_reader


def integer_list(value):
    try:
        values = tuple(int(item) for item in value.split(","))
        if min(values) < 1 or len(set(values)) != len(values):
            raise ValueError
        return values
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated distinct positive integers")


def stage_frames(directory, scratch, raw):
    """Keep a reusable replay source off the destination filesystem when possible."""
    with raw_reader(directory / "A", raw, "nexus", []) as (read, calibration):
        width, angles = raw["tile"], len(raw["angles_deg"])
        for tile, (flat, dark) in enumerate(calibration):
            np.save(scratch / f"flat_{tile}.npy", flat)
            np.save(scratch / f"dark_{tile}.npy", dark)
            pixels = np.lib.format.open_memmap(scratch / f"tile_{tile}.npy", mode="w+",
                                               dtype="<u2", shape=(angles, width, width))
            for first in range(0, angles, raw["angle_block"]):
                stop = min(first + raw["angle_block"], angles)
                selection = (slice(first, stop), slice(0, width), slice(0, width))
                pixels[selection] = read(tile, selection, "matrix_source")
            pixels.flush()


def synthetic_config(width, frames):
    block = 4
    if width < 2 or width % 2:
        raise ValueError("detector size must be a positive even pixel count")
    if frames < block or frames % block:
        raise ValueError(f"frame count must be a positive multiple of {block}")
    raw = dict(size=width, tile=width, tile_origins_yx_px=[(0, 0)],
               angles_deg=np.linspace(0, 180, frames, endpoint=False).tolist(),
               angle_block=block, compression="gzip", sync=True, queue_frames=16,
               zarr_concurrency=4, seed=42, photons=20000, dark=100)
    chunks = {case: (1, width, width) if layout == "acquisition" else (block, width // 2, width // 2)
              for case, (_, layout) in CASES.items()}
    return raw, chunks


def stage_synthetic(scratch, raw):
    width, count = raw["tile"], len(raw["angles_deg"])
    required = (count * 2 + 8) * width**2 + 4096
    free = disk_usage(scratch).free
    if required > free:
        raise ValueError(f"detector source needs {required / (1 << 30):.2f} GiB on scratch, "
                         f"with {free / (1 << 30):.2f} GiB available at {scratch.parent}")
    flat = np.full((width, width), raw["photons"] + raw["dark"], dtype="<f4")
    dark = np.full((width, width), raw["dark"], dtype="<f4")
    np.save(scratch / "flat_0.npy", flat)
    np.save(scratch / "dark_0.npy", dark)
    axis = np.linspace(-1, 1, width, dtype=np.float32)
    expected = flat - .6 * raw["photons"] * np.exp(-6 * (axis[:, None]**2 + axis**2))
    pixels = np.lib.format.open_memmap(scratch / "tile_0.npy", mode="w+", dtype="<u2",
                                       shape=(count, width, width))
    rng = np.random.default_rng(raw["seed"])
    for index in range(count):
        pixels[index] = rng.poisson(expected)
    pixels.flush()


def trial_throughput(stats, phase):
    useful = stats["published_bytes"]
    client_bytes = phase["counter_delta"].get("client_bytes_written")
    return dict(setup_to_last_gb_s=gb_per_second(useful, stats["setup_to_last_publish_seconds"]),
                full_run_gb_s=gb_per_second(useful, stats["acquisition_wall_seconds"]),
                gpfs_client_write_gb_s=(gb_per_second(client_bytes, phase["seconds"])
                                        if client_bytes is not None else None))


def trial(directory, scratch, case, config, label, counter_command):
    sources = [np.load(path, mmap_mode="r") for path in sorted(scratch.glob("tile_*.npy"),
                                                               key=lambda path: int(path.stem.split("_")[1]))]
    kind = CASES[case][0]
    timing = Timeline(directory, counter_command)
    with zarr.config.set({"async.concurrency": config.get("zarr_concurrency", zarr.config.get("async.concurrency"))}):
        with timing.phase(label, "acquisition"):
            started = time.perf_counter()
            writer = RawWriter(directory, config, kind, scratch)
            metadata_seconds = time.perf_counter() - started
            stats, partitions, _, origin = run_replay(writer, sources, config)
            end = time.perf_counter()
    first = min(row["published_seconds"] for row in partitions)
    stats.update(metadata_setup_seconds=metadata_seconds, setup_to_first_publish_seconds=origin + first - started,
                 setup_to_last_publish_seconds=origin + stats["replay_seconds"] - started,
                 acquisition_wall_seconds=end - started)
    phase = timing.rows[0]
    stats.update(trial_throughput(stats, phase))
    storage = footprint(writer.root_path)
    stats.update(stored_files=storage["files"], raw_logical_bytes=storage["logical_bytes"],
                 raw_allocated_bytes=storage["allocated_bytes"])
    if stats["missing_frames"] == 0:
        verify_raw(directory, config, kind, sources, scratch)
    stats["verified"] = stats["missing_frames"] == 0
    stats.pop("occupancy")  # The CSV retains peak depth and queue-full events.
    return stats, partitions, phase


def summarize(samples, combinations):
    summary = []
    keys = ("sustained_received_fps", "sustained_published_fps", "blocked_seconds", "publish_p95_ms",
            "setup_to_last_publish_seconds", "setup_to_first_publish_seconds", "acquisition_wall_seconds",
            "worker_setup_seconds", "encode_seconds", "write_seconds", "sync_seconds", "raw_logical_bytes",
            "stored_files", "received_gb_s", "published_gb_s", "payload_gb_s", "setup_to_last_gb_s",
            "full_run_gb_s", "gpfs_client_write_gb_s")
    for rate, writers, case in combinations:
        group = [row for row in samples if (row["target_fps"], row["writers"], row["case"]) == (rate, writers, case)]
        entry = {key: median(row[key] for row in group if row[key] is not None)
                 for key in keys if any(row[key] is not None for row in group)}
        summary.append(dict(case=case, format=group[0]["format"], layout=group[0]["layout"], target_fps=rate,
                            writers=writers, missing_frames=sum(row["missing_frames"] for row in group),
                            **entry))
    return summary


def plot_matrix(path, summary, groups):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    colors = ("#4578a8", "#82a9ce", "#257e6c", "#76b7a5")
    series = (("published_gb_s", "Useful detector GB/s", "During replay"),
              ("setup_to_last_publish_seconds", "Seconds", "Writer setup to last publication"))
    xs = np.arange(len(groups))
    for ax, (key, ylabel, title) in zip(axes, series):
        for index, (case, color) in enumerate(zip(CASES, colors)):
            rows = [row for row in summary if row["case"] == case]
            fmt = "HDF5" if rows[0]["format"] == "nexus" else "Zarr"
            layout = "frames" if rows[0]["layout"] == "acquisition" else rows[0]["layout"]
            ax.bar(xs + (index - 1.5) * .18, [row[key] for row in rows], width=.17,
                   color=color, label=f"{fmt} · {layout}")
        ax.set_xticks(xs, [f"{rate:g} fps\n{writers} {'writer' if writers == 1 else 'writers'}" for rate, writers in groups])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=.24)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(output, setup, summary):
    groups = list(dict.fromkeys((row["target_fps"], row["writers"]) for row in summary))
    plot_matrix(output / "write-matrix.png", summary, groups)
    raw = setup["raw"]
    filesystem = setup["destination_mount"].get("fstype", "unknown")
    scratch = setup["source_mount"].get("fstype", "unknown")
    compression = "gzip compression" if raw["compression"] == "gzip" else "uncompressed output"
    sync = "on" if raw["sync"] else "off"
    quality = "Every trial was reread and matched the source." if setup["verified"] else "At least one trial was incomplete."
    chunk_shapes = " · ".join(f"{case} {' × '.join(map(str, shape))}" for case, shape in setup["raw_chunks"].items())
    width = raw["tile"]
    frames = len(raw["angles_deg"]) * len(raw["tile_origins_yx_px"])
    source = "synthetic counts with Poisson noise" if setup["baseline"] is None else "saved detector replay"
    passes = "1 pass" if setup["repeats"] == 1 else f"{setup['repeats']} passes"
    lines = ["# Detector write matrix", "",
             f"{passes} at each setting, with a fresh process and fresh output per trial. "
             f"Destination was **{filesystem}**, staged source **{scratch}**. "
             f"{compression}, fsync {sync}, {raw['angle_block']} angles per file or shard, "
             f"{raw['queue_frames']} queued frames. Zarr async concurrency was {setup['zarr_concurrency']}. "
             f"All cases shared the same {len(setup['cpu_affinity'])} logical CPUs. {quality}", "",
             f"Raw chunk dimensions in angles × detector y × detector x. {chunk_shapes}.", "",
             "The replay source pauses when the bounded queue fills. Sustained received rate and blocked time "
             "show how closely the writer kept up with the configured source rate. "
             "The setup-to-last clock starts before initializing the writer and ends at the last visible marker.", "",
             f"Each trial replays {frames:,} uint16 frames of {width} × {width} pixels from {source}. "
             f"One frame contains {width**2 * 2:,} uncompressed detector bytes. GB/s uses decimal gigabytes. "
             "Useful GB/s counts the original bytes in published frames. Payload GB/s counts the HDF5 part file "
             "or Zarr shard bytes handed to the filesystem during replay. Setup-inclusive GB/s counts useful "
             "bytes from writer setup through last publication.", "",
             "![Writer scaling and time to last publication](write-matrix.png)", "",
             "| Target fps | Writers | Case | Format / layout | Received fps | Published fps | "
             "Useful GB/s | Payload GB/s | Setup-inclusive GB/s | Blocked s | P95 ms | "
             "Setup to first s | Setup to last s | Missing |",
             "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        fmt = "HDF5" if row["format"] == "nexus" else "Zarr"
        latency = row.get("publish_p95_ms", float("nan"))
        lines.append(f"| {row['target_fps']} | {row['writers']} | {row['case']} | {fmt} / {row['layout']} | "
                     f"{row['sustained_received_fps']:.0f} | {row['sustained_published_fps']:.0f} | "
                     f"{row['published_gb_s']:.3f} | {row['payload_gb_s']:.3f} | {row['setup_to_last_gb_s']:.3f} | "
                     f"{row['blocked_seconds']:.3f} | {latency:.0f} | "
                     f"{row['setup_to_first_publish_seconds']:.3f} | "
                     f"{row['setup_to_last_publish_seconds']:.3f} | {row['missing_frames']} |")
    gpfs = [row for row in summary if "gpfs_client_write_gb_s" in row]
    if gpfs:
        lines += ["", "## GPFS client write counters", "",
                  "Client throughput uses the sampled byte delta over the full acquisition phase. "
                  "NSD counters provide server-side storage traffic.", "",
                  "| Target fps | Writers | Case | Client write GB/s |", "|---:|---:|---|---:|"]
        for row in gpfs:
            lines.append(f"| {row['target_fps']} | {row['writers']} | {row['case']} | {row['gpfs_client_write_gb_s']:.3f} |")
    lines += ["", "Times and rates are medians across trials. P95 is the median of the within-trial frame P95s. "
              "Encode, write and fsync times in the CSV are summed over workers and can overlap. "
              "The full acquisition duration also includes final synchronization and worker teardown.", "",
              "[Trial data](samples.csv) · [Per-partition publications](partitions.csv) · "
              "[UTC timeline and optional counters](timeline.json)", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", help="verified experiment with its original detector data")
    parser.add_argument("output", type=Path, help="new write matrix directory on the filesystem under test")
    parser.add_argument("--detector-size", type=int, metavar="PIXELS", help="synthetic square detector width in pixels")
    parser.add_argument("--frames", type=int, help="synthetic frame count (default: 256, multiple of 4)")
    parser.add_argument("--compression", choices=("gzip", "none"),
                        help="override output compression for all cases")
    parser.add_argument("--rates", type=integer_list, default=(500, 4000), help="comma-separated source rates")
    parser.add_argument("--writers", type=integer_list, default=(1, 2), help="comma-separated writer counts")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--scratch", type=Path, help="temporary source path, ideally on local storage")
    parser.add_argument("--gpfs-counter-command", default=os.environ.get("VOXEL_GPFS_COUNTER_COMMAND"))
    args = parser.parse_args()
    try:
        if args.repeats < 1:
            raise ValueError("repeats must be positive")
        if (args.directory is None) == (args.detector_size is None):
            raise ValueError("provide either a verified experiment directory or --detector-size")
        directory = args.directory.resolve() if args.directory else None
        if directory is None:
            raw, chunks = synthetic_config(args.detector_size, args.frames if args.frames is not None else 256)
        else:
            if args.frames is not None:
                raise ValueError("--frames requires --detector-size")
            experiment = json.loads((directory / "experiment.json").read_text())
            if not experiment["verified"]:
                raise ValueError("a verified four-case experiment is required")
            raw = experiment["raw"]
            chunks = {row["case"]: row["raw_chunks"] for row in experiment["case_results"]}
        if args.compression is not None:
            raw = dict(raw, compression=args.compression)
        if max(args.writers) > len(raw["angles_deg"]) // raw["angle_block"]:
            raise ValueError("writer count exceeds available angle blocks")
        if args.scratch:
            args.scratch.mkdir(parents=True, exist_ok=True)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        combinations = list(itertools.product(args.rates, args.writers, CASES))
        setup = dict(baseline=str(directory) if directory else None, destination_mount=mount_info(output), raw=raw,
                     raw_chunks=chunks,
                     zarr_concurrency=raw.get("zarr_concurrency", zarr.config.get("async.concurrency")),
                     cpu_affinity=sorted(os.sched_getaffinity(0)), repeats=args.repeats)
        samples, partitions, timeline = [], [], []
        with TemporaryDirectory(prefix="voxel-source-", dir=args.scratch) as source:
            scratch = Path(source)
            setup["source_mount"] = mount_info(scratch)
            if directory is None:
                stage_synthetic(scratch, raw)
            else:
                stage_frames(directory, scratch, raw)
            with fresh_trial_pool() as pool:
                for repeat in range(1, args.repeats + 1):
                    order = combinations if repeat % 2 else combinations[::-1]
                    for index, (rate, writers, case) in enumerate(order, 1):
                        label = f"pass{repeat}-{rate}fps-{writers}writers-{case}"
                        settings = dict(raw, fps=rate, workers=writers, overflow="block", raw_chunks=chunks[case])
                        with TemporaryDirectory(prefix=f"{label}-", dir=output) as target:
                            sample, published, phase = pool.submit(
                                trial, Path(target), scratch, case, settings, label, args.gpfs_counter_command).result()
                        kind, layout = CASES[case]
                        if layout == "bricks" and raw.get("detector_bricks") == "sinogram":
                            layout = "sinogram"
                        samples.append(dict(trial=label, repeat=repeat, target_fps=rate, writers=writers, case=case,
                                            format=kind, layout=layout, **sample))
                        partitions.extend(dict(trial=label, **row) for row in published)
                        timeline.append(phase)
                        if index % 4 == 0:
                            print(f"Pass {repeat}/{args.repeats}: {index}/{len(order)} trials", flush=True)
        setup["verified"] = all(row["verified"] for row in samples)
        summary = summarize(samples, combinations)
        save_csv(output / "samples.csv", samples)
        save_csv(output / "partitions.csv", partitions)
        save_json(output / "timeline.json", timeline)
        save_json(output / "summary.json", dict(setup=setup, results=summary))
        report(output, setup, summary)
        print(output / "report.md", flush=True)
        return 0 if setup["verified"] else 1
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
