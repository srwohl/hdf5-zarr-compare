"""Isolate gzip cost and profile cached Zarr detector reads."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from threading import Lock
import time
from unittest.mock import patch

import numpy as np

from demo import MIB
from experiment import CASES, mount_info, save_csv, save_json
from profile_raw_reads import ORDERS, fresh_reader_pool, load_trace, measure_case, read_case, trace_requests
from replay import RawWriter
from tiled import raw_reader, request_batches


def copy_raw(source, target, raw, records, *, compression):
    calibration_path = target / "calibration"
    calibration_path.mkdir()
    config = dict(raw, compression=compression, sync=False)
    with raw_reader(source / "A", raw, "nexus", []) as (read, calibration):
        for tile, (flat, dark) in enumerate(calibration):
            np.save(calibration_path / f"flat_{tile}.npy", flat)
            np.save(calibration_path / f"dark_{tile}.npy", dark)
        writers = []
        for case, record in records.items():
            directory = target / case
            directory.mkdir()
            settings = dict(config, raw_chunks=record["raw_chunks"])
            writers.append(RawWriter(directory, settings, record["format"], calibration_path))
        width, block = raw["tile"], raw["angle_block"]
        for start in range(0, len(raw["angles_deg"]), block):
            selection = (slice(start, start + block), slice(0, width), slice(0, width))
            frames = [read(tile, selection, "copy") for tile in range(len(calibration))]
            for writer in writers:
                writer.publish(start, frames, [])
    return config


def instrument_zarr(directory, record, config, batches):
    # A separate, instrumented diagnostic pass in the locked Zarr version.
    # Neither timed control changes the normal array or codec implementation.
    from zarr.codecs.gzip import GzipCodec
    from zarr.storage import _local

    counters = {key: dict(calls=0, cpu_ns=0, returned_bytes=0) for key in ("gzip_decode", "local_fetch")}
    lock = Lock()

    def measure(function, key):
        def call(*args, **kwargs):
            started = time.thread_time_ns()
            values = function(*args, **kwargs)
            elapsed = time.thread_time_ns() - started
            size = len(values) if values is not None else 0
            with lock:
                counter = counters[key]
                counter["calls"] += 1
                counter["cpu_ns"] += elapsed
                counter["returned_bytes"] += size
            return values
        return call

    with patch.object(GzipCodec, "_decode_sync", measure(GzipCodec._decode_sync, "gzip_decode")):
        with patch.object(_local, "_get", measure(_local._get, "local_fetch")):
            sample, _ = read_case(directory, record, config, batches, 1)

    result = dict(case=record["case"], compression=config["compression"], cpu_seconds=sample["cpu_seconds"])
    for key, info in counters.items():
        result[f"{key}_calls"] = info["calls"]
        result[f"{key}_cpu_seconds"] = info["cpu_ns"] / 1e9
        result[f"{key}_returned_bytes"] = info["returned_bytes"]
    result["other_cpu_seconds"] = result["cpu_seconds"] - sum(info["cpu_ns"] for info in counters.values()) / 1e9
    return result


def write_report(output, mount, cache_mount, cpus, samples, traced):
    storage_reads = [r["process_storage_read_bytes"] for r in samples if r["process_storage_read_bytes"] is not None]
    storage_note = (f"Linux process counters attributed up to {max(storage_reads) / MIB:.2f} MiB to storage reads per pass."
                    if storage_reads else "Linux process storage-read counters were unavailable.")
    lines = ["# Cached detector-read diagnosis", "",
             "One application reader each with native Zarr concurrency. Each pass uses a fresh Python process. "
             "The clock starts before reader setup and ends at final delivery. The verification pass confirms "
             "all four cases return the same bytes with and without gzip and warms the filesystem cache.", "",
             f"Original mount {mount.get('fstype', '?')}. Uncompressed-copy mount {cache_mount.get('fstype', '?')}. "
             f"CPU affinity allowed {cpus} logical CPUs. The cases and compression settings rotate within each layout. "
             f"{storage_note}", "",
             "| Case | Format | Layout | Gzip wall s | Uncompressed wall s | Gzip CPU s | Uncompressed CPU s | "
             "Gzip read-call MiB | Uncompressed read-call MiB |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    medians = {}
    for case in "ABCD":
        groups = [[r for r in samples if r["case"] == case and r["compression"] == mode] for mode in ("gzip", "none")]
        row = [[median(r[key] for r in group) for group in groups]
               for key in ("time_to_last_batch_seconds", "cpu_seconds", "logical_read_bytes")]
        medians[case] = {"wall_seconds": row[0], "cpu_seconds": row[1], "logical_read_bytes": row[2]}
        layout = groups[0][0]["layout"]
        kind = "HDF5" if CASES[case][0] == "nexus" else "Zarr"
        lines.append(f"| {case} | {kind} | {layout} | {row[0][0]:.3f} | {row[0][1]:.3f} | {row[1][0]:.3f} | "
                     f"{row[1][1]:.3f} | {row[2][0] / MIB:.1f} | {row[2][1] / MIB:.1f} |")
    lines += ["", "## Zarr CPU samples", "",
              "Separate instrumented passes measure thread CPU time inside gzip decoding and local file fetches. "
              "Remaining CPU includes scheduling, indexing, checksums, buffer assembly and instrumentation. "
              "CPU seconds can accumulate across threads.", "",
              "| Case | Compression | Gzip chunks decoded | Gzip CPU s | Local fetch CPU s | Other CPU s |",
              "|---|---|---:|---:|---:|---:|"]
    for row in traced:
        lines.append(f"| {row['case']} | {row['compression']} | {row['gzip_decode_calls']:,} | "
                     f"{row['gzip_decode_cpu_seconds']:.3f} | {row['local_fetch_cpu_seconds']:.3f} | "
                     f"{row['other_cpu_seconds']:.3f} |")
    lines += ["", "[Full timing samples](samples.csv) · [Instrumented Zarr passes](zarr-cpu.csv)", ""]
    (output / "report.md").write_text("\n".join(lines))
    save_json(output / "summary.json", dict(medians=medians, zarr_cpu=traced))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="verified experiment written with gzip")
    parser.add_argument("output", type=Path, help="new diagnostic directory")
    args = parser.parse_args()
    try:
        directory = args.directory.resolve()
        meta = json.loads((directory / "experiment.json").read_text())
        raw = meta["raw"]
        if not meta["verified"] or raw["compression"] != "gzip":
            raise ValueError("expected a verified experiment written with gzip")
        records = {record["case"]: record for record in meta["case_results"]}
        batches = list(request_batches(raw))
        requests = [request for batch in batches for request in batch]
        if any(trace_requests(load_trace(directory / case / "raw-reads.csv")) != requests for case in records):
            raise ValueError("cases must contain the same recorded detector read plan")

        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        with TemporaryDirectory(prefix="uncompressed-", dir=output) as temporary:
            temp = Path(temporary)
            print("Copying the stored detector data with compression disabled", flush=True)
            configs = {"gzip": raw, "none": copy_raw(directory, temp, raw, records, compression="none")}
            sources = {"gzip": directory, "none": temp}
            digests = {read_case(sources[mode], records[case], config, batches, 1, verify=True)[0]
                       for mode, config in configs.items() for case in records}
            if len(digests) != 1:
                raise ValueError("detector selection bytes differ")
            samples = []
            with fresh_reader_pool() as pool:
                for pass_number, order in enumerate(ORDERS, 1):
                    for a, b in ("AC", "BD"):
                        choices = dict(zip("ABCD", (("gzip", a), ("gzip", b), ("none", a), ("none", b))))
                        for option in order:
                            mode, case = choices[option]
                            sample, _ = pool.submit(measure_case, sources[mode], records[case],
                                                    configs[mode], batches, 1, None).result()
                            samples.append(dict(pass_number=pass_number, compression=mode, case=case,
                                                layout=records[case]["layout"], **sample))
                    print(f"Completed read pass {pass_number}/{len(ORDERS)}", flush=True)
                traced = [pool.submit(instrument_zarr, sources[mode], records[case], configs[mode], batches).result()
                          for case in "CD" for mode in configs]
            cache_mount = mount_info(temp)
        save_csv(output / "samples.csv", samples)
        save_csv(output / "zarr-cpu.csv", traced)
        write_report(output, mount_info(directory), cache_mount, len(os.sched_getaffinity(0)), samples, traced)
        print(f"Report: {output / 'report.md'}", flush=True)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
