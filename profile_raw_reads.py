"""Compare raw detector reads at matched layouts without reconstructing."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import multiprocessing
import os
import resource
from pathlib import Path
from statistics import median
import time

import numpy as np
import zarr

from demo import MIB
from experiment import mount_info, save_csv, save_json
from feature import process_io
from tiled import parallel_raw_batches, raw_reader, request_batches

# Each case appears in each position and immediately follows every other case.
ORDERS = ("ABDC", "BCAD", "CDBA", "DACB")


def load_trace(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def trace_requests(trace):
    return [(int(row["tile"]), tuple(slice(*bounds) for bounds in json.loads(row["selection"])),
             row["pattern"]) for row in trace]


def cpu_time():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def parallel_sample(directory, config, kind, batches, workers, digest):
    trace = []
    with parallel_raw_batches(directory, config, kind, batches, workers) as pending:
        before = process_io()
        read_started = time.perf_counter_ns()
        for batch, (arrays, rows) in enumerate(pending):
            delivered = time.perf_counter_ns()
            for row in rows:
                row.update(batch=batch, delivered_ns=delivered)
            trace.extend(rows)
            if digest is not None:
                for values in arrays:
                    digest.update(memoryview(np.ascontiguousarray(values)).cast("B"))
        after = process_io()
    if kind == "nexus":
        # The parent process's I/O counters do not describe its reader processes.
        before = after = {}
    return trace, read_started, before, after


def read_case(directory, record, raw, batches, workers, *, verify=False):
    trace = []
    digest = hashlib.sha256() if verify else None
    config = dict(raw, raw_chunks=record["raw_chunks"])
    directory = directory / record["case"]
    kind = record["format"]
    cpu_start = cpu_time()
    startup_started = time.perf_counter_ns()
    if workers > 1:
        trace, read_started, before, after = parallel_sample(directory, config, kind, batches, workers, digest)
    else:
        with raw_reader(directory, config, kind, trace, load_calibration=False) as (read, _):
            before = process_io()
            read_started = time.perf_counter_ns()
            for batch, requests in enumerate(batches):
                first = len(trace)
                for request in requests:
                    values = read(*request)
                    if digest is not None:
                        digest.update(memoryview(np.ascontiguousarray(values)).cast("B"))
                delivered = time.perf_counter_ns()
                for row in trace[first:]:
                    row.update(batch=batch, delivered_ns=delivered)
            after = process_io()
    cpu = cpu_time() - cpu_start
    if digest is not None:
        return digest.hexdigest(), []
    first_band = next(row["delivered_ns"] for row in trace if row["pattern"] == "sinogram_band")
    last_batch = trace[-1]["delivered_ns"]
    read_calls = sum(row["elapsed_ms"] for row in trace) / 1000
    lookup = sum(row["array_lookup_ms"] for row in trace) / 1000

    def delta(key):
        return after[key] - before[key] if key in before and key in after else None

    return dict(measurement_pid=os.getpid(), startup_started_ns=startup_started, read_started_ns=read_started,
                time_to_first_band_seconds=(first_band - startup_started) / 1e9,
                time_to_last_batch_seconds=(last_batch - startup_started) / 1e9,
                setup_seconds=(read_started - startup_started) / 1e9,
                read_seconds=(last_batch - read_started) / 1e9,
                read_call_seconds_sum=read_calls, array_lookup_seconds=lookup,
                slice_seconds=read_calls - lookup, cpu_seconds=cpu,
                logical_read_bytes=delta("rchar"), process_storage_read_bytes=delta("read_bytes"),
                read_syscalls=delta("syscr"), returned_bytes=sum(row["returned_bytes"] for row in trace),
                full_chunk_model_bytes=sum(row["full_chunk_model_bytes"] for row in trace)), trace


def fresh_reader_pool():
    # Each sample starts with a fresh application process for both formats.
    return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                               max_tasks_per_child=1)


def measure_case(directory, record, raw, batches, workers, concurrency):
    if concurrency is None:
        return read_case(directory, record, raw, batches, workers)
    with zarr.config.set({"async.concurrency": concurrency}):
        return read_case(directory, record, raw, batches, workers)


def stats(samples, cases):
    keys = ("time_to_first_band_seconds", "time_to_last_batch_seconds", "read_seconds", "setup_seconds", "array_lookup_seconds", "slice_seconds", "cpu_seconds",
            "logical_read_bytes", "process_storage_read_bytes", "read_syscalls",
            "returned_bytes", "full_chunk_model_bytes")
    result = {}
    for case in cases:
        passed = [row for row in samples if row["case"] == case]
        result[case] = {key: median(row[key] for row in passed)
                        for key in keys if all(row[key] is not None for row in passed)}
    return result


def write_report(output, records, samples, requests, config, mount, environment, original_seconds):
    summaries = stats(samples, records)
    max_threads = environment["zarr_threading_max_workers"]
    threads = "library default" if max_threads is None else str(max_threads)
    configuration = "inherited runtime settings" if environment["zarr_native_settings"] else "profile concurrency override"
    lines = ["# Matched detector-read profile", "",
             f"Replayed {len(requests)} identical selections from a verified acquisition, "
             f"with {len(samples) // len(records)} measured passes per case. Each case first ran one verification pass "
             "to check returned bytes and warm the filesystem cache. Measured passes use fresh Python processes.", "",
             "**Common start.** The read clock starts after Python imports, before opening an array or launching "
             "reader workers. First band is delivery of the first sinogram row band after the frame checks. "
             "Last batch is delivery of the final request. Both endpoints include reader setup, dispatch and "
             "transfer to the caller.", "",
             f"Filesystem **{mount.get('fstype', 'unknown')}**. Compression **{config['compression']}**. "
             f"Shared CPU affinity mask **{', '.join(map(str, environment['cpu_affinity']))}**. "
             f"Application reader workers **{environment['read_workers']}**. "
             f"Zarr used {configuration} with async concurrency {environment['zarr_async_concurrency']} "
             f"and thread pool size {threads}. "
             f"Every pass returned {samples[0]['returned_bytes'] / MIB:.2f} MiB. Times are medians per pass.", "",
             "| Case | Format / layout | First band s | Last batch s | Setup to dispatch s | Dispatch to last batch s |",
             "|---|---|---:|---:|---:|---:|"]
    for case, record in records.items():
        row = summaries[case]
        lines.append(f"| {case} | {record['format']} / {record['layout']} | "
                     f"{row['time_to_first_band_seconds']:.3f} | {row['time_to_last_batch_seconds']:.3f} | "
                     f"{row['setup_seconds']:.3f} | {row['read_seconds']:.3f} |")
    lines += ["", "## Read diagnostics", "",
              "| Case | Array lookup s sum | Slice s sum | Process CPU s | Read-call MiB | "
              "Process storage MiB | Read syscalls | Full-chunk MiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]

    def show(data, key, scale=1., precision=3):
        return f"{data[key] / scale:.{precision}f}" if key in data else "—"

    for case, row in summaries.items():
        lines.append(f"| {case} | {row['array_lookup_seconds']:.3f} | "
                     f"{row['slice_seconds']:.3f} | {row['cpu_seconds']:.3f} | "
                     f"{show(row, 'logical_read_bytes', MIB, 2)} | {show(row, 'process_storage_read_bytes', MIB, 2)} | "
                     f"{show(row, 'read_syscalls', precision=0)} | {row['full_chunk_model_bytes'] / MIB:.2f} |")
    lines += ["", "Array lookup times the per-request retrieval of the detector array. Slice time includes selection, "
              "library scheduling, compressed reads, decoding and writing the returned array to memory. "
              "Process counters are Linux /proc/self/io deltas and include cache-served calls. "
              "Full-chunk coverage models uncompressed bytes per request. "
              "The parent process's read counters are omitted for HDF5 subprocess readers. "
              "CPU totals include reader startup and shutdown. Read-call counters cover dispatch through delivery.", "",
              "## Paired differences", "",
              "Positive gaps mean Zarr took longer. Each gap compares formats within the same pass.", "",
              "| Matched layout | First band gap | Last batch gap, median (range) | Dispatch to last batch gap |",
              "|---|---:|---:|---:|"]
    paired = {}
    for layout, a, b in (("acquisition", "A", "C"), ("bricks", "B", "D")):
        hdf = {r["pass_number"]: r for r in samples if r["case"] == a}
        zar = {r["pass_number"]: r for r in samples if r["case"] == b}

        def gaps(key):
            return [zar[i][key] - hdf[i][key] for i in hdf]

        first_gaps = gaps("time_to_first_band_seconds")
        last_gaps = gaps("time_to_last_batch_seconds")
        dispatch_gaps = gaps("read_seconds")
        paired[layout] = dict(original_gap_seconds=original_seconds[b] - original_seconds[a],
                              median_difference_seconds=median(dispatch_gaps),
                              range_seconds=[min(dispatch_gaps), max(dispatch_gaps)],
                              median_ratio=median(zar[i]["read_seconds"] / hdf[i]["read_seconds"] for i in hdf),
                              first_band_gap_seconds=median(first_gaps), last_batch_gap_seconds=median(last_gaps),
                              last_batch_gap_range_seconds=[min(last_gaps), max(last_gaps)])
        lines.append(f"| {layout} ({b} − {a}) | {median(first_gaps):+.3f} s | {median(last_gaps):+.3f} s "
                     f"({min(last_gaps):+.3f} to {max(last_gaps):+.3f} s) | {median(dispatch_gaps):+.3f} s |")
    lines += ["", "[Per-pass measurements](samples.csv) and [individual requests](reads.csv).", ""]
    (output / "report.md").write_text("\n".join(lines))
    save_json(output / "summary.json", dict(mount=mount, environment=environment, medians=summaries, paired=paired))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="completed four-case experiment")
    parser.add_argument("output", type=Path, help="new directory for read-profile artifacts")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--read-workers", type=int, help="override the experiment's detector reader count")
    zarr_settings = parser.add_mutually_exclusive_group()
    zarr_settings.add_argument("--zarr-concurrency", type=int,
                               help="override Zarr async.concurrency setting for chunk scheduling")
    zarr_settings.add_argument("--native-zarr", action="store_true",
                               help="inherit Zarr's runtime concurrency and thread-pool settings")
    args = parser.parse_args()
    try:
        if args.repeats < 1:
            raise ValueError("repeats must be positive")
        directory = args.directory.resolve()
        metadata = json.loads((directory / "experiment.json").read_text())
        if not metadata["verified"]:
            raise ValueError("a verified four-case experiment is required")
        records = {r["case"]: r for r in sorted(metadata["case_results"], key=lambda r: r["case"])}
        original_traces = {case: load_trace(directory / case / "raw-reads.csv") for case in records}
        requests = trace_requests(original_traces["A"])
        if not requests or any(trace_requests(trace) != requests for trace in original_traces.values()):
            raise ValueError("cases must contain the same nonempty raw read trace")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        raw = metadata["raw"]
        workers = raw.get("read_workers", 1) if args.read_workers is None else args.read_workers
        concurrency = None
        if not args.native_zarr:
            concurrency = (raw.get("zarr_concurrency", zarr.config.get("async.concurrency"))
                           if args.zarr_concurrency is None else args.zarr_concurrency)
        effective_concurrency = zarr.config.get("async.concurrency") if concurrency is None else concurrency
        if min(workers, effective_concurrency) < 1:
            raise ValueError("reader counts and Zarr concurrency must be positive")
        batches = list(request_batches(raw))
        if list(itertools.chain.from_iterable(batches)) != requests:
            raise ValueError("recorded reads do not match the configured detector bands")
        workers = min(workers, len(batches))
        if concurrency is not None:
            zarr.config.set({"async.concurrency": concurrency})
        print("Verifying and warming the recorded detector reads", flush=True)
        digests = {read_case(directory, record, raw, batches, workers, verify=True)[0] for record in records.values()}
        if len(digests) != 1:
            raise ValueError("raw selection bytes differ across cases")

        samples, reads = [], []
        with fresh_reader_pool() as pool:
            for pass_number, order in enumerate(itertools.islice(itertools.cycle(ORDERS), args.repeats), start=1):
                for case in order:
                    sample, trace = pool.submit(measure_case, directory, records[case], raw, batches,
                                                    workers, concurrency).result()
                    samples.append(dict(pass_number=pass_number, order=order, case=case, **sample))
                    reads.extend(dict(pass_number=pass_number, case=case, request=index, **row)
                                 for index, row in enumerate(trace))
                print(f"Completed pass {pass_number}/{args.repeats} in order {order}", flush=True)
        save_csv(output / "samples.csv", samples)
        save_csv(output / "reads.csv", reads)
        cpu_affinity = sorted(os.sched_getaffinity(0))
        environment = dict(available_cpus=len(cpu_affinity), cpu_affinity=cpu_affinity, read_workers=workers,
                           zarr_async_concurrency=effective_concurrency,
                           zarr_threading_max_workers=zarr.config.get("threading.max_workers"),
                           zarr_native_settings=args.native_zarr, fresh_application_process=True)
        original_seconds = {case: sum(float(row["elapsed_ms"]) for row in trace) / 1000
                            for case, trace in original_traces.items()}
        write_report(output, records, samples, requests, raw, mount_info(directory), environment, original_seconds)
        print(f"Report: {output / 'report.md'}", flush=True)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
