"""Summarize strace file I/O per write-matrix trial or experiment case.

A fallback for nodes without Darshan. ptrace stops every traced call, so use the
counts, sizes and offsets from a traced run, not its throughput.
"""
import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path
import re
from statistics import quantiles

from darshan_summary import group_for

SYSCALLS = ("open", "openat", "creat", "read", "pread64", "readv", "preadv", "write", "pwrite64", "writev",
            "pwritev", "pwritev2", "preadv2", "lseek", "fsync", "fdatasync", "close", "rename", "renameat", "renameat2",
            "mkdir", "mkdirat", "unlink", "unlinkat", "fallocate", "ftruncate")
LINE = re.compile(r"^(?:\d+ )?[\d:.]+ (\w+)\((.*)\) += (.*?) <([\d.]+)>$")
FD_PATH = re.compile(r"^-?\d+<([^>]*)>")
QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')
RESULT = re.compile(r"^(-?\d+)(?:<([^>]*)>)?")
WRITES = {"write", "pwrite64", "writev", "pwritev", "pwritev2"}
READS = {"read", "pread64", "readv", "preadv", "preadv2"}
EXPLICIT_OFFSET = {"pwrite64", "pread64", "pwritev", "pwritev2", "preadv", "preadv2"}
MIB = 1 << 20


def call_path(syscall, arguments, result_path):
    """The file a call acts on, using strace -y decorations where available."""
    if syscall in ("open", "openat", "creat") and result_path:
        return result_path
    match = FD_PATH.match(arguments)
    if match and syscall not in ("openat", "mkdirat", "renameat", "renameat2", "unlinkat"):
        return match.group(1)
    quoted = QUOTED.search(arguments)
    if not quoted:
        return None
    path = quoted.group(1)
    if not path.startswith("/") and match:
        path = os.path.join(match.group(1), path)
    return path


def parse_trace(lines, prefix=None):
    """Yield one event per completed call in a single-thread strace -ff file."""
    position = {}
    for line in lines:
        match = LINE.match(line.rstrip("\n"))
        if not match:
            continue
        syscall, arguments, result, seconds = match.groups()
        if syscall not in SYSCALLS:
            continue
        value, result_path = RESULT.match(result).groups() if RESULT.match(result) else (None, None)
        value = int(value) if value is not None else None
        path = call_path(syscall, arguments, result_path)
        if path is None or (prefix and not path.startswith(prefix)):
            continue
        offset = None
        if syscall in ("open", "openat", "creat") and value is not None and value >= 0:
            position[path] = 0
        elif syscall == "lseek" and value is not None and value >= 0:
            position[path] = value
        elif syscall in WRITES | READS and value is not None and value >= 0:
            if syscall in EXPLICIT_OFFSET:
                # pwritev2 and preadv2 end with a flags argument after the offset.
                offset = int(arguments.rsplit(",", 2 if syscall.endswith("2") else 1)[1])
            else:
                offset = position.get(path)
                if offset is not None:
                    position[path] = offset + value
        yield dict(syscall=syscall, path=path, value=value, seconds=float(seconds), offset=offset,
                   flags=arguments)


def summarize(events, block):
    groups = defaultdict(lambda: defaultdict(float))
    files = defaultdict(set)
    directories = defaultdict(set)
    fsync_times = defaultdict(list)
    for event in events:
        group = group_for(event["path"])
        if group is None:
            continue
        totals, syscall, value = groups[group], event["syscall"], event["value"]
        failed = value is None or value < 0
        files[group].add(event["path"])
        if syscall in ("open", "openat", "creat"):
            totals["failed_opens" if failed else "opens"] += 1
            totals["open_seconds"] += event["seconds"]
            if not failed and "O_CREAT" in event["flags"]:
                totals["creates"] += 1
            if not failed and "O_DIRECTORY" in event["flags"]:
                directories[group].add(event["path"])
        elif failed:
            totals["failed_calls"] += 1
        elif syscall in WRITES:
            totals["writes"] += 1
            totals["bytes_written"] += value
            totals["write_seconds"] += event["seconds"]
            totals["writes_under_1mib"] += value < MIB
            totals["writes_full_block_or_larger"] += value >= block
            if event["offset"] is None:
                totals["writes_unknown_offset"] += 1
            else:
                totals["writes_unaligned_start"] += event["offset"] % block != 0
                end = event["offset"] + value
                totals["writes_partial_block"] += event["offset"] % block != 0 or end % block != 0
        elif syscall in READS:
            totals["reads"] += 1
            totals["bytes_read"] += value
            totals["read_seconds"] += event["seconds"]
        elif syscall in ("fsync", "fdatasync"):
            kind = "directory_fsyncs" if event["path"] in directories[group] else "file_fsyncs"
            totals[kind] += 1
            totals["fsync_seconds"] += event["seconds"]
            fsync_times[group].append(event["seconds"])
        elif syscall in ("rename", "renameat", "renameat2"):
            totals["renames"] += 1
            totals["rename_seconds"] += event["seconds"]
        elif syscall in ("mkdir", "mkdirat"):
            totals["mkdirs"] += 1
        elif syscall == "close":
            totals["close_seconds"] += event["seconds"]
    keys = ("opens", "creates", "failed_opens", "failed_calls", "writes", "bytes_written", "writes_under_1mib",
            "writes_full_block_or_larger", "writes_unaligned_start", "writes_partial_block",
            "writes_unknown_offset", "reads", "bytes_read", "file_fsyncs", "directory_fsyncs", "renames", "mkdirs",
            "open_seconds", "write_seconds", "read_seconds", "fsync_seconds", "rename_seconds", "close_seconds")
    rows = []
    for group in sorted(groups):
        totals, times = groups[group], sorted(fsync_times[group])
        p95 = quantiles(times, n=20)[-1] if len(times) > 1 else (times[0] if times else 0.)
        rows.append(dict(group=group, block_bytes=block, files=len(files[group]),
                         **{key: totals[key] for key in keys},
                         mean_write_bytes=totals["bytes_written"] / totals["writes"] if totals["writes"] else 0,
                         fsync_p95_seconds=p95, fsync_max_seconds=times[-1] if times else 0.))
    return rows


def markdown(rows, block):
    size = f"{block / MIB:g} MiB" if block >= MIB else f"{block / 1024:g} KiB"
    lines = ["# strace file I/O summary", "",
             f"Calls on files in each trial or case, checked against {size} blocks. ptrace stops each traced "
             "call, so use counts, sizes and offsets here and take throughput from an untraced run. "
             "**Unaligned** writes start off a block boundary. **Partial-block** writes start or end inside a block; "
             "on GPFS those can need a read-modify-write. The final write of a file usually ends inside a block. "
             "Seconds are summed over threads.", "",
             "| Trial or case | Files | Opens | Creates | Writes | Mean write MiB | <1 MiB | Unaligned | Partial-block | "
             "File fsyncs | Dir fsyncs | Renames | Write s | fsync s | fsync P95 ms | Rename s | Open s | Close s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['group']} | {row['files']} | {row['opens']:.0f} | {row['creates']:.0f} | "
                     f"{row['writes']:.0f} | {row['mean_write_bytes'] / MIB:.2f} | {row['writes_under_1mib']:.0f} | "
                     f"{row['writes_unaligned_start']:.0f} | {row['writes_partial_block']:.0f} | "
                     f"{row['file_fsyncs']:.0f} | {row['directory_fsyncs']:.0f} | {row['renames']:.0f} | "
                     f"{row['write_seconds']:.3f} | {row['fsync_seconds']:.3f} | {row['fsync_p95_seconds'] * 1e3:.1f} | "
                     f"{row['rename_seconds']:.3f} | {row['open_seconds']:.3f} | {row['close_seconds']:.3f} |")
    if any(row["writes_unknown_offset"] for row in rows):
        lines += ["", "Some writes came from a thread that did not open the file, so their offsets are unknown. "
                      "They are counted in `writes_unknown_offset` in the CSV."]
    return "\n".join(lines) + "\n"


def write_summary(trace_directory, output, prefix, block):
    traces = sorted(trace_directory.glob("trace.*"))
    if not traces:
        raise FileNotFoundError(f"no strace output under {trace_directory}")

    def events():
        for trace in traces:
            with trace.open(errors="replace") as handle:
                yield from parse_trace(handle, prefix)

    rows = summarize(events(), block)
    if rows:
        with (output / "strace-summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output / "strace-summary.md").write_text(markdown(rows, block))
    return len(traces), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="benchmark output directory")
    parser.add_argument("traces", type=Path, help="directory of strace -ff trace.PID files")
    parser.add_argument("--block", type=int, help="block bytes (default: st_blksize of OUTPUT)")
    args = parser.parse_args()
    output = args.output.resolve()
    block = args.block or output.stat().st_blksize
    try:
        traces, groups = write_summary(args.traces, output, str(output), block)
    except OSError as error:
        parser.exit(1, f"{error}\n")
    print(f"Summarized {traces} trace files into {groups} groups: {output / 'strace-summary.md'}")


if __name__ == "__main__":
    main()
