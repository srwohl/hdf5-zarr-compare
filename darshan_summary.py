"""Summarize Darshan POSIX counters per write-matrix trial or experiment case."""
import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess

SIZE_BUCKETS = ("0_100", "100_1K", "1K_10K", "10K_100K", "100K_1M", "1M_4M", "4M_10M", "10M_100M",
                "100M_1G", "1G_PLUS")
SUMMED = ("POSIX_OPENS", "POSIX_READS", "POSIX_WRITES", "POSIX_SEEKS", "POSIX_STATS", "POSIX_FSYNCS",
          "POSIX_FDSYNCS", "POSIX_RENAME_SOURCES", "POSIX_BYTES_READ", "POSIX_BYTES_WRITTEN",
          "POSIX_FILE_NOT_ALIGNED", "POSIX_MEM_NOT_ALIGNED", "POSIX_F_READ_TIME", "POSIX_F_WRITE_TIME",
          "POSIX_F_META_TIME", *(f"POSIX_SIZE_WRITE_{bucket}" for bucket in SIZE_BUCKETS))
TRIAL = re.compile(r"/(pass\d+-[^/]+)-[A-Za-z0-9_]{8}(?:/|$)")
CASE = re.compile(r"/([ABCD])/")


def group_for(path):
    """Write-matrix trials use labelled temporary directories; experiments use case directories."""
    match = TRIAL.search(path) or CASE.search(path)
    return match.group(1) if match else None


def parse_records(lines, prefix=None):
    """Yield (group, counter, value, file system type) for POSIX records in darshan-parser output."""
    for line in lines:
        if line.startswith("#") or not line.startswith("POSIX\t"):
            continue
        columns = line.rstrip("\n").split("\t")
        if len(columns) < 8:
            continue
        _, _, _, counter, value, name, _, fstype = columns[:8]
        if prefix and not name.startswith(prefix):
            continue
        group = group_for(name)
        if group is not None:
            yield group, counter, float(value), name, fstype


def summarize(records):
    totals = defaultdict(lambda: defaultdict(float))
    files = defaultdict(set)
    alignment = defaultdict(set)
    fstypes = defaultdict(set)
    for group, counter, value, name, fstype in records:
        files[group].add(name)
        fstypes[group].add(fstype)
        if counter in SUMMED and value > 0:
            totals[group][counter] += value
        elif counter == "POSIX_FILE_ALIGNMENT" and value > 0:
            alignment[group].add(int(value))
    rows = []
    for group in sorted(files):
        values = totals[group]
        writes = values["POSIX_WRITES"]
        rows.append(dict(group=group, fstype="/".join(sorted(fstypes[group])), files=len(files[group]),
                         file_alignment_bytes="/".join(map(str, sorted(alignment[group]))),
                         mean_write_bytes=values["POSIX_BYTES_WRITTEN"] / writes if writes else 0,
                         **{counter.removeprefix("POSIX_").lower(): values[counter] for counter in SUMMED}))
    return rows


def markdown(rows):
    lines = ["# Darshan POSIX summary", "",
             "Counters are summed over every process log and every file in each trial or case. "
             "`Not aligned` counts reads and writes that did not start on the file alignment Darshan "
             "recorded for that filesystem. Meta seconds include open, stat, seek, fsync and rename "
             "time. Time counters overlap when workers run in parallel.", "",
             "| Trial or case | FS | Files | Opens | Writes | Mean write MiB | Not aligned | Alignment | "
             "fsyncs | Renames | Write s | Meta s | Writes ≤1 MiB | Writes 1–4 MiB | Writes 4–10 MiB | Writes >10 MiB |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        small = sum(row[f"size_write_{bucket}".lower()] for bucket in SIZE_BUCKETS[:5])
        large = sum(row[f"size_write_{bucket}".lower()] for bucket in SIZE_BUCKETS[7:])
        lines.append(f"| {row['group']} | {row['fstype']} | {row['files']} | {row['opens']:.0f} | "
                     f"{row['writes']:.0f} | {row['mean_write_bytes'] / 2**20:.2f} | {row['file_not_aligned']:.0f} | "
                     f"{row['file_alignment_bytes']} | {row['fsyncs'] + row['fdsyncs']:.0f} | "
                     f"{row['rename_sources']:.0f} | {row['f_write_time']:.3f} | {row['f_meta_time']:.3f} | "
                     f"{small:.0f} | {row['size_write_1m_4m']:.0f} | {row['size_write_4m_10m']:.0f} | {large:.0f} |")
    return "\n".join(lines) + "\n"


def parser_lines(logs, parser):
    for log in logs:
        result = subprocess.run([parser, str(log)], capture_output=True, text=True, check=True,
                                env={key: value for key, value in os.environ.items() if key != "LD_PRELOAD"})
        yield from result.stdout.splitlines()


def write_summary(log_directory, output, prefix=None, parser=None):
    parser = parser or shutil.which("darshan-parser")
    if parser is None:
        raise FileNotFoundError("darshan-parser is not on PATH; load the Darshan module first")
    logs = sorted(log_directory.rglob("*.darshan"))
    if not logs:
        raise FileNotFoundError(f"no Darshan logs under {log_directory}")
    rows = summarize(parse_records(parser_lines(logs, parser), prefix))
    if rows:
        with (output / "darshan-summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output / "darshan-summary.md").write_text(markdown(rows))
    return len(logs), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="benchmark output directory containing darshan/")
    parser.add_argument("--logs", type=Path, help="Darshan log directory (default: OUTPUT/darshan)")
    args = parser.parse_args()
    try:
        logs, groups = write_summary(args.logs or args.output / "darshan", args.output, str(args.output.resolve()))
    except (OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Summarized {logs} logs into {groups} groups: {args.output / 'darshan-summary.md'}")


if __name__ == "__main__":
    main()
