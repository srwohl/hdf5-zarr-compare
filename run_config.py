"""Launch a configured benchmark and keep its input with the results."""
import argparse
from datetime import UTC, datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import mkdtemp
import tomllib
from uuid import uuid4

import darshan_summary
import strace_summary


TASKS = {"experiment": "experiment.py", "write-matrix": "write_matrix.py"}


def option_arguments(options):
    if not isinstance(options, dict):
        raise ValueError("options must be a TOML table")
    arguments = []
    for name, value in options.items():
        if name in ("scratch", "help") or not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise ValueError(f"invalid config option {name!r}")
        flag = f"--{name}"
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
        elif isinstance(value, (str, int, float)):
            arguments.append(f"{flag}={value}")
        elif isinstance(value, list) and value:
            if name in ("rates", "writers") and all(type(item) is int for item in value):
                arguments.append(f"{flag}={','.join(map(str, value))}")
            elif name == "geometry" and all(isinstance(item, str) for item in value):
                arguments.extend(f"{flag}={item}" for item in value)
            else:
                raise ValueError(f"unsupported array for option {name!r}")
        else:
            raise ValueError(f"unsupported value for option {name!r}")
    return arguments


def darshan_environment(library, logs, excluded, trace):
    """Preload Darshan into the benchmark processes. Logs go to a node-local directory."""
    # Builds configured with --with-log-path-by-env read DARSHAN_LOG_DIR_PATH directly.
    # Others use DARSHAN_LOGPATH, which expects darshan-mk-log-dirs' YEAR/MONTH/DAY layout.
    today = datetime.now()
    (logs / str(today.year) / str(today.month) / str(today.day)).mkdir(parents=True)
    environment = dict(os.environ, DARSHAN_ENABLE_NONMPI="1", DARSHAN_LOG_DIR_PATH=str(logs),
                       DARSHAN_LOGPATH=str(logs), DARSHAN_EXCLUDE_DIRS=",".join(map(str, excluded)))
    environment["LD_PRELOAD"] = ":".join(filter(None, [str(library), os.environ.get("LD_PRELOAD")]))
    if trace:
        environment["DXT_ENABLE_IO_TRACE"] = "1"
    return environment


def collect_darshan(logs, output):
    found = sorted(logs.rglob("*.darshan"))
    destination = output / "darshan"
    destination.mkdir()
    for log in found:
        shutil.move(log, destination / log.name)
    shutil.rmtree(logs)
    if not found:
        print("Darshan wrote no logs. Check DARSHAN_* settings for your site's build.", file=sys.stderr)
        return
    print(f"Darshan logs: {destination} ({len(found)} files)", flush=True)
    try:
        _, groups = darshan_summary.write_summary(destination, output, str(output))
        print(f"Darshan summary: {output / 'darshan-summary.md'} ({groups} groups)", flush=True)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Darshan summary skipped: {error}", file=sys.stderr)


def strace_command(command, traces):
    """Trace file calls in every process and thread, one output file per thread."""
    strace = shutil.which("strace")
    if strace is None:
        raise ValueError("VOXEL_STRACE needs strace on PATH (try `module load strace`, or `pixi add strace`)")
    calls = ",".join(strace_summary.SYSCALLS)
    return [strace, "-f", "-ff", "-qq", "-tt", "-T", "-y", "-s", "0", "-e", f"trace={calls}",
            "-o", str(traces / "trace"), *command]


def collect_strace(traces, output, returncode):
    found = sorted(traces.glob("trace.*"))
    if not found:
        print(f"strace wrote no traces (exit status {returncode}). ptrace may be disabled on this node: "
              "check /proc/sys/kernel/yama/ptrace_scope.", file=sys.stderr)
        shutil.rmtree(traces)
        return
    try:
        _, groups = strace_summary.write_summary(traces, output, str(output), output.stat().st_blksize)
        print(f"strace summary: {output / 'strace-summary.md'} ({groups} groups)", flush=True)
    except OSError as error:
        print(f"strace summary skipped: {error}", file=sys.stderr)
    archive = shutil.make_archive(str(output / "strace"), "gztar", traces)
    shutil.rmtree(traces)
    print(f"strace traces: {archive} ({len(found)} files)", flush=True)


def run_config(config, project, scratch):
    contents = config.read_bytes()
    settings = tomllib.loads(contents.decode())
    unknown = settings.keys() - {"task", "baseline", "options"}
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    task = settings.get("task")
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError("task must be 'experiment' or 'write-matrix'")
    command = [sys.executable, str(Path(__file__).with_name(TASKS[task]))]
    if "baseline" in settings:
        baseline = settings["baseline"]
        if task != "write-matrix" or not isinstance(baseline, str) or not baseline.strip():
            raise ValueError("baseline must be a path for the write-matrix task")
        command.append(str((config.parent / baseline).resolve()))
    options = option_arguments(settings.get("options", {}))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = project / f"{config.stem}-{stamp}-{uuid4().hex[:8]}"
    print(f"Benchmark output: {output}", flush=True)
    environment = logs = traces = None
    library = os.environ.get("VOXEL_DARSHAN_LIB")
    if library and os.environ.get("VOXEL_STRACE") == "1":
        raise ValueError("use either VOXEL_DARSHAN_LIB or VOXEL_STRACE=1, not both")
    command = [*command, str(output), *options, "--scratch", str(scratch)]
    if os.environ.get("VOXEL_STRACE") == "1":
        traces = Path(mkdtemp(prefix="strace-", dir=scratch))
        command = strace_command(command, traces)
        print("strace: tracing file calls; timings in this run include ptrace overhead", flush=True)
    if library:
        if not Path(library).is_file():
            raise ValueError(f"VOXEL_DARSHAN_LIB is not a file: {library}")
        logs = Path(mkdtemp(prefix="darshan-", dir=scratch))
        environment = darshan_environment(library, logs, [scratch.parent, Path(sys.prefix)],
                                          os.environ.get("VOXEL_DARSHAN_DXT") == "1")
        print(f"Darshan preload: {library}", flush=True)
    returncode = None
    try:
        returncode = subprocess.run(command, check=False, env=environment).returncode
        return returncode
    finally:
        if output.is_dir():
            (output / "config.toml").write_bytes(contents)
            if logs is not None:
                collect_darshan(logs, output)
            if traces is not None:
                collect_strace(traces, output, returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("project", type=Path)
    parser.add_argument("scratch", type=Path)
    args = parser.parse_args()
    try:
        return run_config(args.config, args.project, args.scratch)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
