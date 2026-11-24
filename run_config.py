"""Launch a configured benchmark and keep its input with the results."""
import argparse
from datetime import UTC, datetime
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from uuid import uuid4


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
    try:
        return subprocess.run([*command, str(output), *options, "--scratch", str(scratch)], check=False).returncode
    finally:
        if output.is_dir():
            (output / "config.toml").write_bytes(contents)


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
