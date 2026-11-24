"""Profile reconstruction from a verified experiment's stored detector data."""
import argparse
import cProfile
import hashlib
from inspect import unwrap
import io
import json
from os.path import relpath
from pathlib import Path
import pstats
from tempfile import TemporaryDirectory
import time

import numpy as np
import zarr

from experiment import CASES, save_json
import tiled


def function_profile(stats, function):
    code = unwrap(function).__code__
    _, calls, own, cumulative, _ = stats.stats[(code.co_filename, code.co_firstlineno, code.co_name)]
    return calls, own, cumulative


def interpolation_profile(stats):
    calls, _, seconds = function_profile(stats, np.interp)
    return dict(interpolation_calls=calls, interpolation_seconds=seconds)


def digest_volume(path):
    volume = np.load(path, mmap_mode="r")
    digest = hashlib.sha256()
    for start in range(0, len(volume), 8):
        digest.update(memoryview(volume[start:start + 8]).cast("B"))
    return digest.hexdigest()


def profile_case(directory, output, case, config, expected):
    kind = CASES[case][0]
    trace = []
    with TemporaryDirectory(prefix="voxel-profile-") as temporary:
        scratch = Path(temporary)
        # The baseline verified every raw frame against its source. Stage its
        # first frames for tiled.reconstruct's normal input check.
        width = config["tile"]
        frame = (slice(0, 1), slice(0, width), slice(0, width))
        with tiled.raw_reader(directory, config, kind, []) as (read, calibration):
            for tile in range(len(calibration)):
                np.save(scratch / f"tile_{tile}.npy", read(tile, frame, "profile_setup"))
        reconstruction = scratch / "reconstruction.npy"
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.runcall(tiled.reconstruct, directory, config, kind, trace,
                         source=scratch, output=reconstruction)
        wall = time.perf_counter() - started
        if digest_volume(reconstruction) != expected:
            raise ValueError(f"case {case} reconstruction differs from the verified baseline")

    profiler.dump_stats(str(output / f"{case}.prof"))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    _, _, fbp = function_profile(stats, tiled.iradon)
    _, _, stitching = function_profile(stats, tiled.stitched_band)
    interpolation = interpolation_profile(stats)
    stats.strip_dirs().sort_stats("cumulative").print_stats(35)
    stats.sort_stats("tottime").print_stats(35)
    (output / f"{case}-functions.txt").write_text(stream.getvalue())
    raw = sum(read["elapsed_ms"] for read in trace) / 1000
    band_raw = sum(read["elapsed_ms"] for read in trace if read["pattern"] == "sinogram_band") / 1000
    stitch = stitching - band_raw
    return dict(case=case, format=kind, wall_seconds=wall, fbp_seconds=fbp, raw_read_seconds=raw,
                stitch_excluding_reads_seconds=stitch, other_seconds=wall - fbp - raw - stitch,
                raw_read_calls=len(trace), raw_returned_bytes=sum(r["returned_bytes"] for r in trace),
                full_chunk_model_bytes=sum(r["full_chunk_model_bytes"] for r in trace),
                reconstruction_matches=True, **interpolation)


def write_report(output, baseline, rows, row_band):
    report = [
        "# Reconstruction profile", "",
        f"Profiled up to {row_band} detector rows per band from the "
        f"[verified experiment](<{relpath(baseline / 'report.md', output)}>).",
        "Each case reopened the raw data and reproduced its verified reconstruction hash. "
        "Detector simulation, replay, analysis, and result export were outside the profile. "
        "The function profiler uses one application-level reader. The first-frame setup read occurs before "
        "profiling. Cache state was uncontrolled.", "",
        "| Case | Reconstruction wall (s) | Filtered backprojection (s) | Detector reads (s) | "
        "Stitching and correction (s) | Other (s) | Raw read calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(f"| {row['case']} | {row['wall_seconds']:.3f} | {row['fbp_seconds']:.3f} | "
                      f"{row['raw_read_seconds']:.3f} | {row['stitch_excluding_reads_seconds']:.3f} | "
                      f"{row['other_seconds']:.3f} | {row['raw_read_calls']} |")
    report += ["", "Filtered backprojection and stitching use cumulative cProfile wall time. "
               "Detector reads include the array library and decoder work, plus any filesystem access. "
               "Other includes calibration, setup, output writes and flush, and profiler overhead.", "",
               "## Interpolation inside backprojection", "",
               "| Case | NumPy interpolation calls | Interpolation (s) | Share of reconstruction wall |",
               "|---|---:|---:|---:|"]
    for row in rows:
        fraction = row["interpolation_seconds"] / row["wall_seconds"]
        report.append(f"| {row['case']} | {row['interpolation_calls']:,} | "
                      f"{row['interpolation_seconds']:.3f} | {fraction:.1%} |")
    report.append("")
    for row in rows:
        case = row["case"]
        report += [f"- Case {case} [function timings]({case}-functions.txt) and "
                   f"[cProfile data]({case}.prof)."]
    (output / "report.md").write_text("\n".join(report) + "\n")
    save_json(output / "profile.json", {"baseline": str(baseline), "row_band": row_band, "cases": rows})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, help="completed experiment directory")
    p.add_argument("output", type=Path, help="new directory for profiling artifacts")
    p.add_argument("--cases", default="ABCD", help="case IDs in comparison order")
    p.add_argument("--row-band", type=int, help="override the baseline's reconstruction row band")
    args = p.parse_args()
    try:
        baseline = args.directory.resolve()
        metadata = json.loads((baseline / "experiment.json").read_text())
        if not metadata["complete"] or not metadata["verified"]:
            raise ValueError("a completed, verified experiment is required")
        available = {row["case"]: row for row in metadata["case_results"]}
        if not args.cases or len(set(args.cases)) != len(args.cases) or set(args.cases) - available.keys():
            raise ValueError("case IDs must be unique and present in the experiment")
        raw = metadata["raw"]
        zarr.config.set({"async.concurrency": raw.get("zarr_concurrency", zarr.config.get("async.concurrency"))})
        row_band = args.row_band if args.row_band is not None else raw["row_band"]
        if row_band < 1:
            raise ValueError("row band must be positive")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        for case in args.cases:
            print(f"Profiling case {case} from stored detector data", flush=True)
            record = available[case]
            config = dict(raw, row_band=row_band, raw_chunks=record["raw_chunks"], read_workers=1)
            row = profile_case(baseline / case, output, case, config, record["reconstruction_sha256"])
            rows.append(row)
            write_report(output, baseline, rows, row_band)
            print(f"Case {case}: {row['wall_seconds']:.2f} s including {row['fbp_seconds']:.2f} s backprojection",
                  flush=True)
        print(f"Report: {output / 'report.md'}", flush=True)
    except (OSError, ValueError, KeyError) as error:
        p.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
