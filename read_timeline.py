"""Draw detector reads from reader setup to final delivery on a shared time axis."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import zarr

from experiment import CASES, mount_info, save_csv
from profile_raw_reads import fresh_reader_pool, load_trace, measure_case, read_case, trace_requests
from tiled import request_batches

INK = "#142b42"
MUTED = "#486074"
BAND_COLORS = ("#60abc1", "#91c6a9", "#a0b5df", "#c4b4df")
CHECKS = "#b7c3cf"
OVERLAP = "#ffefbd"
SETUP = "#e8dff0"


def worker(row):
    return row["worker_pid"], row["worker_thread"]


def overlapping_reads(trace):
    events = defaultdict(int)
    for row in trace:
        events[row["started_ns"]] += 1
        events[row["finished_ns"]] -= 1
    active = 0
    previous = None
    for now, delta in sorted(events.items()):
        if active >= 2:
            yield previous, now
        active += delta
        previous = now


def draw_pass(ax, title, sample, trace, max_seconds):
    origin = sample["startup_started_ns"]
    lanes = list(dict.fromkeys(worker(row) for row in sorted(trace, key=lambda r: r["started_ns"])))
    consume = len(lanes)
    grouped = defaultdict(list)
    for row in trace:
        grouped[row["batch"]].append(row)

    def seconds(ns):
        return (ns - origin) / 1e9

    dispatch = seconds(sample["read_started_ns"])
    ax.axvspan(0, dispatch, color=SETUP, lw=0, zorder=0)
    for first, last in overlapping_reads(trace):
        ax.axvspan(seconds(first), seconds(last), color=OVERLAP, lw=0, zorder=0)
    ax.hlines(range(consume + 1), 0, max_seconds, color="#dfe6ec", lw=1, zorder=1)
    for batch, rows in sorted(grouped.items()):
        position = lanes.index(worker(rows[0]))
        start = seconds(rows[0]["started_ns"])
        stop = seconds(rows[-1]["finished_ns"])
        color = BAND_COLORS[(batch - 1) % len(BAND_COLORS)] if batch else CHECKS
        ax.barh(position, stop - start, left=start, height=.55, color=color,
                edgecolor=INK, linewidth=.6, zorder=3)
        if 0 < batch <= 3 and stop - start >= max_seconds * .027:
            ax.text((start + stop) / 2, position, str(batch), color=INK,
                    va="center", ha="center", fontsize=8.5, fontweight="bold", zorder=4)
        ax.plot(seconds(rows[-1]["delivered_ns"]), consume, "o", ms=4.8, color=color,
                markeredgecolor=INK, markeredgewidth=.65, zorder=4)

    elapsed = sample["time_to_last_batch_seconds"]
    ax.axvline(elapsed, color=INK, lw=1, ls=(0, (2, 3)), alpha=.6)
    ax.set(xlim=(0, max_seconds), ylim=(consume + .6, -.62))
    ax.set_yticks(range(consume + 1), [f"Reader {i+1}" for i in range(consume)] + ["Ready in order"])
    ax.tick_params(axis="y", length=0, pad=10, colors=INK, labelsize=10)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.grid(axis="x", color="#e4eaf0", lw=.7, zorder=1)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color("#c4d0da")

    ax.text(0, 1.13, title, transform=ax.transAxes, ha="left", va="bottom",
            color=INK, fontsize=14, fontweight="bold")
    ax.text(1, 1.13, f"{elapsed:.2f} s total · {sample['time_to_first_band_seconds']:.2f} s first band",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=10)


def draw_timeline(output, subtitle, measured, note):
    heights = [len({worker(row) for row in trace}) + 1 for _, _, trace in measured]
    fig, axes = plt.subplots(2, 1, figsize=(11.9, 6.8), gridspec_kw={"height_ratios": heights})
    fig.subplots_adjust(left=.17, right=.96, bottom=.21, top=.76, hspace=.66)
    max_seconds = max(sample["time_to_last_batch_seconds"] for _, sample, _ in measured) * 1.055
    fig.text(.06, .94, "Detector reads from a common start", color=INK, fontsize=23, fontweight="bold")
    fig.text(.06, .888, subtitle, color=MUTED, fontsize=11.5)
    handles = [Patch(facecolor=SETUP, label="Reader setup"),
               Patch(facecolor=CHECKS, edgecolor=INK, label="Frame checks"),
               Patch(facecolor=BAND_COLORS[0], edgecolor=INK, label="Detector row band"),
               Patch(facecolor=OVERLAP, label="Two read calls active")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, .858), frameon=False,
               ncol=4, fontsize=9.5, labelcolor=MUTED)
    for ax, (title, sample, trace) in zip(axes, measured):
        draw_pass(ax, title, sample, trace, max_seconds)
    axes[-1].set_xlabel("Seconds after reader setup begins", color=INK, labelpad=7)
    fig.text(.06, .097, note, color=INK, fontsize=9.4)
    fig.text(.06, .060, "Dots mark delivery in request order. Bars span the tile reads for one batch.",
             color=MUTED, fontsize=9.3)
    fig.text(.06, .026, "Clock starts before opening data or launching readers. Setup and transfer are included.",
             color=MUTED, fontsize=9.3)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def draw_matched_profile(directory, case, output):
    summary = json.loads((directory / "summary.json").read_text())
    environment = summary["environment"]
    if not environment.get("fresh_application_process"):
        raise ValueError("regenerate the read profile to capture the common start")
    layout = CASES[case][1]
    cases = [letter for letter, (_, geometry) in CASES.items() if geometry == layout]
    matched = defaultdict(dict)
    for row in load_trace(directory / "samples.csv"):
        if row["case"] in cases:
            sample = {key: int(row[key]) for key in ("startup_started_ns", "read_started_ns")}
            sample.update({key: float(row[key])
                           for key in ("time_to_first_band_seconds", "time_to_last_batch_seconds")})
            matched[row["pass_number"]][row["case"]] = sample
    complete = {i: pair for i, pair in matched.items() if len(pair) == len(cases)}
    if not complete:
        raise ValueError("no measured pass contains both formats")
    gaps = {i: pair[cases[1]]["time_to_last_batch_seconds"] - pair[cases[0]]["time_to_last_batch_seconds"]
            for i, pair in complete.items()}
    middle = median(gaps.values())
    chosen = min(gaps, key=lambda i: abs(gaps[i] - middle))

    traces = defaultdict(list)
    for row in load_trace(directory / "reads.csv"):
        if row["pass_number"] == chosen and row["case"] in cases:
            for key in ("started_ns", "finished_ns", "delivered_ns", "batch", "worker_pid", "worker_thread"):
                row[key] = int(row[key])
            traces[row["case"]].append(row)
    measured = [(title, complete[chosen][letter], traces[letter])
                for title, letter in zip(("NeXus/HDF5", "Sharded Zarr"), cases)]
    workers = environment["read_workers"]
    selection = "Acquisition-oriented chunks" if layout == "acquisition" else "Spatial bricks"
    reader_label = "reader" if workers == 1 else "readers"
    subtitle = f"{selection} · {len(traces[cases[0]])} identical tile reads · {workers} application {reader_label}"
    if environment.get("zarr_native_settings"):
        subtitle += " · native Zarr settings"
    filesystem = summary["mount"].get("fstype", "an unidentified filesystem")
    note = (f"Pass {chosen} of {len(complete)} on warmed {filesystem}, closest to the median format gap. "
            f"Both formats had the same {environment['available_cpus']}-CPU affinity mask.")
    output = output or directory / f"common-start-{'-'.join(cases)}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_timeline(output, subtitle, measured, note)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="experiment directory or matched read profile directory")
    parser.add_argument("--compare-formats", action="store_true", help="plot HDF5 and Zarr from a matched read profile")
    parser.add_argument("--case", choices=CASES, default="A")
    parser.add_argument("--output", type=Path, help="PNG path, defaults to the experiment directory")
    args = parser.parse_args()
    try:
        directory = args.directory.resolve()
        if args.compare_formats:
            print(draw_matched_profile(directory, args.case, args.output).resolve())
            return
        metadata = json.loads((directory / "experiment.json").read_text())
        records = {record["case"]: record for record in metadata["case_results"]}
        record = records[args.case]
        if not record["verified"]:
            raise ValueError(f"case {args.case} must be verified")
        raw = metadata["raw"]
        concurrency = raw.get("zarr_concurrency", zarr.config.get("async.concurrency"))
        zarr.config.set({"async.concurrency": concurrency})
        batches = list(request_batches(raw))
        saved = load_trace(directory / args.case / "raw-reads.csv")
        if list(itertools.chain.from_iterable(batches)) != trace_requests(saved):
            raise ValueError("recorded reads do not match the configured detector bands")

        digests = [read_case(directory, record, raw, batches, readers, verify=True)[0] for readers in (1, 2)]
        if digests[0] != digests[1]:
            raise ValueError("one and two readers returned different detector data")
        orders = ((1, 2), (2, 1), (1, 2))
        passes = {1: [], 2: []}
        with fresh_reader_pool() as pool:
            for order in orders:
                for readers in order:
                    trial = pool.submit(measure_case, directory, record, raw, batches, readers, concurrency).result()
                    passes[readers].append(trial)
        measured = [("One reader" if readers == 1 else "Two readers",
                     *sorted(trials, key=lambda trial: trial[0]["time_to_last_batch_seconds"])[len(trials) // 2])
                    for readers, trials in passes.items()]
        output = args.output or directory / f"read-timeline-{args.case}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        filesystem = mount_info(directory).get("fstype", "an unidentified filesystem")
        dataset = "NeXus/HDF5" if record["format"] == "nexus" else "Sharded Zarr"
        subtitle = f"Case {args.case} · {dataset} · {len(saved)} identical tile reads in {len(batches)} batches"
        note = (f"Median of {len(orders)} passes per setup on warmed {filesystem}. "
                "Every pass starts in a fresh application process.")
        draw_timeline(output, subtitle, measured, note)
        save_csv(output.with_suffix(".csv"), [
            dict(readers=readers, startup_started_ns=sample["startup_started_ns"],
                 read_started_ns=sample["read_started_ns"], **row)
            for readers, (_, sample, trace) in enumerate(measured, start=1) for row in trace])
        for readers, trials in passes.items():
            print(f"{readers} reader(s), setup to last batch in seconds: " +
                  ", ".join(f"{sample['time_to_last_batch_seconds']:.3f}" for sample, _ in trials))
        print(output.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
