"""Visualize recorded request geometry against chunk and partition boundaries."""
import argparse
import csv
import json
from math import ceil, prod
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import demo


REQUEST = "#ef9c32"
COVERAGE = "#a9dce5"
CHUNK = "#7696a3"
PARTITION = "#1b4058"


def read_raw_traces(directory, results):
    traces = {}
    for result in results:
        case = result["case"]
        path = directory / case / "raw-reads.csv"
        if path.is_file():
            with path.open(newline="") as source:
                traces[case] = list(csv.DictReader(source))
    return traces


def bounds(row):
    values = row["selection"]
    if isinstance(values, str):
        values = json.loads(values)
    return tuple(slice(lo, hi) for lo, hi in values)


def byte_label(count):
    if count >= demo.MIB:
        return f"{count / demo.MIB:.2f} MiB"
    if count >= 1024:
        return f"{count / 1024:.1f} KiB"
    return f"{count} B"


def add_box(axis, limits, **style):
    major, y, _ = limits
    axis.add_patch(Rectangle((major.start, y.start), major.stop - major.start, y.stop - y.start, **style))


def show_request(axis, selection, shape, chunks, partitions, major_limit, major_label):
    coverage = tuple(slice(s.start // c * c, min(n, (s.stop + c - 1) // c * c))
                     for s, c, n in zip(selection, chunks, shape))
    add_box(axis, coverage, facecolor=COVERAGE, edgecolor="none")
    for dimensions, color, linestyle, thickness in (
            (chunks, CHUNK, "-", .55), (partitions, PARTITION, "--", 1.25)):
        major_lines = range(dimensions[0], major_limit, dimensions[0])
        y_lines = range(dimensions[1], shape[1], dimensions[1])
        axis.vlines(major_lines[::max(1, ceil(len(major_lines) / 128))], 0, shape[1],
                    colors=color, linestyles=linestyle, linewidth=thickness)
        axis.hlines(y_lines[::max(1, ceil(len(y_lines) / 128))], 0, major_limit,
                    colors=color, linestyles=linestyle, linewidth=thickness)
    add_box(axis, selection, facecolor=REQUEST, edgecolor="#854012", linewidth=1.2, zorder=3)
    axis.set(xlim=(0, major_limit), ylim=(0, shape[1]), xlabel=major_label)
    axis.set_ylabel("Detector y" if major_label == "Angle index" else "Voxel y")
    axis.tick_params(labelsize=8, length=3)
    axis.locator_params(nbins=5)
    for spine in axis.spines.values():
        spine.set_color(CHUNK)

    requested = prod(s.stop - s.start for s in selection) * demo.DTYPE.itemsize
    touched = demo.touched(selection, chunks)
    parts = demo.touched(selection, partitions)
    modeled = touched * prod(chunks) * demo.DTYPE.itemsize
    chunk_label = "chunk" if touched == 1 else "chunks"
    part_label = "part" if parts == 1 else "parts"
    axis.text(.5, -.33, f"{touched:,} {chunk_label} · {parts:,} {part_label}\n"
             f"{byte_label(requested)} requested / {byte_label(modeled)} covered",
             transform=axis.transAxes, ha="center", va="top", fontsize=8.5, linespacing=1.4)


def plot_access_patterns(path, raw, results, traces):
    rows = []
    for layout in ("acquisition", "bricks"):
        cases = [r for r in results if r["layout"] == layout and r.get("reads") and traces.get(r["case"])]
        if not cases:
            continue
        example = cases[0]
        trace = traces[example["case"]]
        frame = next((row for row in trace if row["pattern"] == "detector_frame"), None)
        if frame is None:
            continue
        same_tile = [row for row in trace if row["tile"] == frame["tile"] and row["pattern"] == "sinogram_band"]
        if not same_tile:
            continue
        band = min(same_tile, key=lambda row: abs(bounds(row)[1].start + bounds(row)[1].stop - raw["tile"]))
        roi = next((row for row in example["reads"] if row["pattern"] == "roi"), None)
        if roi is not None:
            rows.append((layout, example, frame, band, roi))
    if not rows:
        return False

    cols = ("Detector frame", "Sinogram band", "Full-resolution ROI")
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(14.2, 2.9 * len(rows) + 1.15), squeeze=False)
    fig.subplots_adjust(left=.095, right=.985, bottom=.17 if len(rows) > 1 else .25, top=.80, wspace=.35, hspace=.78)
    fig.suptitle("Which chunks does a read touch?", x=.095, y=.98, ha="left", fontsize=20, fontweight="bold")
    fig.text(.095, .925 if len(rows) > 1 else .90,
             "Recorded detector and image requests projected along x. Counts and bytes use all three dimensions.",
             fontsize=10.5)
    labels = {"acquisition": "Acquisition-oriented\nCases A and C", "bricks": "Spatial bricks\nCases B and D"}
    if raw.get("detector_bricks") == "sinogram":
        labels["bricks"] = "Sinogram raw / image bricks\nCases B and D"

    for idx, (layout, example, frame, band, roi) in enumerate(rows):
        width = raw["tile"]
        raw_shape = (len(raw["angles_deg"]), width, width)
        raw_chunks = example["raw_chunks"]
        raw_parts = (raw["angle_block"], width, width)
        settings = dict(size=raw["size"], chunks=example["image_chunks"], shards=example["image_partitions"])
        shape, chunks, parts = demo.geometry(settings, roi["level"])
        panels = [(axes[idx, 0], frame, raw_shape, raw_chunks, raw_parts, raw_parts[0], "Angle index"),
                  (axes[idx, 1], band, raw_shape, raw_chunks, raw_parts, raw_shape[0], "Angle index"),
                  (axes[idx, 2], roi, shape, chunks, parts, shape[0], "Voxel z")]
        for col, (axis, read, shape, chunks, parts, limit, label) in enumerate(panels):
            selection = bounds(read)
            show_request(axis, selection, shape, chunks, parts, limit, label)
            if idx == 0:
                axis.set_title(cols[col], fontweight="bold", fontsize=11, pad=10)
        first = axes[idx, 0].get_position()
        fig.text(.009, (first.y0 + first.y1) / 2, labels[layout], ha="left", va="center",
                 fontsize=9, fontweight="bold", rotation=90)

    fig.legend(handles=[Patch(facecolor=REQUEST, edgecolor="#854012", label="Requested region"),
                        Patch(facecolor=COVERAGE, label="Intersected chunk coverage"),
                        Line2D([], [], color=CHUNK, label="Chunk edge"),
                        Line2D([], [], color=PARTITION, linestyle="--", label="File / shard boundary")],
               loc="center right", bbox_to_anchor=(.99, .895 if len(rows) > 1 else .855),
               frameon=False, ncol=4, fontsize=9)
    fig.text(.095, .021, "Detector frame panels show the first angle partition. The sinogram example uses one tile. "
             "Coverage models uncompressed chunks.", fontsize=9, color="#334c58")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="experiment run containing recorded reads")
    args = parser.parse_args()
    try:
        metadata = json.loads((args.directory / "experiment.json").read_text())
        results = sorted(metadata["case_results"], key=lambda row: row["case"])
        path = args.directory / "access-patterns.png"
        if not plot_access_patterns(path, metadata["raw"], results, read_raw_traces(args.directory, results)):
            raise ValueError("no case has a complete detector and image read trace")
        print(path.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
