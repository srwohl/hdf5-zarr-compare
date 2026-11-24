"""Render the comparison and registered image/label previews."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from access_patterns import plot_access_patterns, read_raw_traces
from replay import replay_throughput

MIB = 1024**2


def context_plot(path, overview, step, bounds, values, mask, roi, seed, accuracy):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    z, y, x = seed
    _, ys, xs = roi
    size = overview.shape[0] * step
    style = dict(cmap="gray", origin="lower", vmin=0, vmax=.3)
    full = (0, size, 0, size)
    crop = (xs.start, xs.stop, ys.start, ys.stop)
    axes[0].imshow(overview[z // step] / 65535, extent=full, **style)
    (_, (y0, y1), (x0, x1)) = bounds
    axes[0].add_patch(Rectangle((x0 * step, y0 * step), (x1 - x0) * step, (y1 - y0) * step,
                               fill=False, edgecolor="tab:blue", label="Specimen bounds"))
    axes[0].add_patch(Rectangle((xs.start, ys.start), xs.stop-xs.start, ys.stop-ys.start,
                               fill=False, edgecolor="orange", label="Selected ROI"))
    axes[0].plot(x + .5, y + .5, "+", color="orange")
    axes[0].legend(loc="lower left", fontsize="small")
    axes[0].set_title(f"Overview · {step} µm voxels")
    index = z - roi[0].start
    for ax in axes[1:]:
        ax.imshow(values[index] / 65535, extent=crop, **style)
    overlay = np.ma.masked_where(~mask[index], mask[index])
    axes[2].imshow(overlay, extent=crop, origin="lower", cmap="autumn_r", vmin=0, vmax=2, alpha=.60)
    axes[1].set_title("Full-resolution ROI")
    axes[2].set_title(f"Segmented feature · Dice {accuracy['dice']:.5f}")
    for ax in axes:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison_plot(path, results):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].set_title("Elapsed from first frame")
    axes[0].set_ylabel("Seconds")
    axes[1].set_title("Raw publication latency")
    axes[1].set_ylabel("P95 (ms)")
    axes[2].set_title("Overview + ROI reads")
    axes[2].set_ylabel("MiB")
    x = np.arange(len(results))
    grouped = (
        (("Overview", [r.get("first_overview_seconds", np.nan) for r in results]),
         ("Analysed ROI", [r.get("analysed_roi_seconds", np.nan) for r in results]),
         ("Image + labels", [r.get("labels_ready_seconds", np.nan) for r in results])),
        (("P95", [r["replay"]["publish_p95_ms"] for r in results]),),
        (("Returned", [sum(t["returned_bytes"] for t in r["reads"]) / MIB if "reads" in r else np.nan
                        for r in results]),
         ("Chunk coverage", [sum(t["full_chunk_model_bytes"] for t in r["reads"]) / MIB
                             if "reads" in r else np.nan for r in results])),
    )
    for ax, series in zip(axes, grouped):
        width = .8 / len(series)
        for i, (label, data) in enumerate(series):
            ax.bar(x + (i - (len(series) - 1) / 2) * width,
                   [np.nan if value is None else value for value in data], width=width, label=label)
        ax.set_xticks(x, [r["case"] for r in results])
        ax.grid(axis="y", alpha=.25)
        if len(series) > 1:
            ax.legend(fontsize="small")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def make_report(directory, metadata, timeline):
    results = sorted(metadata["case_results"], key=lambda r: r["case"])
    comparison_plot(directory / "comparison.png", results)
    mount = metadata["mount"]
    raw = metadata["raw"]
    matching = metadata["identical_reconstruction_and_roi"]
    verdict = "All four cases passed" if metadata["verified"] else "Verification failed or cases were incomplete"
    match_text = "yes" if matching else ("no" if matching is False else "incomplete")
    lines = [
        "# Beamline replay and ROI comparison", "",
        f"**{verdict}.** Identical reconstructed voxels and ROI across all cases: **{match_text}**.", "",
        f"Profile **{metadata['profile']}**, host **{metadata['host']}**, output mount "
        f"**{mount.get('fstype', 'unknown')}** at `{mount.get('target', '?')}`. "
        "Cache state was uncontrolled. Read timings can include cache hits.", "",
        f"{raw['size']}³ voxels at 1 µm spacing, {len(raw['angles_deg'])} angles, "
        f"{len(raw['tile_origins_yx_px'])} tiles per angle, {raw['fps']:g} tile frames/s requested, "
        f"{raw['queue_frames']} queue slots, {raw['angle_block']} angles per raw part, "
        f"{raw.get('read_workers', 1)} detector reader(s), {raw['workers']} writer(s), "
        f"Zarr async concurrency {raw.get('zarr_concurrency', 'library default')}, {raw['compression']} compression, "
        f"fsync {'enabled' if raw['sync'] else 'disabled'}. Case order was {metadata['case_order']}.", "",
        "![Elapsed time, publication latency and read cost by case](comparison.png)", "",
        "## Acquisition", "",
    ]
    if raw.get("detector_bricks") == "sinogram":
        lines += ["Cases B and D use sinogram-oriented raw detector chunks and spatial image bricks.", ""]
    lines += [
        "| Case | Format | Layout | Writers | Writer setup s | Received fps | Published fps | Dropped / missing | "
        "Peak queue | Blocked s | Encode + pack s | Write s | Fsync s | Publish P95 ms | Raw MiB / files |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        stats = result["replay"]
        storage = result["raw_storage"]
        lines.append(f"| {result['case']} | {result['format']} | {result['layout']} | "
                     f"{stats.get('write_workers', raw['workers'])} | {number(stats.get('worker_setup_seconds'), 3)} | "
                     f"{stats['sustained_received_fps']:.1f} | {stats['sustained_published_fps']:.1f} | "
                     f"{stats['dropped_frames']} / {stats['missing_frames']} | {stats['peak_queue_frames']} | "
                     f"{stats['blocked_seconds']:.3f} | {stats['encode_seconds']:.3f} | "
                     f"{stats['write_seconds']:.3f} | {stats['sync_seconds']:.3f} | "
                     f"{number(stats['publish_p95_ms'], 1)} | {storage['logical_bytes']/MIB:.2f} / {storage['files']} |")
    lines += ["", "Raw replay throughput uses decimal GB/s. Useful bytes count uncompressed published detector "
              "frames. Payload bytes count the HDF5 part files or Zarr shards handed to the filesystem.", "",
              "| Case | Received GB/s | Published useful GB/s | Partition payload GB/s |",
              "|---|---:|---:|---:|"]
    for result in results:
        rates = replay_throughput(result["replay"], raw["tile"] ** 2 * 2)
        lines.append(f"| {result['case']} | {rates['received_gb_s']:.3f} | {rates['published_gb_s']:.3f} | "
                     f"{rates['payload_gb_s']:.3f} |")
    lines += ["", "Pack, write and fsync are sums of raw partition task times. "
              "Parallel HDF5 workers use separate processes and independent file handles. "
              "Zarr raw workers use threads, with internal chunk concurrency reported above. "
              "Pack includes in-memory file or shard "
              "construction and compression. The publication signal is the appearance of a per-angle-block JSON marker "
              "after every tile payload is readable. Consumers should use those markers and reopen the raw reader as "
              "needed. Missing frames leave placeholders and cause reconstruction to be skipped for that case.", "",
              "## Scientific result", "",
              "The overview locates the specimen and seeds the right-hand dark inclusion. A connected component at "
              "full resolution is compared to the analytic ellipsoid. The ROI overlay uses specimen coordinates.", "",
              "| Case | Overview s | Analysed ROI s | Labels ready s | Verified s | Dice | Volume error | Truth in ROI | "
              "Centroid error (µm) | Image + labels MiB / files | Result |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        a = r.get("accuracy", {})
        storage = r.get("image_and_label_storage", {})
        link = f"[{r['status']}]({r['case']}/result.png)" if a else r["status"]
        mounted = f"{storage['logical_bytes']/MIB:.2f} / {storage['files']}" if storage else "—"
        lines.append(f"| {r['case']} | {number(r.get('first_overview_seconds'))} | "
                     f"{number(r.get('analysed_roi_seconds'))} | {number(r.get('labels_ready_seconds'))} | "
                     f"{number(r.get('verified_seconds'))} | {number(a.get('dice'), 5)} | "
                     f"{number(a['volume_error_fraction'] * 100, 3) + '%' if a else '—'} | "
                     f"{number(a['truth_coverage_fraction'] * 100, 1) + '%' if a else '—'} | "
                     f"{number(a.get('centroid_error_um'))} | {mounted} | {link} |")
    gate = metadata["accuracy_gate"]
    lines += ["", f"Times start with the first detector frame. Verification follows label publication. "
              f"The accuracy gate requires Dice ≥ {gate['min_dice']:.2f}, absolute volume error ≤ "
              f"{gate['max_volume_error']:.0%} and ROI coverage ≥ {gate['min_truth_coverage']:.0%}.", "",
              "## Read cost", "",
              "| Case | Request | Time (ms) | Returned MiB | Full-chunk MiB | Process read-call MiB | "
              "Process storage MiB | Parts touched |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        for row in r.get("reads", []):
            process = [None if row[k] is None else row[k] / MIB for k in ("process_rchar_bytes", "process_read_bytes")]
            lines.append(f"| {r['case']} | {row['pattern']} | {row['seconds']*1000:.2f} | "
                         f"{row['returned_bytes']/MIB:.3f} | {row['full_chunk_model_bytes']/MIB:.3f} | "
                         f"{number(process[0], 3)} | {number(process[1], 3)} | {row['parts_intersected']} |")
    lines += ["", "Full-chunk MiB models uncompressed coverage. Actual decoder work depends on codecs, range reads and "
              "caching. Linux process counters include cache-served read calls and sampling overhead. Storage traffic "
              "on GPFS must be checked against filesystem telemetry.", ""]
    traces = read_raw_traces(directory, results)
    if plot_access_patterns(directory / "access-patterns.png", raw, results, traces):
        lines += ["![Chunk geometry for recorded reads](access-patterns.png)", ""]
    raw_reads = [(r["case"], traces[r["case"]], r.get("detector_reads", {}))
                 for r in results if traces.get(r["case"])]
    if raw_reads:
        lines += ["Raw reconstruction issues the same detector-frame and sinogram-band reads in each case.",
                  "Read-call times can overlap. Foreground wait is the time fetching data on the reconstruction "
                  "thread, excluding reader setup. Chunk coverage is summed per request.", "",
                  "| Case | Readers | Reader setup s | Foreground wait s | Raw read calls | Read-call s sum | "
                  "Returned MiB | Full-chunk MiB |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for case, rows, metrics in raw_reads:
            seconds = sum(float(row["elapsed_ms"]) for row in rows) / 1000
            returned = sum(int(row["returned_bytes"]) for row in rows) / MIB
            coverage = sum(int(row["full_chunk_model_bytes"]) for row in rows) / MIB
            lines.append(f"| {case} | {metrics.get('workers', 1)} | {number(metrics.get('setup_seconds'), 3)} | "
                         f"{number(metrics.get('foreground_wait_seconds'), 3)} | {len(rows)} | {seconds:.3f} | "
                         f"{returned:.2f} | {coverage:.2f} |")
        lines.append("")
    probes = [r for r in results if "deflate_probe" in r]
    if probes:
        lines += ["The in-memory DEFLATE reference uses the full-resolution chunk containing the ROI seed.",
                  "It isolates codec work from filesystem reads and application library overhead.", "",
                  "| Case | Encoded MiB | Decoded MiB | Encode ms | Decode ms |",
                  "|---|---:|---:|---:|---:|"]
        for r in probes:
            probe = r["deflate_probe"]
            lines.append(f"| {r['case']} | {probe['encoded_bytes']/MIB:.3f} | "
                         f"{probe['decoded_bytes']/MIB:.3f} | {probe['encode_seconds']*1000:.2f} | "
                         f"{probe['decode_seconds']*1000:.2f} |")
        lines.append("")
    if timeline and "cpu_user_seconds" in timeline[0]:
        lines += ["## Application phase cost", "",
                  "| Case | Phase | Wall s | User CPU s | System CPU s |",
                  "|---|---|---:|---:|---:|"]
        for row in timeline:
            lines.append(f"| {row['case']} | {row['phase']} | {row['seconds']:.3f} | "
                         f"{row['cpu_user_seconds']:.3f} | {row['cpu_system_seconds']:.3f} |")
        lines.append("")
    lines += ["## Open the result elsewhere", "",
              "- **OME-Zarr.** Open the image group with napari and enable the dark-inclusion label layer. "
              "The reader discovers both pyramids from OME-Zarr metadata. "
              "[napari-ome-zarr documentation](https://github.com/ome/napari-ome-zarr#usage)", "",
              "  ```bash", "  napari --plugin napari-ome-zarr C/phantom.ome.zarr", "  ```", "",
              "- **NeXus/HDF5.** Keep the master and its relative part paths together. An external viewer adapter "
              "needs to open the VDS, map the NXdata levels and coordinate axes, and load the label pyramid as a "
              "separate layer. Image and label paths are `/entry/levels/{level}/data` and "
              "`/entry/labels/levels/{level}/data` in `A/nexus/phantom.nxs`.", "",
              "## Filesystem correlation", "",
              "The [phase timeline](timeline.json) records UTC bounds for acquisition, reconstruction, export, analysis "
              "and verification. Snapshot execution times and failures are recorded alongside the samples. "
              "The [run manifest](experiment.json) includes actual chunk shapes, publication latency "
              "percentiles, queue observations, allocated bytes and software versions.", ""]
    for r in results:
        if "viewer" in r:
            viewer = r["viewer"]
            lines += [f"Case {r['case']} external reader: **{'passed' if viewer['passed'] else 'failed'}** "
                      f"({number(viewer.get('seconds'))} s including process startup).", ""]
    samples = [row for row in timeline if row["before"] or row["after"]]
    deltas = [row for row in samples if row["counter_delta"]]
    if deltas:
        lines += ["| Case | Phase | Wall s | Counter deltas |", "|---|---|---:|---|"]
        for row in deltas:
            delta = json.dumps(row["counter_delta"]).replace("|", "\\|")
            lines.append(f"| {row['case']} | {row['phase']} | {row['seconds']:.3f} | `{delta}` |")
        lines.append("")
    errors = sum("error" in row[side] for row in samples for side in ("before", "after"))
    if errors:
        lines += [f"**{errors} counter samples failed.** Details are in the timeline.", ""]
    if not deltas:
        lines += ["No GPFS counter deltas were collected for this run.", ""]
    lines += ["Per-case `frames.csv`, `publications.json`, `raw-reads.csv` and `analysis-reads.csv` "
              "provide the underlying measurements. Cases with missing frames have acquisition traces only.", ""]
    (directory / "report.md").write_text("\n".join(lines))
