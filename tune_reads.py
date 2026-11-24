"""Test larger sinogram reads and a detector chunk layout with the same byte budget."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np

from demo import MIB
from diagnose_reads import copy_raw
from experiment import mount_info, save_csv, save_json
from profile_raw_reads import fresh_reader_pool, load_trace, measure_case, trace_requests
from tiled import raw_reader, request_batches, sinogram_chunks, stitched_band


def sinogram_digest(directory, record, raw):
    digest = hashlib.sha256()
    with raw_reader(directory / record["case"], raw, record["format"], []) as (read, calibration):
        for first in range(0, raw["size"], raw["row_band"]):
            stop = min(raw["size"], first + raw["row_band"])
            values = stitched_band(read, calibration, raw, first, stop)
            digest.update(memoryview(np.ascontiguousarray(values.swapaxes(0, 1))).cast("B"))
    return digest.hexdigest()


def report(source, output, raw, alternate_band, samples):
    labels = {"acquisition": "Whole frames", "bricks": "Spatial bricks", "sinogram": "Sinogram-oriented"}
    lines = ["# Detector read tuning", "",
             "The same detector data, codec and physical partition shapes were used for every case. "
             "The sinogram-oriented chunks keep the same uncompressed bytes per chunk. "
             "Every case produced exactly the same stitched sinograms.", "",
             f"Original mount **{mount_info(source).get('fstype', 'unknown')}** and temporary copy mount "
             f"**{mount_info(output).get('fstype', 'unknown')}** with warmed caches. "
             "One application reader per format, native Zarr concurrency, and a fresh process per pass. "
             "Times are medians of three passes and include reader setup. "
             "Each first-band time refers to the row count in that row.", "",
             "| Format | Detector layout | Chunk shape (angles × y × x) | Rows per band | Read calls | "
             "Modeled chunk MiB | First band s | All data s |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    summary = []
    for layout in labels:
        for band in (raw["row_band"], alternate_band):
            for kind in ("nexus", "zarr"):
                group = [r for r in samples if (r["layout"], r["row_band"], r["format"]) == (layout, band, kind)]
                medians = {key: median(row[key] for row in group)
                           for key in ("time_to_first_band_seconds", "time_to_last_batch_seconds",
                                       "full_chunk_model_bytes", "returned_bytes")}
                first = group[0]
                summary.append(dict(format=kind, layout=layout, row_band=band, raw_chunks=first["raw_chunks"],
                                    request_count=first["request_count"], **medians))
                fmt = "HDF5" if kind == "nexus" else "Zarr"
                lines.append(f"| {fmt} | {labels[layout]} | {' × '.join(map(str, first['raw_chunks']))} | "
                             f"{band} | {first['request_count']} | {medians['full_chunk_model_bytes']/MIB:.2f} | "
                             f"{medians['time_to_first_band_seconds']:.3f} | {medians['time_to_last_batch_seconds']:.3f} |")
    lines += ["", "The wider reads return the same total bytes per pass and reduce repeated intersections with chunks. "
              "They use larger buffers. Modeled chunk coverage counts full chunks per request before caching, "
              "compression and file range coalescing.", "", "[Timed passes](samples.csv)", ""]
    (output / "report.md").write_text("\n".join(lines))
    save_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="verified experiment")
    parser.add_argument("output", type=Path, help="new tuning report directory")
    parser.add_argument("--row-band", type=int, default=64, help="larger row band to test")
    args = parser.parse_args()
    try:
        source = args.directory.resolve()
        metadata = json.loads((source / "experiment.json").read_text())
        raw = metadata["raw"]
        if not metadata["verified"]:
            raise ValueError("expected a verified four-case experiment")
        if not raw["row_band"] < args.row_band <= raw["size"]:
            raise ValueError("choose a row band larger than the baseline and no larger than the detector mosaic")
        records = {row["case"]: row for row in metadata["case_results"]}
        plan = [request for batch in request_batches(raw) for request in batch]
        if any(trace_requests(load_trace(source / case / "raw-reads.csv")) != plan for case in records):
            raise ValueError("cases must contain the same recorded read plan")
        frame_chunk = records["A"]["raw_chunks"][0]
        width = raw["tile"]
        layout = "sinogram"
        chunks = sinogram_chunks(width, raw["angle_block"], frame_chunk)
        if raw.get("detector_bricks") == "sinogram":
            layout = "bricks"
            chunks = (4 * frame_chunk, width // 2, width // 2)
            for case in "BD":
                records[case]["layout"] = "sinogram"
        alternative = {case: dict(records[case], layout=layout, raw_chunks=chunks) for case in "BD"}
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        samples = []
        with TemporaryDirectory(prefix="detector-chunks-", dir=output) as folder:
            stage = Path(folder)
            print(f"Packing {layout} chunks {chunks} from the verified detector data", flush=True)
            copy_raw(source, stage, raw, alternative, compression=raw["compression"])
            variants = [(source, records[case]) for case in "ABCD"] + [(stage, alternative[case]) for case in "BD"]
            configurations = [(root, record, dict(raw, row_band=band))
                              for band in (raw["row_band"], args.row_band) for root, record in variants]
            print("Checking identical stitched sinograms before timing the reads", flush=True)
            digests = {sinogram_digest(root, record, config) for root, record, config in configurations}
            if len(digests) != 1:
                raise ValueError("tuned read plans produce different stitched sinograms")
            offset = len(variants)
            orders = (configurations, configurations[::-1], configurations[offset:] + configurations[:offset])
            with fresh_reader_pool() as pool:
                for pass_number, order in enumerate(orders, 1):
                    for root, record, config in order:
                        batches = list(request_batches(config))
                        sample, trace = pool.submit(measure_case, root, record, config, batches, 1, None).result()
                        samples.append(dict(pass_number=pass_number, format=record["format"], layout=record["layout"],
                                            raw_chunks=record["raw_chunks"], row_band=config["row_band"],
                                            request_count=len(trace), **sample))
                    print(f"Completed read pass {pass_number}/{len(orders)}", flush=True)
        save_csv(output / "samples.csv", samples)
        report(source, output, raw, args.row_band, samples)
        print(output / "report.md")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
