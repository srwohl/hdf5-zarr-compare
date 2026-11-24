"""Replay simulated detector frames and publish complete raw partitions."""
from collections import Counter, deque
import json
import os
from queue import Full, Queue
from threading import Event, Thread
import time

import h5py
import numpy as np
import zarr

import demo
import tiled
from io_workers import pool_for, start_workers


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_payload(path, payload, sync):
    """Publish bytes from a private buffer. Return write and fsync wall times."""
    started = time.perf_counter()
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as out:
        out.write(payload)
        out.flush()
        written = time.perf_counter()
        if sync:
            os.fsync(out.fileno())
        flushed = time.perf_counter()
    temporary.replace(path)
    return written - started + time.perf_counter() - flushed, flushed - written if sync else 0.


class RawWriter:
    def __init__(self, directory, config, kind, raw_source):
        self.kind = kind
        block, width = config["angle_block"], config["tile"]
        self.shard = (block, width, width)
        self.chunks = tuple(config["raw_chunks"])
        self.sync = config["sync"]
        self.compression = config["compression"]
        self.publications = directory / "publications"
        self.publications.mkdir()
        if kind == "nexus":
            self.root_path = directory / "raw-nexus"
            for tile in range(len(config["tile_origins_yx_px"])):
                (self.root_path / f"tile_{tile}").mkdir(parents=True)
            tiled.create_raw_master(directory, config, raw_source)
        else:
            self.root_path = directory / "raw.zarr"
            root = zarr.open_group(str(self.root_path), mode="w", zarr_format=3)
            root.attrs.update(config)
            root.attrs["schema"] = "voxel-io-demo raw projection mosaic; axes include angle"
            for tile, origin in enumerate(config["tile_origins_yx_px"]):
                group = root.create_group(f"tile_{tile}")
                group.attrs["origin_yx_px"] = origin
                array = self.zarr_array(group, (len(config["angles_deg"]), width, width))
                array.attrs["units"] = "counts"
                self.key = array.metadata.encode_chunk_key
                for name in ("flat", "dark"):
                    data = np.load(raw_source / f"{name}_{tile}.npy")
                    group.create_array(name, data=data, compressors=None)
        self.setup_sync_seconds = demo.sync_tree(self.root_path) if self.sync else 0.
        if self.sync:
            sync_directory(directory)
            sync_directory(directory.parent)

    def zarr_array(self, group, shape):
        compressors = [zarr.codecs.GzipCodec(level=1)] if self.compression == "gzip" else None
        return group.create_array("data", shape=shape, dtype="<u2", chunks=self.chunks,
                                  shards=self.shard, compressors=compressors,
                                  dimension_names=("angle", "detector_y", "detector_x"),
                                  config={"write_empty_chunks": True})

    def pack(self, tile, start, values):
        if self.kind == "zarr":
            # The same sharding codec produces the payload in a memory store.
            # Only its payload is copied to the corresponding persistent key.
            memory = {}
            group = zarr.open_group(store=zarr.storage.MemoryStore(memory), mode="w", zarr_format=3)
            self.zarr_array(group, self.shard)[:] = values
            buffer = memory.get(f"data/{self.key((0, 0, 0))}")
            path = self.root_path / f"tile_{tile}/data" / self.key((start // self.shard[0], 0, 0))
            return path, None if buffer is None else buffer.as_buffer_like()
        path = self.root_path / f"tile_{tile}/part_{start}.nxs"
        options = {"compression": "gzip", "compression_opts": 1} if self.compression == "gzip" else {}
        with h5py.File(str(path), "w", driver="core", backing_store=False) as handle:
            handle.attrs["default"] = "entry"
            entry = demo.nxgroup(handle, "entry", "NXentry")
            entry.attrs["default"] = "data"
            data = demo.nxgroup(entry, "data", "NXdata")
            data.attrs["signal"] = "data"
            data.create_dataset("data", data=values, chunks=self.chunks, **options).attrs["units"] = "counts"
            handle.flush()
            return path, handle.id.get_file_image()

    def publish(self, start, frames, missing):
        encode_seconds = write_seconds = sync_seconds = payload_bytes = 0
        for tile, values in enumerate(frames):
            started = time.perf_counter()
            path, payload = self.pack(tile, start, values)
            encode_seconds += time.perf_counter() - started
            if payload is None:
                continue
            payload_bytes += len(payload)
            started = time.perf_counter()
            path.parent.mkdir(parents=True, exist_ok=True)
            write_seconds += time.perf_counter() - started
            written, synced = atomic_payload(path, payload, self.sync)
            write_seconds += written
            sync_seconds += synced
            if self.sync:
                started = time.perf_counter()
                for parent in path.parents:
                    sync_directory(parent)
                    if parent == self.root_path:
                        break
                sync_seconds += time.perf_counter() - started
        notice = dict(angle_start=start, angle_stop=start + self.shard[0], missing_frames=missing)
        written, synced = atomic_payload(self.publications / f"{start:08d}.json",
                                         json.dumps(notice).encode(), self.sync)
        published = time.perf_counter()
        write_seconds += written
        sync_seconds += synced
        if self.sync:
            sync_directory(self.publications)
            sync_seconds += time.perf_counter() - published
        return dict(**notice, published=published, encode_seconds=encode_seconds,
                    write_seconds=write_seconds, sync_seconds=sync_seconds, payload_bytes=payload_bytes)


def produce(sources, config, queue, stop, stats, timing):
    period = 1 / config["fps"]
    origin = deadline = time.perf_counter()
    stats["origin"] = origin
    for angle in range(len(sources[0])):
        for tile, source in enumerate(sources):
            if stop.is_set():
                return
            if stop.wait(max(0., deadline - time.perf_counter())):
                return
            index = angle * len(sources) + tile
            received = time.perf_counter()
            timing[index][0] = received - origin
            timing[index][1] = max(0., received - deadline)
            frame = source[angle].copy()
            stats["source_copy_seconds"] += time.perf_counter() - received
            entry = (index, frame)
            try:
                queue.put_nowait(entry)
            except Full:
                stats["queue_full_events"] += 1
                if config["overflow"] == "drop":
                    timing[index][2] = -1.
                    deadline += period
                    continue
                waiting = time.perf_counter()
                while not stop.is_set():
                    try:
                        queue.put(entry, timeout=.1)
                        break
                    except Full:
                        continue
                blocked = time.perf_counter() - waiting
                timing[index][2] = blocked
                # A flow-controlled source pauses its clock for the blocked interval.
                deadline += blocked
            occupancy = queue.qsize()
            stats["occupancy"][occupancy] += 1
            deadline += period


def gb_per_second(byte_count, seconds):
    return byte_count / (seconds * 1_000_000_000)


def replay_throughput(stats, frame_bytes):
    published_bytes = stats["published_frames"] * frame_bytes
    replay_seconds = stats["replay_seconds"]
    return dict(frame_bytes=frame_bytes, published_bytes=published_bytes,
                received_gb_s=stats["sustained_received_fps"] * frame_bytes / 1_000_000_000,
                published_gb_s=gb_per_second(published_bytes, replay_seconds),
                payload_gb_s=gb_per_second(stats["payload_bytes"], replay_seconds))


def run_replay(writer, sources, config):
    """Bound memory by queue capacity plus one assembly and a block per worker."""
    count = len(sources[0]) * len(sources)
    per_block = config["angle_block"] * len(sources)
    timing = np.zeros((count, 4), dtype=float)
    queue = Queue(config["queue_frames"])
    stop = Event()
    stats = dict(occupancy=Counter(), queue_full_events=0, source_copy_seconds=0.)
    failures = []

    def send():
        try:
            produce(sources, config, queue, stop, stats, timing)
            stats["source_finished"] = time.perf_counter()
        except BaseException as error:
            failures.append(error)
        finally:
            while not stop.is_set():
                try:
                    queue.put(None, timeout=.1)
                    break
                except Full:
                    continue

    producer = Thread(target=send)
    partitions = []
    pending = deque()
    item = None
    end_of_stream = False
    try:
        workers = min(config["workers"], count // per_block)
        started = time.perf_counter()
        with pool_for(writer.kind, workers) as pool:
            if workers > 1:
                start_workers(pool, workers)
            stats["worker_setup_seconds"] = time.perf_counter() - started
            stats["write_workers"] = workers
            producer.start()
            for lo in range(0, count, per_block):
                blocks = np.zeros((len(sources), *writer.shard), dtype="<u2")
                present = np.zeros(per_block, dtype=bool)
                while not end_of_stream:
                    if item is None:
                        item = queue.get()
                    if item is None:
                        end_of_stream = True
                        break
                    index, frame = item
                    if index >= lo + per_block:
                        break
                    offset = index - lo
                    angle, tile = divmod(offset, len(sources))
                    blocks[tile, angle] = frame
                    present[offset] = True
                    item = None
                    if offset == per_block - 1:
                        break
                if failures:
                    raise failures[0]
                start = lo // len(sources)
                missing = (np.flatnonzero(~present) + lo).tolist()
                pending.append(pool.submit(writer.publish, start, blocks, missing))
                if len(pending) >= workers:
                    partitions.append(pending.popleft().result())
            partitions.extend(task.result() for task in pending)
    finally:
        stop.set()
        if producer.ident is not None:
            producer.join()
    if failures:
        raise failures[0]
    last_published = max(p["published"] for p in partitions)
    for partition in partitions:
        lo = partition["angle_start"] * len(sources)
        hi = lo + per_block
        partition["published_seconds"] = partition.pop("published") - stats["origin"]
        timing[lo:hi, 3] = partition["published_seconds"]
    missing = [index for partition in partitions for index in partition["missing_frames"]]
    kept = timing[:, 2] >= 0
    latency = timing[kept, 3] - timing[kept, 0]
    stats.update(expected_frames=count, received_frames=count, dropped_frames=int(np.count_nonzero(~kept)),
                 missing_frames=len(missing), published_frames=len(latency),
                 source_seconds=stats["source_finished"] - stats["origin"],
                 replay_seconds=last_published - stats["origin"],
                 blocked_seconds=float(timing[kept, 2].sum()),
                 peak_queue_frames=queue.maxsize if stats["queue_full_events"] else max(stats["occupancy"], default=0),
                 receive_jitter_p95_ms=float(np.percentile(timing[:, 1], 95) * 1000))
    stats["sustained_received_fps"] = (count - 1) / (timing[-1, 0] - timing[0, 0])
    stats["sustained_published_fps"] = len(latency) / stats["replay_seconds"]
    for key in ("encode_seconds", "write_seconds", "sync_seconds", "payload_bytes"):
        stats[key] = sum(p[key] for p in partitions)
    stats.update(replay_throughput(stats, sources[0][0].nbytes))
    stats["setup_sync_seconds"] = writer.setup_sync_seconds
    stats["publish_p50_ms"] = float(np.percentile(latency, 50) * 1000) if len(latency) else None
    stats["publish_p95_ms"] = float(np.percentile(latency, 95) * 1000) if len(latency) else None
    stats["publish_max_ms"] = float(latency.max() * 1000) if len(latency) else None
    origin = stats.pop("origin")
    del stats["source_finished"]
    timing[~kept, 3] = np.nan
    return stats, partitions, timing, origin
