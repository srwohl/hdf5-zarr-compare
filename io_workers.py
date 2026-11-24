"""Bounded independent format workers."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice
import multiprocessing
import os


def pool_for(kind, workers, **options):
    if kind == "nexus" and workers > 1:
        # h5py serializes HDF5 calls within a process. Spawn independent handles.
        return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), **options)
    return ThreadPoolExecutor(max_workers=workers, **options)


def start_workers(pool, workers):
    ready = [pool.submit(os.getpid) for _ in range(workers)]
    for task in ready:
        task.result()


def ordered_batches(pool, function, items, limit):
    items = iter(items)
    pending = deque(pool.submit(function, item) for item in islice(items, limit))
    for item in items:
        result = pending.popleft().result()
        pending.append(pool.submit(function, item))
        yield result
    while pending:
        yield pending.popleft().result()
