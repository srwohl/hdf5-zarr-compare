"""Localize and measure the right-hand dark inclusion of the tiled phantom."""
import copy
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import ndimage
import zarr

import demo
from phantom import ELLIPSOIDS

LABEL = "dark-inclusion"
THRESHOLD = .085


def process_io():
    try:
        return {key: int(value) for key, value in
                (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}
    except OSError:
        return {}


def read_volume(array, selection, settings, level, pattern, trace):
    before = process_io()
    started = time.perf_counter()
    values = np.asarray(array[selection])
    elapsed = time.perf_counter() - started
    after = process_io()
    _, chunks, partitions = demo.geometry(settings, level)
    trace.append(dict(pattern=pattern, level=level, seconds=elapsed,
                      selection=[[s.start, s.stop] for s in selection],
                      returned_bytes=values.nbytes,
                      full_chunk_model_bytes=demo.touched(selection, chunks) * int(np.prod(chunks)) * 2,
                      parts_intersected=demo.touched(selection, partitions),
                      process_rchar_bytes=after["rchar"] - before["rchar"] if after and before else None,
                      process_read_bytes=after["read_bytes"] - before["read_bytes"] if after and before else None))
    return values


def locate(overview, step, size):
    specimen = overview > 0.12 * 65535
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(specimen.any(axis=tuple(a for a in range(3) if a != axis)))
        if not len(occupied):
            raise ValueError("no specimen found in the overview")
        bounds.append((int(occupied[0]), int(occupied[-1]) + 1))

    # An operator chooses the right-hand dark feature. Search the central
    # interior of the specimen's bounding box with that directional prior.
    lo, hi = np.array(bounds).T
    target = lo + (hi - lo - 1) * [.50, .44, .66]
    half_window = np.maximum(1, (hi - lo) * [.13, .14, .12])
    start = np.maximum(0, np.ceil(target - half_window).astype(int))
    stop = np.minimum(overview.shape, np.floor(target + half_window).astype(int) + 1)
    window = tuple(slice(a, b) for a, b in zip(start, stop))
    candidates = np.argwhere(overview[window] < THRESHOLD * 65535) + start
    seed = target if not len(candidates) else candidates[np.argmin(np.sum((candidates - target)**2, axis=1))]
    center = np.rint((seed + .5) * step - .5).astype(int)
    radius = np.ceil(size * np.array([.16, .24, .13])).astype(int)
    roi = tuple(slice(max(0, int(c - r)), min(size, int(c + r + 1))) for c, r in zip(center, radius))
    return bounds, center, roi


def segment(values, seed, roi):
    components, count = ndimage.label(values < THRESHOLD * 65535)
    if not count:
        return np.zeros(values.shape, dtype=bool)
    local = seed - np.array([s.start for s in roi])
    # Choose the connected dark region nearest the selected voxel.
    nearest = np.clip(local, 0, np.array(values.shape) - 1)
    component = components[tuple(nearest)]
    if not component:
        candidates = np.argwhere(components)
        nearest = candidates[np.argmin(np.sum((candidates - local)**2, axis=1))]
        component = components[tuple(nearest)]
    return components == component


def truth_geometry(size, roi):
    _, a, b, c, cx, cy, cz, angle = ELLIPSOIDS[2]
    theta = np.deg2rad(angle)
    xyz = (2 * (np.arange(size, dtype=float) + .5) / size - 1)
    extent_x = np.hypot(a * np.cos(theta), b * np.sin(theta))
    extent_y = np.hypot(a * np.sin(theta), b * np.cos(theta))
    ranges = [np.flatnonzero(np.abs(xyz - center) <= radius)
              for center, radius in ((cz, c), (cy, extent_y), (cx, extent_x))]
    geometry = tuple(slice(int(r[0]), int(r[-1]) + 1) for r in ranges)
    z, y, x = np.ix_(xyz[geometry[0]] - cz, xyz[geometry[1]] - cy, xyz[geometry[2]] - cx)
    xr, yr = x * np.cos(theta) + y * np.sin(theta), -x * np.sin(theta) + y * np.cos(theta)
    full = (xr / a)**2 + (yr / b)**2 + (z / c)**2 <= 1
    overlap = [(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(geometry, roi)]
    clipped = np.zeros(tuple(s.stop - s.start for s in roi), dtype=bool)
    if all(a < b for a, b in overlap):
        clipped[tuple(slice(a - s.start, b - s.start) for (a, b), s in zip(overlap, roi))] = full[
            tuple(slice(a - s.start, b - s.start) for (a, b), s in zip(overlap, geometry))]
    centroid = np.asarray(ndimage.center_of_mass(full)) + [s.start for s in geometry]
    return clipped, int(full.sum()), centroid


def accuracy(mask, size, roi, spacing):
    truth, truth_count, truth_center = truth_geometry(size, roi)
    found = int(mask.sum())
    intersection = int(np.count_nonzero(mask & truth))
    center = np.asarray(ndimage.center_of_mass(mask)) + [s.start for s in roi] if found else None
    return dict(dice=2 * intersection / (found + truth_count),
                volume_error_fraction=(found - truth_count) / truth_count,
                truth_coverage_fraction=float(truth.sum() / truth_count),
                truth_volume_um3=truth_count * spacing**3, segmented_volume_um3=found * spacing**3,
                centroid_error_um=float(np.linalg.norm(center - truth_center) * spacing) if found else None)


def label_levels(mask, selection, count):
    for level in range(count):
        yield selection, mask
        if level + 1 < count:
            # Align pooling with the full volume's origin before taking max.
            aligned = tuple(slice(s.start // 2, (s.stop + 1) // 2) for s in selection)
            pad = [(s.start % 2, s.stop % 2) for s in selection]
            z, y, x = (s.stop - s.start for s in aligned)
            mask = np.pad(mask, pad).reshape(z, 2, y, 2, x, 2).any(axis=(1, 3, 5))
            selection = aligned


def write_labels(directory, kind, settings, mask, roi):
    label_data = label_levels(mask, roi, settings["levels"])
    if kind == "zarr":
        root = zarr.open_group(str(directory / "phantom.ome.zarr"), mode="r+")
        collection = root.create_group("labels")
        group = collection.create_group(LABEL)
        for level, (selection, values) in enumerate(label_data):
            shape, chunks, shards = demo.geometry(settings, level)
            codecs = [zarr.codecs.GzipCodec(level=1)] if settings["compression"] == "gzip" else None
            array = group.create_array(str(level), shape=shape, chunks=chunks, shards=shards,
                                       dtype="u1", compressors=codecs, dimension_names=demo.AXES)
            array[selection] = values
        multiscales = copy.deepcopy(root.attrs["ome"]["multiscales"][0])
        multiscales.update(name=LABEL, type="max")
        group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscales],
                              "image-label": {"version": "0.5", "source": {"image": "../../"},
                                              "colors": [{"label-value": 1, "rgba": [255, 165, 0, 255]}]}}
        collection.attrs["ome"] = {"version": "0.5", "labels": [LABEL]}
        return directory / "phantom.ome.zarr" / "labels"
    else:
        with h5py.File(directory / "nexus/phantom.nxs", "r+") as handle:
            collection = demo.nxgroup(handle["entry"], "labels", "NXcollection")
            collection.attrs["source"] = "/entry/levels"
            collection.attrs["label_1"] = LABEL
            collection.attrs["pyramid_method"] = "max"
            levels = demo.nxgroup(collection, "levels", "NXcollection")
            for level, (selection, values) in enumerate(label_data):
                shape, chunks, _ = demo.geometry(settings, level)
                options = {"compression": "gzip", "compression_opts": 1} if settings["compression"] == "gzip" else {}
                dataset = collection.create_dataset("_mask", shape=shape, dtype="u1", chunks=chunks, **options)
                dataset[selection] = values
                demo.nxdata(levels, str(level), dataset, (0, 0, 0), settings["voxel_um"] * 2**level)
                del dataset.attrs["units"]
                dataset.attrs["description"] = "Dark inclusion (1) and background (0)"
                del collection["_mask"]
        return directory / "nexus/phantom.nxs"


def verify_volume(directory, kind, settings, mask, roi):
    with demo.open_levels(directory, kind, settings["levels"]) as arrays:
        for level, array in enumerate(arrays):
            source = np.load(Path(settings["source_directory"]) / f"{level}.npy", mmap_mode="r")
            shape, _, partitions = demo.geometry(settings, level)
            for selection in demo.regions(shape, partitions):
                np.testing.assert_array_equal(array[selection], source[selection])

    def check_label(get_array):
        for level, (selection, values) in enumerate(label_levels(mask, roi, settings["levels"])):
            array = get_array(level)
            assert array.shape == demo.geometry(settings, level)[0]
            np.testing.assert_array_equal(array[selection], values)

    if kind == "zarr":
        root = zarr.open_group(str(directory / "phantom.ome.zarr"), mode="r")
        check_label(lambda level: root[f"labels/{LABEL}/{level}"])
    else:
        with h5py.File(directory / "nexus/phantom.nxs", "r") as handle:
            check_label(lambda level: handle[f"entry/labels/levels/{level}/data"])
