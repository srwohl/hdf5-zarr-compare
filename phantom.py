"""A Shepp–Logan-style 3D phantom, evaluated at voxel centres.

Each row is amplitude, semi-axes xyz, centre xyz, rotation about z in degrees.
This is an explicit 3D ellipsoid variant, not an exact TomoPy reproduction.
The two large ellipsoids form a shell; smaller ellipsoids vary through z.
"""
from pathlib import Path

import numpy as np


ELLIPSOIDS = (
    (1.0, .6900, .920, .810, 0., 0., 0., 0.),
    (-.8, .6624, .874, .780, 0., -.0184, 0., 0.),
    (-.2, .1100, .310, .220, .22, 0., 0., -18.),
    (-.2, .1600, .410, .280, -.22, 0., 0., 18.),
    (.1, .2100, .250, .410, 0., .35, -.15, 0.),
    (.1, .0460, .046, .050, 0., .10, .25, 0.),
    (.1, .0460, .046, .050, 0., -.10, .25, 0.),
    (.1, .0460, .023, .050, -.08, -.605, 0., 0.),
    (.1, .0230, .023, .020, 0., -.606, 0., 0.),
    (.1, .0230, .046, .020, .06, -.605, 0., 0.),
)


def phantom_slab(size, start, stop, noise=0., seed=42):
    """Return uint16 [z,y,x] voxels for a range of z slices.

    Noise is in relative-intensity units before clipping/quantization. A separate
    seed per z slice makes results independent of generation slab size.
    """
    xy = (2 * (np.arange(size, dtype=np.float32) + .5) / size - 1)
    x, y = xy[None, :], xy[:, None]
    z = (2 * (np.arange(start, stop, dtype=np.float32) + .5) / size - 1)
    out = np.zeros((stop - start, size, size), dtype=np.float32)
    for amplitude, a, b, c, cx, cy, cz, angle in ELLIPSOIDS:
        theta = np.deg2rad(angle)
        xr = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
        yr = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)
        radius_xy = (xr / a) ** 2 + (yr / b) ** 2
        for i, coordinate in enumerate(z):
            out[i] += amplitude * (radius_xy + ((coordinate - cz) / c) ** 2 <= 1)
    if noise:
        for i, zi in enumerate(range(start, stop)):
            rng = np.random.default_rng(np.random.SeedSequence([seed, zi]))
            out[i] += rng.normal(0., noise, (size, size)).astype(np.float32)
    return np.rint(np.clip(out, 0., 1.) * 65535).astype("<u2")


def make_sources(directory: Path, size, levels, slab_z, noise, seed):
    """Generate disk-backed arrays; lower levels are actual 2×2×2 box means."""
    directory.mkdir()
    path = directory / "0.npy"
    volume = np.lib.format.open_memmap(path, mode="w+", dtype="<u2", shape=(size,) * 3)
    for start in range(0, size, slab_z):
        stop = min(start + slab_z, size)
        volume[start:stop] = phantom_slab(size, start, stop, noise, seed)
    volume.flush()
    del volume
    make_pyramid(directory, levels, slab_z)


def make_pyramid(directory: Path, levels, slab_z=8):
    """Build lower levels from an existing uint16 source/0.npy."""
    for level in range(1, levels):
        previous = np.load(directory / f"{level - 1}.npy", mmap_mode="r")
        n = previous.shape[0] // 2
        target = np.lib.format.open_memmap(
            directory / f"{level}.npy", mode="w+", dtype="<u2", shape=(n,) * 3)
        for start in range(0, n, slab_z):
            stop = min(start + slab_z, n)
            block = previous[2 * start:2 * stop]
            mean = block.reshape(stop - start, 2, n, 2, n, 2).mean(axis=(1, 3, 5))
            target[start:stop] = np.rint(mean).astype("<u2")
        target.flush()
        del previous, target

