"""Regression checks for incremental raw publication and registered labels."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import feature
import replay
import tiled


class ExperimentTests(unittest.TestCase):
    def test_published_block_can_be_read_while_later_block_is_absent(self):
        config = dict(size=4, tile=4, angle_block=4, angles_deg=list(range(8)),
                      tile_origins_yx_px=[(0, 0)], raw_chunks=(1, 4, 4),
                      compression="gzip", sync=True)
        frames = np.arange(64, dtype="<u2").reshape(4, 4, 4)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            np.save(source / "flat_0.npy", np.full((4, 4), 25, dtype=np.float32))
            np.save(source / "dark_0.npy", np.zeros((4, 4), dtype=np.float32))
            for kind, path in (("nexus", "memory"), ("nexus", "file"), ("zarr", None)):
                with self.subTest(kind=kind, path=path):
                    output = root / (f"{kind}-{path}" if path else kind)
                    output.mkdir()
                    settings = dict(config, hdf5_write_path=path) if path else config
                    writer = replay.RawWriter(output, settings, kind, source)
                    writer.publish(0, [frames], [])
                    published = output / "publications"
                    self.assertEqual(json.loads((published / "00000000.json").read_text()),
                                     dict(angle_start=0, angle_stop=4, missing_frames=[]))
                    self.assertFalse((published / "00000004.json").exists())
                    self.assertFalse(list(output.rglob("*.partial")))
                    with tiled.raw_reader(output, config, kind, []) as (read, _):
                        selection = (slice(0, 4),) * 3
                        np.testing.assert_array_equal(read(0, selection, "published"), frames)

    def test_label_pyramid_uses_the_image_origin(self):
        roi = (slice(1, 6), slice(2, 5), slice(3, 8))
        rng = np.random.default_rng(42)
        mask = rng.integers(0, 2, size=(5, 3, 5)).astype(bool)
        full = np.zeros((8, 8, 8), dtype=bool)
        full[roi] = mask
        for selection, values in feature.label_levels(mask, roi, 3):
            level = np.zeros_like(full)
            level[selection] = values
            np.testing.assert_array_equal(level, full)
            n = full.shape[0] // 2
            if n:
                full = full.reshape(n, 2, n, 2, n, 2).any(axis=(1, 3, 5))
