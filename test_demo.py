"""Small correctness checks for voxel geometry, pyramids and portable reads."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import zarr

from phantom import make_sources, phantom_slab


HERE = Path(__file__).resolve().parent


class PhantomTests(unittest.TestCase):
    def test_three_dimensions_and_block_independent_noise(self):
        whole = phantom_slab(64, 0, 64, noise=.002)
        pieces = np.concatenate([phantom_slab(64, a, a + 8, noise=.002) for a in range(0, 64, 8)])
        np.testing.assert_array_equal(whole, pieces)
        clean = phantom_slab(64, 0, 64)
        self.assertGreater(int(clean.max()), 0)
        self.assertEqual(int(clean[0].max()), 0)
        self.assertFalse(np.array_equal(clean[16], clean[32]))
        self.assertFalse(np.array_equal(clean[:, 16, :], clean[:, 32, :]))
        self.assertFalse(np.array_equal(clean[:, :, 16], clean[:, :, 32]))

    def test_pyramid_averages_the_original_voxels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            make_sources(root, 32, 2, 3, 0., 42)
            fine = np.load(root / "0.npy")
            coarse = np.load(root / "1.npy")
            expected = np.rint(fine.reshape(16, 2, 16, 2, 16, 2).mean(axis=(1, 3, 5))).astype("<u2")
            np.testing.assert_array_equal(coarse, expected)

    def test_both_formats_after_move_and_parallel_compressed_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "original"
            def run(*args):
                subprocess.run([sys.executable, str(HERE / "demo.py"), *map(str, args)],
                               check=True, capture_output=True, text=True, cwd=tmp)
            run("build", original, "--size", 32, "--chunks", "4,8,8", "--shards", "16,16,16",
                "--levels", 2, "--workers", 2, "--compression", "gzip", "--sync", "--noise", .001)
            moved = Path(tmp) / "moved"
            shutil.move(original, moved)
            run("check", moved)
            run("bench", moved, "--reads", 2)
            with h5py.File(moved / "nexus/phantom.nxs", "r") as f:
                self.assertEqual(f["entry/data"].attrs["NX_class"], "NXdata")
                self.assertTrue(f["entry/data/data"].is_virtual)
                np.testing.assert_allclose(f["entry/levels/1/z"][:3], [1, 3, 5])
            root = zarr.open_group(str(moved / "phantom.ome.zarr"), mode="r")
            self.assertEqual(root["0"].shards, (16, 16, 16))
            self.assertEqual(root["0"].chunks, (4, 8, 8))
            transforms = root.attrs["ome"]["multiscales"][0]["datasets"][1]["coordinateTransformations"]
            self.assertEqual(transforms[1]["translation"], [1., 1., 1.])


if __name__ == "__main__":
    unittest.main()
