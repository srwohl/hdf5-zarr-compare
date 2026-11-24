"""Check detector coverage and reconstruction from genuinely stored tile data."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from skimage.transform import iradon

import demo
from tiled import feather, raw_reader, tile_starts

HERE = Path(__file__).resolve().parent


class TiledTests(unittest.TestCase):
    def test_feathered_grid_covers_detector_including_snapped_edge(self):
        n, width, overlap = 100, 40, 8
        starts = tile_starts(n,width,overlap)
        self.assertEqual(starts,[0,32,60])
        coverage = np.zeros((n,n),dtype=np.float32)
        for y in starts:
            for x in starts:
                coverage[y:y+width,x:x+width] += feather((y,x),width,n,overlap)
        self.assertTrue(np.all(coverage>0))

    def test_stored_tiles_recover_reference_fbp_for_both_raw_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for layout in ("frames","rows"):
                path = Path(tmp)/layout
                subprocess.run([sys.executable,str(HERE/"tiled.py"),str(path),
                                "--size","64","--tile","40","--overlap","16",
                                "--angles","128","--angle-block","16","--levels","2",
                                "--layout",layout],check=True,capture_output=True,text=True)
                config=json.loads((path/"manifest.json").read_text())
                self.assertTrue(config["complete"])
                self.assertLess(config["metrics"]["nrmse"],.35)
                a=np.load(path/"reconstruction-nexus.npy")
                b=np.load(path/"reconstruction-zarr.npy")
                np.testing.assert_array_equal(a,b)
                ideal=np.load(path/"ideal-projections.npy",mmap_mode="r")
                expected=iradon(ideal[:,32,:].T,theta=config["angles_deg"],output_size=64,
                                circle=True,filter_name="ramp",preserve_range=True)
                np.testing.assert_allclose(a[32],expected,atol=.003,rtol=0)
                for kind in ("nexus","zarr"):
                    with raw_reader(path,config,kind,[]) as (read,_):
                        for tile in range(4):
                            values=read(tile,(slice(0,128),slice(0,40),slice(0,40)),"test")
                            reference=np.load(path/"raw-source"/f"tile_{tile}.npy")
                            np.testing.assert_array_equal(values,reference)
                    with demo.open_levels(path/"volume",kind,2) as arrays:
                        for level,array in enumerate(arrays):
                            reference=np.load(path/"volume/source"/f"{level}.npy")
                            np.testing.assert_array_equal(array[:],reference)
                with (path/"read-trace.csv").open() as f:
                    patterns={row["pattern"] for row in csv.DictReader(f)}
                self.assertEqual(patterns,{"detector_frame","sinogram_band"})


if __name__=="__main__":
    unittest.main()
