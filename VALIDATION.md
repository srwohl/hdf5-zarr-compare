# Verification performed

Verified with Pixi 0.84.2 on linux-64, Python 3.12, NumPy 2.5.5,
h5py 3.16.0, Zarr-Python 3.4.2 and scikit-image 0.26.0.
Exact dependency builds are in pixi.lock.

- `pixi install`: installed the locked environment successfully.
- `pixi run test`: all five tests passed. These cover variation along each
  spatial axis, noise independent of generation-block size, exact box-mean
  pyramids, two-process compressed writes, fsync, moving an entire dataset,
  relative VDS paths and equality of source/NeXus/Zarr voxel values. The tiled
  tests cover complete detector coverage, two raw chunk layouts, exact tile
  data, consistent reconstructions and agreement with full-detector FBP.
- `pixi run build`: generated the default 256³ uint16 example with three
  resolution levels, matched physical partitions and no compression.
- `pixi run check`: every partition matched at every level in both formats.
- `pixi run bench`: completed all four read patterns at all three levels;
  matching selections returned matching voxel sums. Cache state was uncontrolled.
- `pixi run preview`: rendered and visually inspected the XY/XZ/YZ views.
- Interactive viewer callbacks were exercised using Matplotlib's headless
  backend: changing positions and selecting level 2 produced three 64² images.
  A native GUI window was not exercised in this environment.
- `pixi run tiled data/tiled-verified`: completed the 128³, 256-angle, four-tile
  pipeline. Both raw formats yielded identical float reconstructions. The
  central slice differed from ideal full-detector FBP by at most 0.000123 in
  relative intensity. Overall NRMSE against the original sharp voxel phantom
  was about 0.205, reflecting FBP discretization/interpolation as well as
  quantization; this is not a claim of exact reconstruction.
- A 64³, four-tile run with independent Poisson noise and row-oriented chunks
  also completed, with identical reconstructions from the two formats.
- The default tiled output occupied about 83 MiB locally, including retained
  simulation references and both reconstructed outputs.

This environment is not the target GPFS system. No GPFS throughput, physical
storage traffic, GPU transfer or cold-cache performance claim is made.
The GPFS commands in README.md are a reproducible starting configuration for
the user's system, whose known block size is 8 MiB.

## Beamline replay comparison

The `gpfs` profile completed on the local ext4 workspace with 256³ voxels,
256 angles, four detector tiles per angle, and 8 MiB image chunks. See
[data/beamline-comparison/report.md](data/beamline-comparison/report.md).
The four cases published all 1,024 frames each. Reconstructed voxels,
requested ROI and segmentation agreed across layouts and formats. The
segmentation measured 65,964 µm³ against the phantom's 65,968 µm³ and
achieved Dice 0.99997. Verified results took 34.37–40.08 seconds from the
first frame.

- The napari-ome-zarr 0.11.0 reader decoded the full-resolution image and
  label ROI in both OME-Zarr layouts. Napari 0.10.0 rendered the registered
  overlay in a virtual X display.
- A high-rate replay with a one-slot queue and drop policy recorded matching
  dropped IDs in the frame trace and publication markers and skipped incomplete
  reconstructions. The command generated a report and exited with status 1.
- A two-worker reverse-order run exercised independent raw partitions and
  volume processes. A synthetic counter command verified phase deltas and a
  fixture verified the bundled adapter's `mmpmon` parsing.
- `pixi run test` passed all seven tests, including reading a published
  partition while the next is absent and checking label pyramid alignment.

These runs used uncontrolled cache conditions. GPFS client and NSD telemetry
remain to be collected on the target filesystem.

## Parallel I/O

The default two-reader, two-writer run passed all four cases of the 8 MiB
preset on ext4. Each case published all 1,024 frames, and the reconstruction,
ROI and mask matched the single-worker reference exactly. Publication sustained
479–488 tile frames/s at a 500 frame/s source rate. Reader setup was reported
separately from the 0.046–0.095 s the reconstruction spent waiting for input.

The read-only profiles used the same 44 requests and fixed Zarr async
concurrency at four for each four-pass comparison. HDF5 acquisition chunks
went from 1.59 s to 0.93 s per pass with two readers and spatial bricks went
from 0.86 s to 0.53 s. The two HDF5 reader processes took about 0.79 s to start.
Zarr changed from 0.96 s to 0.91 s for acquisition chunks and from 0.62 s to
0.63 s for bricks. These are warmed local read timings.
