# Voxel I/O demo

A small experiment using the **same true 3D phantom** in NeXus/HDF5 and
**sharded OME-Zarr v3**. Ten ellipsoids vary in x, y and z: it is not a stack
of copies of a 2D image. Everything uses explicit **[z, y, x]** indexing.

The generator is a Shepp–Logan-style variant, not an exact reproduction of a
particular published phantom. `build` directly generates a volume for storage
tests. The new **`tiled` task also simulates detector acquisition and performs
tomographic reconstruction**. There is no measured camera data. Volume
intensities are clipped to [0,1] and quantized to uint16; the physical coordinate
convention is 1 µm per base voxel by default.

## Beamline replay experiment

Measure the time from the first detector frame to a verified measurement of
an internal feature of the phantom.

```bash
pixi run experiment --profile gpfs
```

The command creates a timestamped run directory under `data/`. On the beamline,
pass a fresh directory on the actual GPFS mount and use local scratch for the
precomputed detector frames and reconstruction intermediates.

```bash
pixi run experiment /gpfs/project/voxel-baseline --profile gpfs --scratch /local/scratch
```

Use `--profile local` for a quick 64³ workflow check. The GPFS preset uses
256³ voxels, 256 angles, four overlapping detector tiles, gzip level 1, and
8 MiB uncompressed full-resolution image chunks. The acquisition-oriented
image chunks are 64 × 256 × 256 and the spatial bricks are 128 × 128 × 256.
Both use 128 × 256 × 256 image partitions (16 MiB uncompressed). Raw chunks
have equal uncompressed size for both layouts, with a separate partition
size of 32 angles per tile. Both presets default to two application readers
and two writers, with Zarr's async.concurrency set to four.

| Case | Format | Layout |
|---|---|---|
| A | NeXus/HDF5 files and VDS | Whole detector frames and XY image slabs |
| B | NeXus/HDF5 files and VDS | Detector subregions and spatial image bricks |
| C | Sharded Zarr 3 raw data and sharded OME-Zarr results | Whole frames and XY slabs |
| D | Sharded Zarr 3 raw data and sharded OME-Zarr results | Detector subregions and image bricks |

The source emits individual detector images in angle-major, tile-minor
order. Simulation runs before the replay clock starts. `--fps` controls
the rate of tile images and `--queue-frames` bounds the handoff queue.
`--overflow block` applies backpressure to the replay clock.
`--overflow drop` keeps the source clock running and records missing
frame IDs. The writer also holds an assembly buffer and up to one in-flight
partition per worker. Each raw writer owns an angle block, and each image writer
owns a distinct part file or shard. Each `publications/*.json` marker makes that angle
range available after all its tile payloads have been written. The VDS is
created up front and consumers must respect the markers because an absent
source can otherwise look like zero-valued data. Raw reconstruction runs
after replay and skips cases with missing frames. Readers prefetch a bounded
number of detector row bands and consume them in order. HDF5 uses separate
processes and independent VDS handles for detector readers and raw writers.
Zarr uses application threads for both raw stages and schedules chunk
operations internally. The replay producer is
the integration point for a future detector transport such as tango-ucx.

Open the run's `report.md` for the case comparison and `comparison.png`
for the main plots. Each completed case also has a `result.png` that locates
the specimen and ROI in its overview, then overlays the full-resolution
segmentation on specimen coordinates. Timing includes full reconstruction and
image export before the first overview read. Verification compares stored
detector frames, calibration, voxels and labels, and checks all four reconstruction
and mask hashes. Segmentation thresholds the full-resolution ROI at 8.5% of
the quantized intensity range and keeps the component nearest the seed. Dice,
volume error, truth coverage and centroid error use the analytic ellipsoid. The command exits with status 1 if the four-case verification
fails and still writes a report for frame-loss or accuracy failures.

Both formats contain registered image and label pyramids. With an existing
napari environment that has the OME-Zarr plugin, open case C or D and enable the
dark-inclusion layer. Pass `--viewer-python /path/to/napari-env/bin/python` to
include the plugin's headless layer and ROI read check in the report.
[napari-ome-zarr usage](https://github.com/ome/napari-ome-zarr#usage) ·
[OME-Zarr 0.5 label metadata](https://ngff.openmicroscopy.org/0.5/#labels-md)

```bash
napari --plugin napari-ome-zarr /gpfs/project/voxel-baseline/C/phantom.ome.zarr
```

### Run on an HPC compute node

`run-hpc.sh` reads a TOML config and writes the results to GPFS. It uses
node-local scratch for staging and works with no home directory. It
reuses the `tango-ucx-benchmarks` Pixi deployment under `/tmp/$(id -un)/tango-ucx`
when available. It can also use Pixi on `PATH` or download the Linux x86_64
binary into scratch. The locked environment may need to download packages on
the first run.

From your compute-node shell, export `GPFS_PROJECT` once. `VOXEL_WORK` can
point to any executable node-local scratch directory. Its default is
`voxel-io` under `$SLURM_TMPDIR`, or `/tmp/$(id -un)/voxel-io`.

```bash
export GPFS_PROJECT=/gpfs/YOUR_PROJECT/YOUR_WRITABLE_DIR
export VOXEL_WORK="${VOXEL_WORK:-${SLURM_TMPDIR:-/tmp/$(id -un)}/voxel-io}"
(umask 077; mkdir -p "$VOXEL_WORK")
git clone https://github.com/srwohl/hdf5-zarr-compare.git "$VOXEL_WORK/src"
cd "$VOXEL_WORK/src"

./run-hpc.sh --config configs/beamline-gpfs.toml
```

Each invocation prints and creates a fresh result directory named after the
config, with a UTC timestamp and a unique suffix. It saves the input as
`config.toml` alongside `report.md` and `timeline.json`.

Copy a file from `configs/` to try a different experiment. Set `task` to
`experiment` or `write-matrix`. Keys in `[options]` match the task's CLI flags
without the leading `--`. Use arrays for `rates`, `writers`, and `geometry`.
Set switch flags such as `no-sync` or `compare-hdf5-writes` to `true` to enable
them. For a matrix based on a saved experiment, set `baseline` above
`[options]` and remove the synthetic detector options. Relative baseline paths
are resolved from the config's directory.

### Run a small write matrix

Replay the same verified detector frames at two source rates with one and
two writers. The default matrix runs all four cases twice and reverses
the trial order on the second pass. Each output is reread to check that
its frames match the source.

```bash
pixi run write-matrix data/parallel-gpfs-preset data/write-matrix-local
```

The report includes frames/s and decimal GB/s during replay, time from
writer setup to first and last publication, P95 publication latency, and
blocked time. Useful GB/s uses uncompressed published detector bytes.
Payload GB/s uses the bytes in the HDF5 part files or Zarr shards handed
to the filesystem. Setup-inclusive GB/s starts before writer setup.
If the counter hook returns client bytes written, the report also shows
GPFS client write GB/s. NSD counters measure server-side storage traffic.
The command writes per-trial and per-partition CSVs and a UTC timeline. The temporary
datasets are discarded after verification. Use `--rates`, `--writers`,
and `--repeats` to change the matrix.

On GPFS, put the matrix output on the filesystem under test and stage the
detector source on local scratch.

```bash
pixi run write-matrix data/parallel-gpfs-preset /gpfs/project/write-matrix \
  --scratch /local/scratch --gpfs-counter-command 'python gpfs_snapshot.py gpfs1'
```

### Collect detector-sized write measurements

Run the same four HDF5 and Zarr cases with synthetic 2048 × 2048 uint16 frames
and Poisson noise. Each frame is 8 MiB uncompressed, so 500 frames/s offers
4.19 GB/s. The command stages one set of 256 frames (2 GiB plus calibration)
on local scratch before timing and skips reconstruction. All four cases
replay and verify the same pixels, with gzip and fsync enabled.

```bash
./run-hpc.sh --config configs/detector-gzip.toml
```

Open the output `report.md` for received, published, and compressed payload
GB/s. Increase `--frames` in multiples of four for longer runs. The synthetic
mode takes a square detector size in pixels, with an even width.

To measure how gzip affects throughput, run the matching control with
`./run-hpc.sh --config configs/detector-uncompressed.toml`. In `samples.csv`,
compare `encode_seconds` (packing and any compression), `write_seconds`, and
`sync_seconds`. Writer times can overlap.

#### Sweep chunk and partition sizes on GPFS

Against the 8 MiB GPFS block-size reference, start with four geometry points
and hold source rate, workers, and fsync fixed. `--geometry` takes the inner
chunk and the HDF5 part file or Zarr shard size in uncompressed MiB.

| Chunk MiB | Partition MiB | Comparison |
|---:|---:|---|
| 8 | 32 | Current detector baseline |
| 8 | 128 | Larger files or shards at the same chunk size |
| 2 | 128 | Smaller chunks at the same file or shard size |
| 32 | 128 | Larger chunks at the same file or shard size |

```bash
./run-hpc.sh --config configs/detector-geometry.toml
```

This stages the source once and runs four cases per geometry. Compare useful
GB/s, raw file count and publication P95. The report shows the exact chunk
shapes. Each partition holds whole frames and each run holds whole partitions.
Larger partitions increase memory use per worker and take more frames to
assemble before publication. Record the GPFS block size for the mount you
test. The report includes payload bytes and optional GPFS client counters.

For follow-up passes, compare gzip on the best geometries, then vary writer
count using `writers = [2, 4, 8]` on a suitably sized CPU allocation. Use
`no-sync = true` in a separate pass to measure buffered publication. For
reconstruction read patterns, use the saved-experiment read profiling commands
below.

#### Compare HDF5 memory and file writes

Run a paired test with the same frames, layouts, chunk sizes, writer count,
and fsync. Each layout uses the HDF5 core driver in one trial and the regular
filesystem driver in the other. The filesystem path builds the part in a
temporary file on GPFS, then closes, fsyncs and renames it. Both paths use
the same publication markers. Two repeats reverse the trial order on the
second pass.

```bash
./run-hpc.sh --config configs/hdf5-write-paths.toml
```

Compare **Full-run GB/s** within A (acquisition chunks) and within B (bricks).
It includes final sync and worker teardown. In `samples.csv`, `write_seconds`
includes HDF5 file construction for the file path. For the memory path, look
at `encode_seconds` and `write_seconds` together. Writer times can overlap.

### See how parallel reads overlap

Replay the same recorded detector selections with one and two readers and
draw a timeline from their measured start, finish and delivery times.

```bash
pixi run read-timeline data/parallel-gpfs-preset --case A
```

The resulting `read-timeline-A.png` shows one lane per active reader and
a separate lane for batches delivered in request order. Purple shading marks
reader setup and yellow marks when two application read calls are active.
The clock starts before reader setup for HDF5 and Zarr. Use `--case C` for
the Zarr application-reader timeline. Zarr can also schedule chunk work
inside each read. The diagram shows the middle of three passes for each
setup and saves those events to CSV.

To put both formats on the same time axis, plot a matched read profile
using `--compare-formats`. Use case A for acquisition chunks or case B
for spatial bricks. The chart shows a paired pass close to the median gap.

```bash
pixi run read-timeline data/common-start-reads-two --compare-formats --case A
```

### Visualize the access patterns

The report includes `access-patterns.png`, a side-by-side view of recorded
detector-frame, sinogram-band and full-resolution ROI reads.
Orange marks the request, blue shows the chunks it intersects, and dashed
lines mark HDF5 part or Zarr shard boundaries. The display projects along x
and the annotations count chunks and bytes in all three dimensions.
The byte coverage is modeled from chunk geometry.

Regenerate the figure from a saved run without replaying the acquisition.

```bash
pixi run access-patterns data/parallel-gpfs-preset
```

### Profile detector reads

Replay the exact detector-frame and sinogram-band selections from a completed
comparison. The read profiler checks that all four cases return the same
bytes, then rotates the case order across four timed passes.

```bash
pixi run profile-reads data/beamline-comparison data/matched-read-profile
```

The report compares HDF5 and Zarr at matched chunk layouts. Each pass
starts in a fresh Python process and times from before opening the data
or launching reader workers through delivery of the first sinogram band
and last batch. The report also shows setup and post-dispatch timings.
The verification pass warms filesystem caches. To give every case the
same CPU budget, use a fixed allocation or launch the profile with
`taskset -c 0-3` using four CPUs from your allocation. The report records
the inherited CPU mask. The per-pass and per-request CSVs retain the
timings and Linux process read counters where available.

For a single-caller baseline, give each format one application reader and
leave Zarr's concurrency and thread-pool settings at their runtime defaults.

```bash
pixi run profile-reads data/parallel-gpfs-preset data/one-reader-native-zarr \
  --read-workers 1 --native-zarr
```

To compare the single-worker and parallel paths on the same stored data,
hold Zarr's internal concurrency fixed and use fresh output directories.

```bash
pixi run profile-reads data/beamline-comparison data/reads-one --read-workers 1 --zarr-concurrency 4
pixi run profile-reads data/beamline-comparison data/reads-two --read-workers 2 --zarr-concurrency 4
```

Use `pixi run experiment --profile gpfs --workers 1` for a full
single-application-worker baseline, or add `--zarr-concurrency 1` to run
one Zarr chunk operation at a time. `--read-workers` and `--write-workers`
can be set independently to change just one side of the pipeline. In the
experiment report, reader setup and foreground wait are separate from
the sum of read-call times because prefetched reads can overlap.

### Diagnose cached read cost

Compare the same detector requests with and without gzip on each format.
The diagnostic builds disposable uncompressed copies of the stored raw data,
checks the returned bytes, and profiles Zarr's gzip and local-fetch CPU
time in separate passes under the locked environment.

```bash
pixi run diagnose-reads data/parallel-gpfs-preset data/zarr-read-diagnosis
```

### Tune detector read granularity

Compare the recorded row band with a wider one and test sinogram-oriented
chunks at the same chunk and partition byte budget. The read tuner checks
that all configurations produce bit-identical stitched sinograms.

```bash
pixi run tune-reads data/parallel-gpfs-preset data/detector-read-tuning --row-band 64
```

Use the same filesystem for the run and the tuning output when comparing
storage performance. Temporary detector copies are written under the
tuning output directory. Wider bands use larger buffers and reduce how
often requests touch the same chunks. A sinogram-oriented chunk spans
more angles and fewer rows.

Apply those settings to the full experiment with:

```bash
pixi run experiment --profile gpfs --row-band 64 --detector-bricks sinogram
```

This sets the raw detector layout in B and D while keeping their image
bricks and the whole-frame raw layout in A and C.

### Profile the reconstruction

Rerun the reconstruction from a verified experiment's saved detector data.
This skips detector simulation and replay, and checks that the profiled run
reproduces the baseline reconstruction hash.

```bash
pixi run profile-experiment data/beamline-comparison data/reconstruction-profile --cases BC
```

The output includes a stage breakdown, the top function timings, and the
raw `.prof` files for Python's `pstats`. It uses one application reader so
the profiler can attribute reads to the calling thread. Use `--cases ABCD` to profile all
four cases, or `--row-band` with a fresh output directory to compare how many
detector reads reconstruction issues. The experiment's `timeline.json` still
covers the full run, including precomputing the detector data.

### GPFS telemetry and repeat runs

`timeline.json` records phase windows in UTC with wall and process CPU
times. Use those windows to align with node, filesystem, network and NSD
metrics. With access to cumulative client counters, add a snapshot command
that prints a flat JSON object of numeric values. Each measured phase records
both snapshots, sampling overhead and deltas. The bundled adapter reads
`mmpmon` filesystem counters when invoked in a context authorized to run it.

```bash
pixi run experiment /gpfs/project/voxel-counters --profile gpfs --scratch /local/scratch \
  --gpfs-counter-command 'python gpfs_snapshot.py gpfs1'
```

Replace `gpfs1` with the Storage Scale filesystem name. The adapter's read
counter includes cache-served data. Correlate with server-side metrics when
making claims about storage traffic. The report also separates returned
bytes, Linux process read counters, full-chunk coverage and a DEFLATE
encode/decode probe on one representative in-memory image chunk.
[IBM mmpmon reference](https://www.ibm.com/docs/en/storage-scale/5.2.3?topic=reference-mmpmon-command) ·
[IBM Storage Scale performance metrics](https://www.ibm.com/docs/en/storage-scale-system/storage-scale-software/6.2.3?topic=gui-performance-metrics-available-in)

Repeat in fresh directories and change one setting per comparison.

| Setting | Measurement |
|---|---|
| `--angle-block 64` | File/shard footprint, queue wait and publication P95 as raw partitions grow |
| `--read-workers 1` / `--read-workers 4` | Detector prefetch and read throughput with identical writers |
| `--write-workers 1` / `--write-workers 4` | Raw publication and image export with identical readers |
| `--compression none` | Change in CPU time and storage traffic at the same geometry |
| `--size 512` | Larger reconstruction and more full-resolution image partitions with 8 MiB chunks |
| `--partition-factor 2` | Effect of larger image files or shards with fixed chunk shapes |
| `--case-order DCBA` | Sensitivity to run order and cache warming |

Record the actual mount and cache conditions. Sizes of 512³ and above increase
detector simulation, reconstruction and scratch requirements. The default
GPFS preset can also run on a local filesystem and reports the actual
filesystem it used.

## Tiled tomography

```bash
pixi install
pixi run tiled
pixi run view data/tiled/volume --format zarr
pixi run view data/tiled/volume --format nexus
```

Default geometry: a **128³ ground-truth phantom**, **256 angles over 180°**,
and a **2×2 grid of 72×72 detector fields with 16-pixel overlap**. The fields
cover a 128×128 panoramic detector. At each angle this records 20,736 pixels
for a 16,384-pixel panorama: 26.56% extra pixels due to overlaps.

The workflow is:

1. Generate the true 3D object.
2. Forward-project its slices using a parallel-beam Radon transform.
3. Crop overlapping detector fields at known positions. Apply Beer–Lambert
   transmission, tile-specific flat illumination and a dark offset to produce
   **uint16 raw detector counts**. `--poisson` adds independent photon noise
   for each tile and angle; noiseless quantized counts are the default.
4. Write **identical raw tile arrays** as NeXus/HDF5 parts + VDS and sharded
   Zarr. Each physical part contains an angle block of a single tile.
5. Read those stored data, apply flat/dark correction, feather-blend overlapping
   transmission values, take -log, and reconstruct with ramp-filtered FBP.
   Stitching happens in detector-row bands without writing a second complete
   stitched projection stack. No reconstruction reads directly from the ideal
   simulation source.
6. Compare both reconstructions with one another and with ground truth.
   Export the reconstructed volume and its averaged pyramid to both formats.

All offsets are **known integer positions**. This initial model does not
estimate registration, rotation centre, drift or subpixel stage errors. It
implements projection stitching, not reconstruction of isolated truncated
tiles followed by volume stitching. Feather blending is a simple baseline,
not Tomosaic's more sophisticated multiscale blending.

Raw axes are **[angle, detector_y, detector_x]**. They are stored in a custom
raw **Zarr** acquisition collection; this is deliberately **not labelled an
OME spatial image**. OME-Zarr metadata describes the reconstructed [z,y,x]
volume. Raw NeXus uses general NXentry/NXdata conventions and does not claim
NXtomo application-definition conformance. The raw coordinates assume 1 µm
sampling and the scikit-image centre-of-rotation convention.

```text
data/tiled/
  truth/0.npy                 # ground-truth uint16 voxels
  ideal-projections.npy       # simulation reference: float line integrals
  raw-source/                 # authoritative tile counts plus flat/dark arrays
  raw-nexus/raw.nxs           # NeXus master referencing tile/angle-block files
  raw.zarr/                   # sharded raw acquisition collection
  reconstruction-nexus.npy    # float FBP from the HDF5 raw tiles
  reconstruction-zarr.npy     # float FBP from the Zarr raw tiles
  read-trace.csv              # measured frame and sinogram-band read requests
  tiled-overview.png          # object, fields, raw tile, sinogram, reconstruction, error
  manifest.json               # geometry, simulation assumptions, timings, errors
  volume/                    # reconstructed volume in both formats; standard viewer
```

The tiled demo is small and uses CPU reference algorithms; it is a functional
pipeline example, not a GPFS saturation benchmark. See the larger 8 MiB-chunk
volume experiment below for storage scaling. `manifest.json` reports float
reconstruction error **before clipping/uint16 export**. FBP has interpolation
and discretization error even with ideal data; the additional comparison to
full-detector FBP separates that from tile stitching error.

To compare raw frame chunks with detector-row chunks, use separate runs:

```bash
pixi run tiled data/tiled-frames --layout frames
pixi run tiled data/tiled-rows --layout rows
pixi run tiled data/tiled-noisy --poisson
```

This makes the change in access explicit: acquisition writes frames in angle
blocks, whereas reconstruction requests selected detector rows across all
angles and intersecting tiles. `read-trace.csv` records those actual requests;
it does not claim to measure physical GPFS traffic. Tile sources are generated
once outside the storage timers. With fixed geometry/seed, raw counts do not
depend on the chunk layout. `frames` and `rows` use different chunk shapes and
byte sizes; this is a workflow comparison, not a shape-only controlled test.
Known offsets and simple flat fields keep the two formats numerically comparable.

## Start

Install Pixi from https://pixi.sh/latest/installation/ if needed. Then:

```bash
pixi install
pixi run build
pixi run check
pixi run preview
pixi run bench
pixi run view
```

The default volume is **256³ = 32 MiB**, plus two smaller levels, so it is
easy to run on a laptop. The interactive viewer needs a graphical display;
`preview` works headlessly and writes a PNG. The viewer has three position
sliders and a resolution-level slider; it reads from the selected storage
format on demand. View HDF5 with:

```bash
pixi run view --format nexus
```

The supplied lockfile records the Linux environment used for verification.
Generated output directories must be new; builds refuse to overwrite data.

## What is written

```text
data/demo/
  manifest.json             # geometry, ellipsoids, versions, construction timings
  source/0.npy              # authoritative voxels, generated in small z slabs
  source/1.npy              # rounded 2×2×2 box mean of level 0
  source/2.npy              # rounded 2×2×2 box mean of level 1
  nexus/phantom.nxs          # NeXus master with a VDS per resolution
  nexus/level_0/part_*.nxs   # one independent file per matching Zarr shard extent
  nexus/level_1/part_*.nxs
  nexus/level_2/part_*.nxs
  phantom.ome.zarr/          # OME-NGFF 0.5 metadata on Zarr v3, genuinely sharded
  preview-zarr.png
  reads.csv
```

Both formats contain identical voxels and actual averaged pyramid levels.
`check` compares every partition at every level against the source arrays.
Relative VDS source paths allow the whole output directory to be moved.

The NeXus files use NXentry, NXdata, NXcollection and NXprocess. Synthetic
provenance is explicit. The file does **not** claim NXtomoproc conformance or
invent a measured acquisition. Each level has axes and units; the association
between pyramid levels is a small demo convention, whereas the OME output
uses standardized `multiscales` metadata. Voxel-centre translations match
the averaged pyramid levels in both representations.

The source arrays are retained so validation is independent of either output.
They add one source copy to the storage budget. Generation and downsampling
are timed separately from format construction. A write task reads one source
partition into a contiguous buffer; peak memory includes that buffer, codecs
and filesystem caches. `--workers` multiplies that memory requirement.

## GPFS experiment: 8 MiB blocks

The known GPFS block size is recorded as context, not configured by this code.
This 1024³ uint16 volume is **2 GiB per full-resolution representation**.
Two outputs, their source, and pyramids need about 6.9 GiB uncompressed.

Start with **8 MiB inner chunks** and **512 MiB spatial partitions**. Each
file/shard contains 64 full-resolution chunks. There are four full-resolution
partitions, so up to four writers can work on that level simultaneously.

```bash
# Spatial bricks: 128 × 128 × 256 × 2 bytes = 8 MiB
pixi run build /your/gpfs/path/bricks \
  --size 1024 --levels 4 \
  --chunks 128,128,256 --shards 256,1024,1024 \
  --workers 2 --sync

# XY slabs: 4 × 1024 × 1024 × 2 bytes = 8 MiB
pixi run build /your/gpfs/path/slabs \
  --size 1024 --levels 4 \
  --chunks 4,1024,1024 --shards 256,1024,1024 \
  --workers 2 --sync

pixi run check /your/gpfs/path/bricks
pixi run bench /your/gpfs/path/bricks --reads 32
pixi run bench /your/gpfs/path/slabs --reads 32
pixi run view /your/gpfs/path/bricks --format zarr
```

Shapes shrink only when a lower-resolution array is smaller than a configured
dimension; the actual geometry at each level is derived in `geometry()`.
Chunks remain separate codec units inside each Zarr shard. Every worker owns
a complete shard or independent HDF5 file. Metadata creation is serial.
The HDF5 master is assembled after data writers finish. This is not a shared
parallel-HDF5 or subfiling-VFD experiment and needs no MPI.

For a compressed comparison, build fresh directories with
`--compression gzip --noise 0.003`. Both use compression level 1, no shuffle;
their codec wrappers/implementations are not byte-identical. The code requests
explicit empty-chunk writing, but a sharding codec can still encode fill-only
inner chunks implicitly. Compare actual stored sizes; a clean phantom's large
zero background can otherwise give a misleading storage advantage.
Repeat with `--write-order zarr,nexus` to expose source-cache/order effects.

## Read measurements

`bench` replays the same seeded XY, XZ, YZ and 3D-ROI requests for both
formats at every level. Singleton slice dimensions are retained. CSV records
selection bounds, elapsed read time, output bytes, a sum, intersected chunks,
and intersected files/shards. Checksums are outside the read timer.

`full_chunk_model_bytes` is **theoretical uncompressed full-chunk coverage**,
not bytes read from GPFS or guaranteed decoder work. With uncompressed data,
the libraries may read portions of chunks. Compression, coalescing, caching
and prefetching change actual traffic. HDF5 raw-data caching is requested off
on the master; that does not establish cache settings on every VDS source.
Zarr uses its defaults. No equivalence of internal concurrency is claimed.

All reads are labelled **cache_state=uncontrolled**. Fresh processes do not
flush GPFS, OS or storage caches. The newly built data will often be warm.
Opening is timed separately; checks and previews also warm data. CSV timings
are local observations, not a filesystem bandwidth claim. Use GPFS client
and NSD counters to distinguish requests from actual storage traffic.

Construction time includes reading the source, encoding, file creation, worker
startup and output writing, but excludes generating the source/pyramids.
Per-task times are summed worker times, not wall-clock throughput.
`--sync` records a separate fsync pass over data files and directories.
Without it, close/write timings do not measure durable completion.

For scaling, choose a dataset larger than the relevant caches and increase
writer count and partition size separately. More workers than available
partitions cannot improve that level's parallelism. Keep a record of storage
load and repeat runs; this prototype does not change GPFS configuration.

## Files to read

- `experiment.py`, `replay.py`, and `feature.py`: the comparison runner,
  controlled detector replay, and scientific ROI workflow.
- `phantom.py`: ellipsoid geometry, slab generation and box-mean pyramids.
- `demo.py`: the two writers, NeXus mapping, reader, trace and simple viewer.
- `tiled.py`: tiled detector simulation, flat/dark correction, blending and FBP.
- `test_demo.py`: true 3D variation, deterministic noise, downsampling,
  parallel compressed writes, portable VDS paths and exact format agreement.
- `test_tiled.py`: coverage, both raw chunk layouts, exact stored tile values,
  reconstruction consistency and agreement with full-detector FBP.

```bash
pixi run test
```

References: [Zarr sharding](https://zarr.readthedocs.io/en/stable/user-guide/arrays/),
[OME-NGFF 0.5](https://ngff.openmicroscopy.org/0.5/),
[NeXus NXdata](https://manual.nexusformat.org/classes/base_classes/NXdata.html),
[HDF5 VDS](https://docs.h5py.org/en/stable/vds.html),
[TomoPy phantom background](https://tomopy.readthedocs.io/en/stable/api/tomopy.misc.phantom.html).
Tiled-workflow references:
[Tomosaic](https://journals.iucr.org/s/issues/2018/05/00/il5010/index.html),
[Tile CLI](https://tile.readthedocs.io/en/latest/source/usage.html),
[scikit-image Radon/FBP](https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html).
