"""Checks for chunk placement, filesystem geometry and Darshan summaries."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import chunk_layout
import darshan_summary
import experiment
import replay
import run_config
import strace_summary
import tiled

BLOCK = 8192  # One (1, 64, 64) uint16 chunk.


class ChunkLayoutTests(unittest.TestCase):
    def write(self, root, kind, **settings):
        config = dict(size=64, tile=64, angle_block=4, angles_deg=list(range(8)),
                      tile_origins_yx_px=[(0, 0)], raw_chunks=(1, 64, 64),
                      compression="none", sync=False, **settings)
        source = root / "source"
        source.mkdir(exist_ok=True)
        np.save(source / "flat_0.npy", np.full((64, 64), 25, dtype=np.float32))
        np.save(source / "dark_0.npy", np.zeros((64, 64), dtype=np.float32))
        output = root / f"{kind}-{len(list(root.iterdir()))}"
        output.mkdir()
        frames = np.arange(4 * 64 * 64, dtype="<u2").reshape(4, 64, 64)
        writer = replay.RawWriter(output, config, kind, source)
        writer.publish(0, [frames], [])
        with tiled.raw_reader(output, config, kind, []) as (read, _):
            np.testing.assert_array_equal(read(0, (slice(0, 4), slice(0, 64), slice(0, 64)), "check"), frames)
        return chunk_layout.summarize(chunk_layout.raw_extents(writer.root_path, kind), BLOCK)

    def test_hdf5_chunks_follow_the_requested_alignment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ("memory", "file"):
                with self.subTest(path=path):
                    default = self.write(root, "nexus", hdf5_write_path=path)
                    aligned = self.write(root, "nexus", hdf5_write_path=path, hdf5_alignment=4 * BLOCK)
                    self.assertEqual(default["layout_chunks"], 4)
                    self.assertEqual(default["layout_misaligned_chunks"], 4)
                    self.assertEqual(aligned["layout_misaligned_chunks"], 0)
                    self.assertEqual(aligned["layout_first_chunk_offsets_mod_block"], [0])

    def test_uncompressed_zarr_shards_are_block_aligned(self):
        with TemporaryDirectory() as directory:
            for path in ("memory", "file"):
                with self.subTest(path=path):
                    layout = self.write(Path(directory), "zarr", zarr_write_path=path)
                    self.assertEqual((layout["layout_files"], layout["layout_chunks"]), (1, 4))
                    self.assertEqual(layout["layout_misaligned_chunks"], 0)

    def test_misalignment_rule(self):
        self.assertFalse(chunk_layout.misaligned(0, 2 * BLOCK, BLOCK))
        self.assertTrue(chunk_layout.misaligned(4536, BLOCK, BLOCK))
        self.assertFalse(chunk_layout.misaligned(BLOCK // 2, BLOCK // 2, BLOCK))
        self.assertTrue(chunk_layout.misaligned(BLOCK - 1, 2, BLOCK))
        self.assertEqual(chunk_layout.hdf5_alignment(8 << 20, (1, 512, 1024)),
                         dict(alignment_threshold=1 << 20, alignment_interval=1 << 20))
        self.assertEqual(chunk_layout.hdf5_alignment(8 << 20, (4, 1024, 1024))["alignment_interval"], 8 << 20)


class StorageScaleTests(unittest.TestCase):
    def test_mmlsfs_parseable_output(self):
        text = ("mmlsfs::HEADER:version:reserved:reserved:deviceName:fieldName:data:remarks:\n"
                "mmlsfs::0:1:::gpfs1:blockSize:8388608::\n"
                "mmlsfs::0:1:::gpfs1:minFragmentSize:16384::\n"
                "mmlsfs::0:1:::gpfs1:blockAllocationType:scatter::\n")
        self.assertEqual(experiment.parse_mmlsfs(text),
                         dict(blockSize=8388608, minFragmentSize=16384, blockAllocationType="scatter"))

    def test_mount_info_reports_block_geometry(self):
        info = experiment.mount_info(Path.cwd())
        self.assertGreater(info["st_blksize"], 0)
        self.assertGreater(info["f_bsize"], 0)


class DarshanTests(unittest.TestCase):
    def test_counters_are_grouped_by_trial(self):
        trial = "/gpfs/p/out/pass1-8-32mib-500fps-2writers-A-aligned-ab3_x9z0/raw-nexus/tile_0/part_0.nxs"
        other = "/gpfs/p/out/pass1-8-32mib-500fps-2writers-C-k2m4n6p8/raw.zarr/tile_0/data/c/0/0/0"
        lines = ["# darshan log version: 3.41",
                 f"POSIX\t-1\t1\tPOSIX_OPENS\t2\t{trial}\t/gpfs\tgpfs",
                 f"POSIX\t-1\t1\tPOSIX_WRITES\t4\t{trial}\t/gpfs\tgpfs",
                 f"POSIX\t-1\t1\tPOSIX_BYTES_WRITTEN\t8388608\t{trial}\t/gpfs\tgpfs",
                 f"POSIX\t-1\t1\tPOSIX_FILE_NOT_ALIGNED\t3\t{trial}\t/gpfs\tgpfs",
                 f"POSIX\t-1\t1\tPOSIX_FILE_ALIGNMENT\t8388608\t{trial}\t/gpfs\tgpfs",
                 f"POSIX\t-1\t1\tPOSIX_SIZE_WRITE_1M_4M\t4\t{trial}\t/gpfs\tgpfs",
                 f"POSIX\t-1\t1\tPOSIX_F_META_TIME\t-1\t{trial}\t/gpfs\tgpfs",
                 f"POSIX\t-1\t2\tPOSIX_WRITES\t1\t{other}\t/gpfs\tgpfs",
                 f"STDIO\t-1\t3\tSTDIO_WRITES\t9\t{trial}\t/gpfs\tgpfs",
                 "POSIX\t-1\t4\tPOSIX_WRITES\t7\t/tmp/unrelated\t/\text4"]
        rows = {row["group"]: row for row in darshan_summary.summarize(darshan_summary.parse_records(lines))}
        self.assertEqual(set(rows), {"pass1-8-32mib-500fps-2writers-A-aligned", "pass1-8-32mib-500fps-2writers-C"})
        row = rows["pass1-8-32mib-500fps-2writers-A-aligned"]
        self.assertEqual((row["opens"], row["writes"], row["file_not_aligned"]), (2, 4, 3))
        self.assertEqual((row["mean_write_bytes"], row["file_alignment_bytes"], row["f_meta_time"]),
                         (2097152, "8388608", 0))
        self.assertIn("| pass1-8-32mib-500fps-2writers-A-aligned | gpfs | 1 | 2 | 4 | 2.00 | 3 |",
                      darshan_summary.markdown(rows.values()))

    def test_preload_environment(self):
        with TemporaryDirectory() as directory:
            logs = Path(directory) / "logs"
            environment = run_config.darshan_environment("/opt/libdarshan.so", logs,
                                                         [Path("/tmp/work"), Path("/env")], True)
            self.assertTrue(environment["LD_PRELOAD"].startswith("/opt/libdarshan.so"))
            self.assertEqual(environment["DARSHAN_EXCLUDE_DIRS"], "/tmp/work,/env")
            self.assertEqual((environment["DARSHAN_ENABLE_NONMPI"], environment["DXT_ENABLE_IO_TRACE"]), ("1", "1"))
            self.assertEqual(len(list(logs.glob("*/*/*"))), 1)


class StraceTests(unittest.TestCase):
    def test_calls_are_grouped_with_offsets(self):
        d = "/gpfs/p/out/pass1-8-32mib-500fps-2writers-A-file-default-ab3_x9z0/raw-nexus/tile_0"
        lines = [
            f'14:59:35.299217 mkdir("{d}", 0777) = 0 <0.000025>',
            f'14:59:35.299257 openat(AT_FDCWD</tmp>, "{d}/a.partial", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC, 0666) '
            f'= 3<{d}/a.partial> <0.000014>',
            f'14:59:35.299315 lseek(3<{d}/a.partial>, 0, SEEK_CUR) = 0 <0.000005>',
            f'14:59:35.301200 write(3<{d}/a.partial>, ""..., 8192) = 8192 <0.001028>',
            f'14:59:35.301300 write(3<{d}/a.partial>, ""..., 100) = 100 <0.000028>',
            f'14:59:35.302588 fsync(3<{d}/a.partial>) = 0 <0.003302>',
            f'14:59:35.305968 rename("{d}/a.partial", "{d}/a") = 0 <0.000023>',
            f'14:59:35.306014 openat(AT_FDCWD</tmp>, "{d}", O_RDONLY|O_CLOEXEC|O_DIRECTORY) = 3<{d}> <0.000010>',
            f'14:59:35.306047 fsync(3<{d}>) = 0 <0.000834>',
            f'14:59:35.307080 openat(AT_FDCWD</tmp>, "{d}/b.h5", O_RDWR) = -1 ENOENT (No such file or directory) '
            '<0.000009>',
            f'14:59:35.309767 pwrite64(4<{d}/b.h5>, ""..., 8192, 4536) = 8192 <0.000381>',
            f'14:59:35.309800 pwritev2(4<{d}/b.h5>, [{{iov_base=""..., iov_len=8192}}], 1, 16384, 0) = 8192 <0.0001>',
            '14:59:35.310000 write(5</tmp/elsewhere>, ""..., 10) = 10 <0.000001>',
        ]
        events = list(strace_summary.parse_trace(lines, "/gpfs/p/out"))
        self.assertEqual([event["offset"] for event in events if event["syscall"] in strace_summary.WRITES],
                         [0, 8192, 4536, 16384])
        [row] = strace_summary.summarize(events, 8192)
        self.assertEqual(row["group"], "pass1-8-32mib-500fps-2writers-A-file-default")
        self.assertEqual((row["opens"], row["creates"], row["failed_opens"], row["mkdirs"]), (2, 1, 1, 1))
        self.assertEqual((row["writes"], row["writes_unaligned_start"], row["writes_partial_block"]), (4, 1, 2))
        self.assertEqual((row["file_fsyncs"], row["directory_fsyncs"], row["renames"]), (1, 1, 1))
        self.assertIn("| pass1-8-32mib-500fps-2writers-A-file-default |", strace_summary.markdown([row], 8192))


if __name__ == "__main__":
    unittest.main()
