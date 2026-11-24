"""Locate stored raw chunks in their files and count those that do not fit filesystem blocks."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import zarr
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec

EMPTY = 2**64 - 1


def hdf5_alignment(requested, chunks):
    """h5py File options that align raw chunks for a requested byte interval.

    Chunks that divide the interval are aligned to their own size, so they pack
    into a block without crossing it. Objects smaller than the threshold, such
    as HDF5 metadata and most gzip chunks, keep HDF5's default placement.
    """
    if not requested:
        return {}
    chunk_bytes = 2 * int(np.prod(chunks))
    interval = chunk_bytes if requested % chunk_bytes == 0 else requested
    return dict(alignment_threshold=interval, alignment_interval=interval)


def hdf5_extents(path):
    with h5py.File(path, "r") as handle:
        chunks = handle["entry/data/data"].id
        return [(info.byte_offset, info.size) for info in map(chunks.get_chunk_info, range(chunks.get_num_chunks()))]


def zarr_extents(array_path):
    """Read each shard's index directly to get chunk offsets inside the shard file."""
    array = zarr.open_array(str(array_path), mode="r")
    codec = array.metadata.codecs[0]
    if (not isinstance(codec, ShardingCodec) or len(codec.index_codecs) != 2
            or not isinstance(codec.index_codecs[0], BytesCodec)
            or not isinstance(codec.index_codecs[1], Crc32cCodec)):
        raise ValueError(f"{array_path}: expected a sharded array with a bytes + crc32c index")
    count = int(np.prod([s // c for s, c in zip(array.metadata.chunk_grid.chunk_shape, codec.chunk_shape)]))
    index_bytes = count * 16 + 4
    extents = {}
    for shard in sorted(path for path in (array_path / "c").rglob("*") if path.is_file()):
        with shard.open("rb") as handle:
            size = handle.seek(0, 2)
            location = getattr(codec.index_location, "value", codec.index_location)
            start = 0 if location == "start" else size - index_bytes
            handle.seek(start)
            index = np.frombuffer(handle.read(index_bytes - 4), "<u8").reshape(count, 2)
        data_start, data_stop = (index_bytes, size) if start == 0 else (0, start)
        present = [(int(offset), int(length)) for offset, length in index if offset != EMPTY]
        if any(offset < data_start or offset + length > data_stop for offset, length in present):
            raise ValueError(f"{shard}: shard index does not match the file size")
        extents[shard] = present
    return extents


def raw_extents(root, kind):
    """Map each raw part file or shard below a RawWriter root to its chunk extents."""
    if kind == "nexus":
        return {path: hdf5_extents(path) for path in sorted(root.glob("tile_*/part_*.nxs"))}
    extents = {}
    for array in sorted(root.glob("tile_*/data")):
        extents.update(zarr_extents(array))
    return extents


def misaligned(offset, size, block):
    """A chunk is misaligned if it touches more blocks than its size requires."""
    if size >= block:
        return offset % block != 0
    return offset // block != (offset + size - 1) // block


def summarize(extents, block):
    chunks = [extent for values in extents.values() for extent in values]
    bad = sum(misaligned(offset, size, block) for offset, size in chunks)
    first = sorted({values[0][0] % block for values in extents.values() if values})
    return dict(layout_block_bytes=block, layout_files=len(extents), layout_chunks=len(chunks),
                layout_misaligned_chunks=bad, layout_first_chunk_offsets_mod_block=first[:8])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="raw-nexus or raw.zarr directory")
    parser.add_argument("--block", type=int, help="block bytes (default: st_blksize of root)")
    args = parser.parse_args()
    kind = "zarr" if args.root.name.endswith(".zarr") else "nexus"
    block = args.block or args.root.stat().st_blksize
    print(json.dumps(summarize(raw_extents(args.root, kind), block), indent=2))


if __name__ == "__main__":
    main()
